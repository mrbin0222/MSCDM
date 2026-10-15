#!/usr/bin/env python3
"""Support latent factor probe capacity."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

_EXP4_ROOT = Path(__file__).resolve().parents[3]
if str(_EXP4_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXP4_ROOT))

from mscdms.checkpoint import config_from_checkpoint, load_checkpoint
from mscdms.control_evaluator import classification_metrics, sha256_file
from mscdms.dataset import CellMicroscopyDataset
from mscdms.latent_vae import LatentAutoEncoder


BRANCH_SIZES = {
    "low": (128, 128),
    "mid": (256, 256),
    "high": (512, 512),
}
FORMAL_STATE_GROUPS = ("bbbc021", "dynamicnuclearnet")
IDENTITY_RE = re.compile(
    r"^(?P<entity>.+?) with (?P<morphology>elongated|oval|round) "
    r"morphology and (?P<area>large|medium|small) area$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/mscdm_h200_v4_isolated_full/checkpoint_best.pt"),
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--manifest_name",
        default="control_metrics_v2/evaluator_manifest.jsonl",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path(
            "data/processed/"
            "control_metrics_v2/label_vocab.json"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/v4_latent_factor_probe_capacity_audit_v1"),
    )
    parser.add_argument("--extract_batch_size", type=int, default=16)
    parser.add_argument("--probe_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--grid_size", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--population_size", type=int, default=16)
    parser.add_argument("--population_reps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16", "fp16"),
        default="bf16",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--max_samples_per_split",
        type=int,
        default=0,
        help="Smoke-test limit. Zero means the complete split.",
    )
    return parser.parse_args()


def observable_identity(label: str) -> str:
    match = IDENTITY_RE.fullmatch(str(label).strip().lower())
    if not match:
        raise ValueError(f"cannot parse identity label: {label!r}")
    return f"{match.group('entity')}|{match.group('morphology')}"


def observable_identity_vocab(full_labels: list[str]) -> tuple[list[str], list[int]]:
    labels: list[str] = []
    label_to_id: dict[str, int] = {}
    full_to_observable: list[int] = []
    for full_label in full_labels:
        label = observable_identity(full_label)
        if label not in label_to_id:
            label_to_id[label] = len(labels)
            labels.append(label)
        full_to_observable.append(label_to_id[label])
    if len(labels) != 12:
        raise ValueError(f"expected 12 observable identity classes, got {len(labels)}")
    return labels, full_to_observable


def latent_descriptor(latent: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Differentiable spatial descriptor used by the independent probes."""
    if latent.ndim != 4:
        raise ValueError(f"expected BCHW latent, got {tuple(latent.shape)}")
    mean_grid = F.adaptive_avg_pool2d(latent, (grid_size, grid_size))
    second_grid = F.adaptive_avg_pool2d(
        latent.float().square(),
        (grid_size, grid_size),
    )
    std_grid = (
        second_grid - mean_grid.float().square()
    ).clamp_min(0).sqrt()
    return torch.cat([mean_grid.float(), std_grid], dim=1).flatten(1)


def load_branch_vae(
    branch: str,
    checkpoint: dict[str, Any],
    cfg: Any,
    device: torch.device,
) -> LatentAutoEncoder:
    vae = LatentAutoEncoder(
        in_ch=3,
        z_ch=cfg.latent_channels,
        base_ch=cfg.vae_base_ch,
        ch_mult=cfg.vae_ch_mult,
        num_heads=cfg.num_heads,
        gradient_checkpointing=False,
    )
    prefix = f"vae_{branch}."
    state = {
        key[len(prefix) :]: value
        for key, value in checkpoint["model"].items()
        if key.startswith(prefix)
    }
    if not state:
        raise KeyError(f"checkpoint has no tensors with prefix {prefix!r}")
    vae.load_state_dict(state)
    return vae.to(device).eval()


class LabeledRealDataset(Dataset[tuple[Any, ...]]):
    def __init__(
        self,
        *,
        data_dir: Path,
        manifest_name: str,
        split: str,
        full_to_observable: list[int],
        max_samples: int,
    ) -> None:
        self.base = CellMicroscopyDataset(
            data_dir,
            split=split,
            image_size=512,
            normalize_mode="percentile",
            manifest_name=manifest_name,
        )
        self.full_to_observable = full_to_observable
        self.state_group_to_id = {
            group: index for index, group in enumerate(FORMAL_STATE_GROUPS)
        }
        self.indices = list(range(len(self.base)))
        if max_samples > 0:
            self.indices = self.indices[:max_samples]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[Any, ...]:
        index = self.indices[position]
        image = self.base[index][0]
        row = self.base.records[index]
        full_identity = int(row["identity_class_id"])
        state_group = str(row["state_group"])
        state_group_id = self.state_group_to_id.get(state_group, -1)
        state_id = row.get("state_class_id")
        return (
            image,
            torch.tensor(
                self.full_to_observable[full_identity],
                dtype=torch.long,
            ),
            torch.tensor(int(row["background_class_id"]), dtype=torch.long),
            torch.tensor(state_group_id, dtype=torch.long),
            torch.tensor(-1 if state_id is None else int(state_id), dtype=torch.long),
            str(row["sample_id"]),
        )


@torch.inference_mode()
def extract_split(
    *,
    branch: str,
    vae: LatentAutoEncoder,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    grid_size: int,
    checkpoint_sha256: str,
    split: str,
) -> dict[str, Any]:
    amp_enabled = device.type == "cuda" and precision != "fp32"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    features: list[torch.Tensor] = []
    identities: list[torch.Tensor] = []
    backgrounds: list[torch.Tensor] = []
    state_groups: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    sample_ids: list[str] = []
    target_size = BRANCH_SIZES[branch]

    for batch_index, batch in enumerate(loader, start=1):
        images, identity, background, state_group, state, batch_ids = batch
        images = images.to(device, non_blocking=True)
        if images.shape[-2:] != target_size:
            images = F.interpolate(
                images,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        with torch.amp.autocast(
            "cuda",
            enabled=amp_enabled,
            dtype=amp_dtype,
        ):
            mean, _ = vae.encode(images)
        descriptor = latent_descriptor(mean, grid_size)
        features.append(descriptor.cpu().to(torch.float16))
        identities.append(identity)
        backgrounds.append(background)
        state_groups.append(state_group)
        states.append(state)
        sample_ids.extend(str(value) for value in batch_ids)
        if batch_index == 1 or batch_index % 50 == 0 or batch_index == len(loader):
            print(
                f"extract {branch}/{split}: "
                f"{min(batch_index * loader.batch_size, len(loader.dataset))}/"
                f"{len(loader.dataset)}",
                flush=True,
            )

    feature_tensor = torch.cat(features)
    if not torch.isfinite(feature_tensor.float()).all():
        raise ValueError(f"{branch}/{split} contains non-finite features")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{branch}/{split} contains duplicate sample IDs")
    return {
        "schema": "exp4.latent_factor_probe.cache.v1",
        "branch": branch,
        "split": split,
        "checkpoint_sha256": checkpoint_sha256,
        "grid_size": int(grid_size),
        "features": feature_tensor,
        "identity": torch.cat(identities),
        "background": torch.cat(backgrounds),
        "state_group": torch.cat(state_groups),
        "state": torch.cat(states),
        "sample_ids": sample_ids,
    }


class FeatureProbe(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.config = {
            "feature_dim": int(feature_dim),
            "hidden_dim": int(hidden_dim),
            "num_classes": int(num_classes),
            "dropout": float(dropout),
        }
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def balanced_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    if (counts <= 0).any():
        missing = torch.nonzero(counts <= 0).flatten().tolist()
        raise ValueError(f"training labels are missing classes: {missing}")
    weights = counts.rsqrt()
    return weights / weights.mean()


@torch.inference_mode()
def predict(
    model: nn.Module,
    features: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    logits: list[torch.Tensor] = []
    for start in range(0, len(features), batch_size):
        logits.append(model(features[start : start + batch_size].to(device).float()).cpu())
    return torch.cat(logits)


def population_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    population_size: int,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    targets: list[int] = []
    predictions: list[int] = []
    for class_id in range(num_classes):
        indices = torch.nonzero(labels == class_id).flatten().numpy()
        if not len(indices):
            raise ValueError(f"validation has no examples for state class {class_id}")
        for _ in range(reps):
            chosen = rng.choice(indices, size=population_size, replace=True)
            predictions.append(int(logits[chosen].mean(dim=0).argmax()))
            targets.append(class_id)
    result = classification_metrics(
        targets,
        predictions,
        num_classes=num_classes,
    )
    result["population_size"] = int(population_size)
    result["reps_per_class"] = int(reps)
    return result


def select_task(
    cache: dict[str, Any],
    *,
    label_key: str,
    state_group_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = cache["features"].float()
    labels = cache[label_key].long()
    selected = labels >= 0
    if state_group_id is not None:
        selected &= cache["state_group"].long() == state_group_id
    return features[selected], labels[selected]


def train_probe(
    *,
    task_name: str,
    branch: str,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    population: bool,
) -> dict[str, Any]:
    torch.manual_seed(args.seed + sum(ord(char) for char in task_name))
    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0).clamp_min(1e-4)
    train_features = (train_features - feature_mean) / feature_std
    val_features = (val_features - feature_mean) / feature_std

    model = FeatureProbe(
        train_features.shape[1],
        args.hidden_dim,
        num_classes,
        args.dropout,
    ).to(device)
    weights = balanced_class_weights(train_labels, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=args.probe_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    task_dir = output_dir / "probes" / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    remaining_patience = args.patience
    best_payload: dict[str, Any] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for feature_batch, label_batch in loader:
            feature_batch = feature_batch.to(device)
            label_batch = label_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(feature_batch)
            loss = F.cross_entropy(logits, label_batch, weight=weights)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(label_batch)
            count += len(label_batch)

        model.eval()
        val_logits = predict(
            model,
            val_features,
            batch_size=args.probe_batch_size,
            device=device,
        )
        val_metrics = classification_metrics(
            val_labels.tolist(),
            val_logits.argmax(dim=1).tolist(),
            num_classes=num_classes,
        )
        selection_metrics = (
            population_metrics(
                val_logits,
                val_labels,
                num_classes=num_classes,
                population_size=args.population_size,
                reps=args.population_reps,
                seed=args.seed + epoch,
            )
            if population
            else val_metrics
        )
        score = float(selection_metrics["macro_f1"])
        epoch_record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, count),
            "val_single": val_metrics,
            "val_selection": selection_metrics,
        }
        history.append(epoch_record)
        print(
            f"{task_name}: epoch={epoch} loss={epoch_record['train_loss']:.6f} "
            f"val_acc={selection_metrics['accuracy']:.4f} "
            f"val_f1={selection_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            remaining_patience = args.patience
            best_payload = {
                "schema": "exp4.latent_factor_probe.checkpoint.v1",
                "task": task_name,
                "branch": branch,
                "model_config": model.config,
                "model": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "num_classes": int(num_classes),
                "best_epoch": int(epoch),
                "selection_metrics": selection_metrics,
                "single_image_metrics": val_metrics,
            }
            torch.save(best_payload, task_dir / "checkpoint_best.pt")
        else:
            remaining_patience -= 1
            if remaining_patience <= 0:
                break

    if best_payload is None:
        raise RuntimeError(f"{task_name}: no best checkpoint was saved")
    (task_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "task": task_name,
        "branch": branch,
        "n_train": int(len(train_labels)),
        "n_val": int(len(val_labels)),
        "train_class_counts": dict(
            sorted(Counter(train_labels.tolist()).items())
        ),
        "val_class_counts": dict(sorted(Counter(val_labels.tolist()).items())),
        "best_epoch": int(best_epoch),
        "single_image": best_payload["single_image_metrics"],
        "selection": best_payload["selection_metrics"],
        "checkpoint": str(task_dir / "checkpoint_best.pt"),
    }


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (_EXP4_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve(args.checkpoint)
    data_dir = resolve(args.data_dir)
    vocab_path = resolve(args.vocab)
    output_dir = resolve(args.output_dir)
    cache_dir = output_dir / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_samples_per_split < 0:
        raise ValueError("--max_samples_per_split cannot be negative")
    if args.population_size <= 0 or args.population_reps <= 0:
        raise ValueError("population settings must be positive")
    device = torch.device(args.device)
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    identity_labels, full_to_observable = observable_identity_vocab(
        vocab["identity"]["id_to_label"]
    )
    state_class_counts = {
        group: len(vocab["state"][group]["id_to_label"])
        for group in FORMAL_STATE_GROUPS
    }
    background_classes = len(vocab["background"]["id_to_label"])
    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint = load_checkpoint(str(checkpoint_path), map_location="cpu")
    cfg = config_from_checkpoint(checkpoint)

    datasets = {
        split: LabeledRealDataset(
            data_dir=data_dir,
            manifest_name=args.manifest_name,
            split=split,
            full_to_observable=full_to_observable,
            max_samples=args.max_samples_per_split,
        )
        for split in ("train", "val")
    }
    if args.max_samples_per_split == 0:
        expected = {"train": 25000, "val": 2500}
        for split, count in expected.items():
            if len(datasets[split]) != count:
                raise ValueError(
                    f"{split} count is {len(datasets[split])}, expected {count}"
                )

    for branch in BRANCH_SIZES:
        required_caches = {
            split: cache_dir / f"{branch}_{split}.pt"
            for split in ("train", "val")
        }
        reuse = True
        for split, cache_path in required_caches.items():
            if not cache_path.is_file():
                reuse = False
                break
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if (
                cached.get("checkpoint_sha256") != checkpoint_sha256
                or cached.get("grid_size") != args.grid_size
                or len(cached.get("sample_ids", [])) != len(datasets[split])
            ):
                reuse = False
                break
        if reuse:
            print(f"reuse complete feature caches for {branch}", flush=True)
            continue

        vae = load_branch_vae(branch, checkpoint, cfg, device)
        for split in ("train", "val"):
            loader = DataLoader(
                datasets[split],
                batch_size=args.extract_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                drop_last=False,
            )
            payload = extract_split(
                branch=branch,
                vae=vae,
                loader=loader,
                device=device,
                precision=args.precision,
                grid_size=args.grid_size,
                checkpoint_sha256=checkpoint_sha256,
                split=split,
            )
            torch.save(payload, required_caches[split])
        del vae
        if device.type == "cuda":
            torch.cuda.empty_cache()

    caches = {
        (branch, split): torch.load(
            cache_dir / f"{branch}_{split}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for branch in BRANCH_SIZES
        for split in ("train", "val")
    }

    task_specs = [
        {
            "name": "subject_low",
            "branch": "low",
            "label_key": "identity",
            "state_group_id": None,
            "num_classes": len(identity_labels),
            "population": False,
        },
        {
            "name": "state_bbbc021_mid",
            "branch": "mid",
            "label_key": "state",
            "state_group_id": 0,
            "num_classes": state_class_counts["bbbc021"],
            "population": True,
        },
        {
            "name": "state_dynamicnuclearnet_mid",
            "branch": "mid",
            "label_key": "state",
            "state_group_id": 1,
            "num_classes": state_class_counts["dynamicnuclearnet"],
            "population": True,
        },
        {
            "name": "background_high",
            "branch": "high",
            "label_key": "background",
            "state_group_id": None,
            "num_classes": background_classes,
            "population": False,
        },
    ]
    results: dict[str, Any] = {}
    for spec in task_specs:
        train_features, train_labels = select_task(
            caches[(spec["branch"], "train")],
            label_key=spec["label_key"],
            state_group_id=spec["state_group_id"],
        )
        val_features, val_labels = select_task(
            caches[(spec["branch"], "val")],
            label_key=spec["label_key"],
            state_group_id=spec["state_group_id"],
        )
        results[spec["name"]] = train_probe(
            task_name=spec["name"],
            branch=spec["branch"],
            train_features=train_features,
            train_labels=train_labels,
            val_features=val_features,
            val_labels=val_labels,
            num_classes=spec["num_classes"],
            args=args,
            device=device,
            output_dir=output_dir,
            population=spec["population"],
        )

    thresholds = {
        "subject_low": {"accuracy": 0.75, "macro_f1": 0.70},
        "state_bbbc021_mid": {"accuracy": 0.80, "macro_f1": 0.75},
        "state_dynamicnuclearnet_mid": {
            "accuracy": 0.90,
            "macro_f1": 0.90,
        },
        "background_high": {"accuracy": 0.95, "macro_f1": 0.95},
    }
    checks: dict[str, Any] = {}
    for task_name, threshold in thresholds.items():
        metrics = results[task_name]["selection"]
        checks[task_name] = {
            "accuracy": float(metrics["accuracy"]) >= threshold["accuracy"],
            "macro_f1": float(metrics["macro_f1"]) >= threshold["macro_f1"],
            "thresholds": threshold,
        }
    passed = all(
        value
        for task_checks in checks.values()
        for key, value in task_checks.items()
        if key != "thresholds"
    )
    summary = {
        "schema": "exp4.latent_factor_probe.capacity_audit.v1",
        "status": "pass" if passed else "fail",
        "date": "2026-07-30",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "data_dir": str(data_dir),
        "splits": {"fit": "train", "selection": "val"},
        "test_accessed": False,
        "generator_training": False,
        "descriptor": {
            "source": "frozen original-v4 VAE posterior means",
            "grid_size": args.grid_size,
            "features": "adaptive mean and local standard-deviation grids",
        },
        "identity_labels": identity_labels,
        "population_size": args.population_size,
        "results": results,
        "gate": {
            "status": "pass" if passed else "fail",
            "checks": checks,
            "pass_action": (
                "freeze probe checkpoints and design factor-feature target/"
                "preservation generator loss"
            ),
            "failure_action": (
                "do not train a generator with latent probes; move the failed "
                "factor to decoded-image or mask-aware supervision"
            ),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Frozen-VAE latent factor-probe capacity audit",
        "",
        f"- Status: **{summary['status'].upper()}**",
        "- Fit split: train",
        "- Capacity gate: validation",
        "- Generator trained: no",
        "- Test accessed: no",
        "",
        "| Task | Branch | N train | N val | Accuracy | Macro-F1 | Gate |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for task_name in [
        "subject_low",
        "state_bbbc021_mid",
        "state_dynamicnuclearnet_mid",
        "background_high",
    ]:
        result = results[task_name]
        metrics = result["selection"]
        task_pass = all(
            checks[task_name][key] for key in ("accuracy", "macro_f1")
        )
        lines.append(
            f"| {task_name} | {result['branch']} | {result['n_train']} | "
            f"{result['n_val']} | {metrics['accuracy']:.4f} | "
            f"{metrics['macro_f1']:.4f} | {'PASS' if task_pass else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "State tasks use 16-cell mean-logit population decisions for the "
            "selection metrics. Subject and background use single-image decisions.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(f"saved {summary_path}")
    print(f"CAPACITY GATE: {summary['status'].upper()}")
    print("TEST DATA ACCESSED: NO")



