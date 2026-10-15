#!/usr/bin/env python3
"""Train attribute evaluators."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mscdms.control_evaluator import (
    ControlProbe,
    classification_metrics,
    extract_clip_features,
    load_clip_image_encoder,
    read_jsonl,
    resolve_image_path,
    sha256_file,
    write_json,
)


STATE_GROUPS = [
    "bbbc021",
    "idr0033",
    "tissuenet_wholecell",
    "tissuenet_nuclear",
    "dynamicnuclearnet",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp4_root", type=Path, default=root)
    ap.add_argument(
        "--protocol_dir",
        type=Path,
        default=Path(
            "data/processed/control_metrics_v2"
        ),
    )
    ap.add_argument(
        "--clip_model_dir",
        type=Path,
        default=Path("models/clip-vit-base-patch32"),
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/v4_control_evaluators_probe_v2"),
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--feature_batch_size", type=int, default=128)
    ap.add_argument("--train_batch_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--rebuild_features", action="store_true")
    ap.add_argument("--fail_on_gate", action="store_true")
    ap.add_argument("--ica_min_accuracy", type=float, default=0.80)
    ap.add_argument("--ica_min_macro_f1", type=float, default=0.75)
    ap.add_argument("--sca_min_accuracy", type=float, default=0.70)
    ap.add_argument("--sca_min_macro_f1", type=float, default=0.60)
    ap.add_argument("--bca_min_accuracy", type=float, default=0.90)
    ap.add_argument("--bca_min_macro_f1", type=float, default=0.90)
    return ap.parse_args()


def from_root(path: Path, root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_vocab(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cache_split_features(
    *,
    rows: list[dict[str, Any]],
    split: str,
    cache_path: Path,
    dataset_root: Path,
    processor: Any,
    encoder: Any,
    device: torch.device,
    batch_size: int,
    vocab_sha256: str,
    clip_model_dir: Path,
) -> None:
    selected = sorted(
        (row for row in rows if row["split"] == split),
        key=lambda row: row["sample_id"],
    )
    paths = [
        resolve_image_path(str(row["image_path"]), dataset_root)
        for row in selected
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images are missing; first={missing[0]}"
        )
    features = extract_clip_features(
        paths,
        processor=processor,
        model=encoder,
        device=device,
        batch_size=batch_size,
    )
    group_to_id = {group: index for index, group in enumerate(STATE_GROUPS)}
    payload = {
        "schema": "exp4.control_evaluator.features.v1",
        "split": split,
        "sample_ids": [row["sample_id"] for row in selected],
        "features": features.to(torch.float16),
        "identity": torch.tensor(
            [int(row["identity_class_id"]) for row in selected],
            dtype=torch.long,
        ),
        "background": torch.tensor(
            [int(row["background_class_id"]) for row in selected],
            dtype=torch.long,
        ),
        "state_group": torch.tensor(
            [group_to_id[row["state_group"]] for row in selected],
            dtype=torch.long,
        ),
        "state": torch.tensor(
            [
                -1 if row["state_class_id"] is None else int(row["state_class_id"])
                for row in selected
            ],
            dtype=torch.long,
        ),
        "vocab_sha256": vocab_sha256,
        "clip_model_dir": str(clip_model_dir),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(f"cached {split}: n={len(selected)} -> {cache_path}")


def class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    valid = labels[(labels >= 0) & (labels < num_classes)]
    counts = torch.bincount(valid, minlength=num_classes).float()
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = torch.sqrt(counts[present].sum() / counts[present])
    if present.any():
        weights[present] /= weights[present].mean()
    return weights


def build_weight_map(
    train: dict[str, Any],
    *,
    identity_classes: int,
    state_classes: dict[str, int],
    background_classes: int,
    device: torch.device,
) -> dict[str, Any]:
    weights: dict[str, Any] = {
        "identity": class_weights(train["identity"], identity_classes).to(device),
        "background": class_weights(
            train["background"], background_classes
        ).to(device),
        "state": {},
    }
    for group_id, group in enumerate(STATE_GROUPS):
        mask = train["state_group"] == group_id
        weights["state"][group] = class_weights(
            train["state"][mask], state_classes[group]
        ).to(device)
    return weights


def state_loss(
    logits: dict[str, torch.Tensor],
    groups: torch.Tensor,
    targets: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for group_id, group in enumerate(STATE_GROUPS):
        mask = (groups == group_id) & (targets >= 0)
        if mask.any():
            losses.append(
                F.cross_entropy(
                    logits[group][mask],
                    targets[mask],
                    weight=weights[group],
                )
            )
    if not losses:
        return torch.zeros((), device=targets.device)
    return torch.stack(losses).mean()


@torch.inference_mode()
def evaluate(
    model: ControlProbe,
    payload: dict[str, Any],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    identity_predictions: list[int] = []
    background_predictions: list[int] = []
    state_predictions: list[int] = []
    state_targets: list[int] = []
    state_groups: list[int] = []
    features = payload["features"]
    for start in range(0, len(features), batch_size):
        stop = min(len(features), start + batch_size)
        batch = features[start:stop].to(device=device, dtype=torch.float32)
        identity_logits, background_logits, states = model(batch)
        identity_predictions.extend(identity_logits.argmax(dim=1).cpu().tolist())
        background_predictions.extend(
            background_logits.argmax(dim=1).cpu().tolist()
        )
        group_batch = payload["state_group"][start:stop]
        target_batch = payload["state"][start:stop]
        for index in range(stop - start):
            target = int(target_batch[index])
            if target < 0:
                continue
            group_id = int(group_batch[index])
            group = STATE_GROUPS[group_id]
            state_groups.append(group_id)
            state_targets.append(target)
            state_predictions.append(
                int(states[group][index].argmax().cpu())
            )

    identity = classification_metrics(
        payload["identity"].tolist(),
        identity_predictions,
        num_classes=model.identity_classes,
    )
    background = classification_metrics(
        payload["background"].tolist(),
        background_predictions,
        num_classes=model.background_classes,
    )
    state_by_group: dict[str, Any] = {}
    combined_targets: list[int] = []
    combined_predictions: list[int] = []
    offset = 0
    for group_id, group in enumerate(STATE_GROUPS):
        indices = [
            index
            for index, value in enumerate(state_groups)
            if value == group_id
        ]
        group_targets = [state_targets[index] for index in indices]
        group_predictions = [state_predictions[index] for index in indices]
        state_by_group[group] = classification_metrics(
            group_targets,
            group_predictions,
            num_classes=model.state_classes[group],
        )
        combined_targets.extend(target + offset for target in group_targets)
        combined_predictions.extend(
            prediction + offset for prediction in group_predictions
        )
        offset += model.state_classes[group]
    state = classification_metrics(
        combined_targets,
        combined_predictions,
        num_classes=sum(model.state_classes.values()),
    )
    state["by_group"] = state_by_group
    selection_score = float(
        np.mean(
            [
                identity["macro_f1"],
                state["macro_f1"],
                background["macro_f1"],
            ]
        )
    )
    return {
        "identity": identity,
        "state": state,
        "background": background,
        "selection_score_macro_f1_mean": selection_score,
    }


def metric_summary(metrics: dict[str, Any]) -> str:
    return (
        f"ICA acc={metrics['identity']['accuracy']:.4f} "
        f"f1={metrics['identity']['macro_f1']:.4f}; "
        f"SCA acc={metrics['state']['accuracy']:.4f} "
        f"f1={metrics['state']['macro_f1']:.4f}; "
        f"BCA acc={metrics['background']['accuracy']:.4f} "
        f"f1={metrics['background']['macro_f1']:.4f}"
    )


def gate_report(metrics: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    thresholds = {
        "identity": {
            "accuracy": args.ica_min_accuracy,
            "macro_f1": args.ica_min_macro_f1,
        },
        "state": {
            "accuracy": args.sca_min_accuracy,
            "macro_f1": args.sca_min_macro_f1,
        },
        "background": {
            "accuracy": args.bca_min_accuracy,
            "macro_f1": args.bca_min_macro_f1,
        },
    }
    checks: dict[str, bool] = {}
    for task, task_thresholds in thresholds.items():
        for metric, threshold in task_thresholds.items():
            checks[f"{task}_{metric}"] = (
                float(metrics[task][metric]) >= threshold
            )
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "thresholds": thresholds,
        "checks": checks,
        "failed_checks": sorted(
            name for name, passed in checks.items() if not passed
        ),
    }


def main() -> None:
    args = parse_args()
    root = args.exp4_root.resolve()
    protocol_dir = from_root(args.protocol_dir, root)
    clip_model_dir = from_root(args.clip_model_dir, root)
    output_dir = from_root(args.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    seed_everything(args.seed)

    manifest_path = protocol_dir / "evaluator_manifest.jsonl"
    vocab_path = protocol_dir / "label_vocab.json"
    rows = read_jsonl(manifest_path)
    vocab = load_vocab(vocab_path)
    vocab_sha256 = sha256_file(vocab_path)
    state_classes = {
        group: len(vocab["state"][group]["id_to_label"])
        for group in STATE_GROUPS
    }
    identity_classes = len(vocab["identity"]["id_to_label"])
    background_classes = len(vocab["background"]["id_to_label"])

    feature_dir = output_dir / "real_features"
    cache_paths = {
        split: feature_dir / f"{split}.pt"
        for split in ["train", "val", "test"]
    }
    need_features = args.rebuild_features or any(
        not path.is_file() for path in cache_paths.values()
    )
    if need_features:
        processor, encoder = load_clip_image_encoder(clip_model_dir, device)
        dataset_root = protocol_dir.parent
        for split, cache_path in cache_paths.items():
            if args.rebuild_features or not cache_path.is_file():
                cache_split_features(
                    rows=rows,
                    split=split,
                    cache_path=cache_path,
                    dataset_root=dataset_root,
                    processor=processor,
                    encoder=encoder,
                    device=device,
                    batch_size=args.feature_batch_size,
                    vocab_sha256=vocab_sha256,
                    clip_model_dir=clip_model_dir,
                )
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payloads = {
        split: torch.load(path, map_location="cpu", weights_only=False)
        for split, path in cache_paths.items()
    }
    for split, payload in payloads.items():
        if payload.get("vocab_sha256") != vocab_sha256:
            raise ValueError(
                f"{split} feature cache vocabulary mismatch; use --rebuild_features"
            )

    train = payloads["train"]
    model = ControlProbe(
        feature_dim=int(train["features"].shape[1]),
        hidden_dim=args.hidden_dim,
        identity_classes=identity_classes,
        state_classes=state_classes,
        background_classes=background_classes,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )
    weights = build_weight_map(
        train,
        identity_classes=identity_classes,
        state_classes=state_classes,
        background_classes=background_classes,
        device=device,
    )
    dataset = TensorDataset(
        train["features"],
        train["identity"],
        train["background"],
        train["state_group"],
        train["state"],
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    best_score = -math.inf
    best_epoch = 0
    stale_epochs = 0
    checkpoint_path = output_dir / "checkpoint_best.pt"
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for feature, identity, background, group, state in loader:
            feature = feature.to(device=device, dtype=torch.float32)
            identity = identity.to(device)
            background = background.to(device)
            group = group.to(device)
            state = state.to(device)
            identity_logits, background_logits, state_logits = model(feature)
            loss_identity = F.cross_entropy(
                identity_logits, identity, weight=weights["identity"]
            )
            loss_background = F.cross_entropy(
                background_logits, background, weight=weights["background"]
            )
            loss_state = state_loss(
                state_logits, group, state, weights["state"]
            )
            loss = loss_identity + loss_state + loss_background
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_metrics = evaluate(
            model,
            payloads["val"],
            device=device,
            batch_size=args.train_batch_size,
        )
        score = val_metrics["selection_score_macro_f1_mean"]
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "val": val_metrics,
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch}/{args.epochs} loss={epoch_record['train_loss']:.6f} "
            f"score={score:.4f} {metric_summary(val_metrics)}"
        )
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "schema": "exp4.control_evaluator.checkpoint.v1",
                    "model": model.state_dict(),
                    "model_config": model.config(),
                    "vocab_sha256": vocab_sha256,
                    "vocab": vocab,
                    "clip_model_dir": str(clip_model_dir),
                    "best_epoch": best_epoch,
                    "best_val_metrics": val_metrics,
                    "training_args": vars(args),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"early stopping after {stale_epochs} stale epochs")
            break

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    val_metrics = evaluate(
        model,
        payloads["val"],
        device=device,
        batch_size=args.train_batch_size,
    )
    test_metrics = evaluate(
        model,
        payloads["test"],
        device=device,
        batch_size=args.train_batch_size,
    )
    gate = gate_report(test_metrics, args)
    report = {
        "schema": "exp4.control_evaluator.real_test_report.v1",
        "status": gate["status"],
        "policy": {
            "train": "small multitask probe fit on real train only",
            "validation": "best checkpoint selected by mean ICA/SCA/BCA macro-F1",
            "test": "opened only after checkpoint selection",
            "generated_images_used_for_training": False,
            "clip_backbone_frozen": True,
        },
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "validation": val_metrics,
        "test": test_metrics,
        "gate": gate,
    }
    write_json(output_dir / "training_history.json", history)
    write_json(output_dir / "real_test_report.json", report)
    markdown = f"""# ICA/SCA/BCA real-image evaluator report

Status: **{gate['status'].upper()}**

| Evaluator | Accuracy | Macro-F1 |
|---|---:|---:|
| ICA | {test_metrics['identity']['accuracy']:.4f} | {test_metrics['identity']['macro_f1']:.4f} |
| SCA | {test_metrics['state']['accuracy']:.4f} | {test_metrics['state']['macro_f1']:.4f} |
| BCA | {test_metrics['background']['accuracy']:.4f} | {test_metrics['background']['macro_f1']:.4f} |

- Best validation epoch: {best_epoch}
- Frozen CLIP image encoder: `{clip_model_dir}`
- Generated images used for evaluator training: **NO**
- Failed gates: {', '.join(gate['failed_checks']) or 'none'}
"""
    (output_dir / "real_test_report.md").write_text(markdown, encoding="utf-8")
    print(f"saved {checkpoint_path}")
    print(f"saved {output_dir / 'real_test_report.json'}")
    print(f"REAL TEST EVALUATOR GATE: {gate['status'].upper()}")
    if args.fail_on_gate and gate["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
