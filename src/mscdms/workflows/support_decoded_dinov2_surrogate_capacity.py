#!/usr/bin/env python3
"""Support decoded dinov2 surrogate capacity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

_EXP4_ROOT = Path(__file__).resolve().parents[3]
if str(_EXP4_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXP4_ROOT))
if str(_EXP4_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_EXP4_ROOT / "scripts"))

from mscdms.workflows.support_latent_factor_probe_capacity import train_probe
from mscdms.control_evaluator import read_jsonl, resolve_image_path, sha256_file


DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("models/dinov2-large"),
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/processed/"
            "control_metrics_v2/evaluator_manifest.jsonl"
        ),
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
        default=Path("outputs/v4_decoded_dinov2_surrogate_capacity_audit_v1"),
    )
    parser.add_argument("--extract_batch_size", type=int, default=64)
    parser.add_argument("--probe_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--population_size", type=int, default=16)
    parser.add_argument("--population_reps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260815)
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
        help="Smoke-test limit; zero uses the complete train/validation splits.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (_EXP4_ROOT / path).resolve()


class DinoRealDataset(Dataset[tuple[Any, ...]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        split: str,
        data_dir: Path,
        max_samples: int,
    ) -> None:
        self.rows = sorted(
            (row for row in rows if row["split"] == split),
            key=lambda row: str(row["sample_id"]),
        )
        if max_samples > 0:
            self.rows = self.rows[:max_samples]
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    256,
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(DINO_MEAN, DINO_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        row = self.rows[index]
        path = resolve_image_path(str(row["image_path"]), self.data_dir)
        with Image.open(path) as image:
            pixels = self.transform(image.convert("RGB"))
        state_id = (
            int(row["state_class_id"])
            if row["source_name"] == "bbbc021"
            and row["state_class_id"] is not None
            else -1
        )
        return (
            pixels,
            torch.tensor(state_id, dtype=torch.long),
            torch.tensor(int(row["background_class_id"]), dtype=torch.long),
            str(row["sample_id"]),
        )


def dino_descriptor(last_hidden_state: torch.Tensor) -> torch.Tensor:
    if last_hidden_state.ndim != 3 or last_hidden_state.shape[1] < 2:
        raise ValueError(
            "DINO last_hidden_state must have shape [batch,tokens,channels]"
        )
    cls = last_hidden_state[:, 0].float()
    patches = last_hidden_state[:, 1:].float()
    return torch.cat(
        [
            cls,
            patches.mean(dim=1),
            patches.std(dim=1, unbiased=False),
        ],
        dim=1,
    )


@torch.inference_mode()
def extract_split(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    split: str,
    device: torch.device,
    precision: str,
    model_sha256: str,
) -> dict[str, Any]:
    amp_enabled = device.type == "cuda" and precision != "fp32"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    features: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    backgrounds: list[torch.Tensor] = []
    sample_ids: list[str] = []

    for batch_index, batch in enumerate(loader, start=1):
        pixels, state, background, batch_ids = batch
        pixels = pixels.to(device, non_blocking=True)
        with torch.amp.autocast(
            "cuda",
            enabled=amp_enabled,
            dtype=amp_dtype,
        ):
            output = model(pixel_values=pixels)
        descriptor = dino_descriptor(output.last_hidden_state)
        features.append(descriptor.cpu().to(torch.float16))
        states.append(state)
        backgrounds.append(background)
        sample_ids.extend(str(value) for value in batch_ids)
        if batch_index == 1 or batch_index % 50 == 0 or batch_index == len(loader):
            print(
                f"extract DINO/{split}: "
                f"{min(batch_index * loader.batch_size, len(loader.dataset))}/"
                f"{len(loader.dataset)}",
                flush=True,
            )

    feature_tensor = torch.cat(features)
    if not torch.isfinite(feature_tensor.float()).all():
        raise ValueError(f"{split} DINO features contain non-finite values")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{split} contains duplicate sample IDs")
    return {
        "schema": "exp4.decoded_dinov2.feature_cache.v1",
        "split": split,
        "model_sha256": model_sha256,
        "features": feature_tensor,
        "state_bbbc021": torch.cat(states),
        "background": torch.cat(backgrounds),
        "sample_ids": sample_ids,
    }


def select_labels(
    cache: dict[str, Any],
    label_key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = cache[label_key].long()
    selected = labels >= 0
    return cache["features"][selected].float(), labels[selected]


def main() -> None:
    args = parse_args()
    model_dir = resolve(args.model_dir)
    data_dir = resolve(args.data_dir)
    manifest_path = resolve(args.manifest)
    vocab_path = resolve(args.vocab)
    output_dir = resolve(args.output_dir)
    cache_dir = output_dir / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        if not (model_dir / name).is_file():
            raise FileNotFoundError(model_dir / name)
    if args.max_samples_per_split < 0:
        raise ValueError("--max_samples_per_split cannot be negative")
    rows = read_jsonl(manifest_path)
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    datasets = {
        split: DinoRealDataset(
            rows,
            split=split,
            data_dir=data_dir,
            max_samples=args.max_samples_per_split,
        )
        for split in ("train", "val")
    }
    if args.max_samples_per_split == 0:
        for split, expected in (("train", 25000), ("val", 2500)):
            if len(datasets[split]) != expected:
                raise ValueError(
                    f"{split} count is {len(datasets[split])}, expected {expected}"
                )

    model_sha256 = sha256_file(model_dir / "model.safetensors")
    cache_paths = {
        split: cache_dir / f"{split}.pt" for split in ("train", "val")
    }
    reuse = all(path.is_file() for path in cache_paths.values())
    if reuse:
        for split, path in cache_paths.items():
            cache = torch.load(path, map_location="cpu", weights_only=False)
            if (
                cache.get("model_sha256") != model_sha256
                or len(cache.get("sample_ids", [])) != len(datasets[split])
            ):
                reuse = False
                break

    if reuse:
        print("reuse complete DINOv2 train/validation feature caches", flush=True)
    else:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            str(model_dir),
            local_files_only=True,
        ).to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        hidden_size = int(model.config.hidden_size)
        if hidden_size != 1024:
            raise ValueError(
                f"expected DINOv2-L hidden size 1024, got {hidden_size}"
            )
        for split in ("train", "val"):
            loader = DataLoader(
                datasets[split],
                batch_size=args.extract_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                drop_last=False,
            )
            cache = extract_split(
                model=model,
                loader=loader,
                split=split,
                device=device,
                precision=args.precision,
                model_sha256=model_sha256,
            )
            torch.save(cache, cache_paths[split])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    caches = {
        split: torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        for split, path in cache_paths.items()
    }
    if caches["train"]["sample_ids"] == caches["val"]["sample_ids"]:
        raise ValueError("train and validation sample IDs are unexpectedly identical")

    task_specs = {
        "state_bbbc021_decoded_dinov2": {
            "label_key": "state_bbbc021",
            "num_classes": len(vocab["state"]["bbbc021"]["id_to_label"]),
            "population": True,
            "thresholds": {"accuracy": 0.80, "macro_f1": 0.75},
        },
        "background_decoded_dinov2": {
            "label_key": "background",
            "num_classes": len(vocab["background"]["id_to_label"]),
            "population": False,
            "thresholds": {"accuracy": 0.95, "macro_f1": 0.95},
        },
    }
    results: dict[str, Any] = {}
    checks: dict[str, Any] = {}
    for task_name, spec in task_specs.items():
        train_features, train_labels = select_labels(
            caches["train"],
            spec["label_key"],
        )
        val_features, val_labels = select_labels(
            caches["val"],
            spec["label_key"],
        )
        results[task_name] = train_probe(
            task_name=task_name,
            branch="decoded_image_dinov2_large",
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
        selection = results[task_name]["selection"]
        checks[task_name] = {
            "accuracy": float(selection["accuracy"])
            >= spec["thresholds"]["accuracy"],
            "macro_f1": float(selection["macro_f1"])
            >= spec["thresholds"]["macro_f1"],
            "thresholds": spec["thresholds"],
        }

    passed = all(
        value
        for task_checks in checks.values()
        for key, value in task_checks.items()
        if key != "thresholds"
    )
    summary = {
        "schema": "exp4.decoded_dinov2_surrogate.capacity_audit.v1",
        "status": "pass" if passed else "fail",
        "date": "2026-07-30",
        "model_dir": str(model_dir),
        "model_sha256": model_sha256,
        "model_policy": {
            "backbone": "frozen DINOv2-L/14",
            "feature": "CLS + patch-token mean + patch-token standard deviation",
            "independent_from_final_evaluator": True,
            "final_evaluator": "frozen microscopy CLIP-L/14 control evaluator",
        },
        "splits": {"fit": "train", "selection": "val"},
        "test_accessed": False,
        "generator_training": False,
        "results": results,
        "gate": {
            "status": "pass" if passed else "fail",
            "checks": checks,
            "pass_action": (
                "freeze DINO probe checkpoints and design a decoded-x0 "
                "factor target/preservation pilot"
            ),
            "failure_action": (
                "do not use frozen-DINO decoded supervision; evaluate a "
                "train-only microscopy adaptation or mask-aware background loss"
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Independent decoded-image surrogate capacity audit",
        "",
        f"- Status: **{summary['status'].upper()}**",
        "- Backbone: frozen DINOv2-L/14",
        "- Fit: train",
        "- Selection: validation",
        "- Generator trained: no",
        "- Test accessed: no",
        "",
        "| Task | N train | N val | Accuracy | Macro-F1 | Gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for task_name in task_specs:
        result = results[task_name]
        metrics = result["selection"]
        task_pass = all(
            checks[task_name][key] for key in ("accuracy", "macro_f1")
        )
        lines.append(
            f"| {task_name} | {result['n_train']} | {result['n_val']} | "
            f"{metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{'PASS' if task_pass else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "BBBC021 state uses 16-cell mean-logit population decisions. "
            "Background uses single-image decisions.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(f"saved {output_dir / 'summary.json'}")
    print(f"DECODED SURROGATE CAPACITY GATE: {summary['status'].upper()}")
    print("TEST DATA ACCESSED: NO")



