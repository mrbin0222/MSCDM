#!/usr/bin/env python3
"""Build a train-only real DNN state-logit distribution bank."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mscdms.workflows.train_teacher_classifier import DNNStateDataset
from mscdms.control_evaluator import read_jsonl, sha256_file
from mscdms.factor_surrogate import AdaptedDinoStateJudge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("models/dinov2-large"),
    )
    parser.add_argument(
        "--judge_checkpoint",
        type=Path,
        default=Path(
            "outputs/v4_dnn_dinov2_state_surrogate_finetune_v1/"
            "checkpoint_best.pt"
        ),
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
        "--output",
        type=Path,
        default=Path(
            "outputs/v4_dnn_real_state_logit_bank_v1/"
            "real_state_logit_bank.pt"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="bf16",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    model_dir = resolve(args.model_dir)
    checkpoint = resolve(args.judge_checkpoint)
    data_dir = resolve(args.data_dir)
    manifest = resolve(args.manifest)
    output = resolve(args.output)
    for path in (
        model_dir / "config.json",
        model_dir / "model.safetensors",
        checkpoint,
        manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    rows = read_jsonl(manifest)
    dataset = DNNStateDataset(
        rows,
        split="train",
        data_dir=data_dir,
        train=False,
    )
    if len(dataset) != 5000:
        raise ValueError(f"expected 5000 DNN train rows, found {len(dataset)}")
    labels_from_rows = [
        int(row["state_class_id"])
        for row in dataset.rows
    ]
    class_counts = Counter(labels_from_rows)
    if set(class_counts) != set(range(6)):
        raise ValueError("train-only DNN bank does not cover six states")
    label_names: dict[int, set[str]] = {
        class_id: set() for class_id in range(6)
    }
    for row in dataset.rows:
        label_names[int(row["state_class_id"])].add(
            str(row["state_label"]).lower()
        )
    if any(len(names) != 1 for names in label_names.values()):
        raise ValueError(f"ambiguous DNN state label mapping: {label_names}")
    id_to_label = [
        next(iter(label_names[class_id]))
        for class_id in range(6)
    ]
    if id_to_label != ["3t3", "hek293", "hela", "pc-3", "raw264", "rpe"]:
        raise ValueError(f"unexpected DNN state label order: {id_to_label}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    device = torch.device(args.device)
    judge = AdaptedDinoStateJudge(
        checkpoint=checkpoint,
        dino_model_dir=model_dir,
    ).to(device).eval()
    amp_enabled = device.type == "cuda" and args.precision == "bf16"
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch_index, (pixels, target) in enumerate(loader, start=1):
            pixels = pixels.to(device, non_blocking=True)
            with torch.amp.autocast(
                "cuda",
                enabled=amp_enabled,
                dtype=torch.bfloat16,
            ):
                prediction = judge(pixels)
            logits.append(prediction.float().cpu())
            labels.append(target.long().cpu())
            if batch_index % 10 == 0 or batch_index == len(loader):
                print(
                    f"real-state logits "
                    f"{min(batch_index * args.batch_size, len(dataset))}/"
                    f"{len(dataset)}",
                    flush=True,
                )
    all_logits = torch.cat(logits)
    all_labels = torch.cat(labels)
    if all_logits.shape != (5000, 6):
        raise ValueError(f"unexpected logit shape: {tuple(all_logits.shape)}")
    if all_labels.tolist() != labels_from_rows:
        raise ValueError("data-loader label order differs from manifest order")

    global_mean = all_logits.mean(dim=0)
    global_std = all_logits.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = (all_logits - global_mean) / global_std
    class_within_distance = []
    class_stats: dict[str, Any] = {}
    for class_id in range(6):
        class_logits = normalized[all_labels == class_id]
        within = torch.cdist(class_logits, class_logits, p=2).mean()
        class_within_distance.append(within)
        class_stats[str(class_id)] = {
            "n": int(len(class_logits)),
            "raw_mean": all_logits[all_labels == class_id].mean(dim=0).tolist(),
            "raw_std": (
                all_logits[all_labels == class_id]
                .std(dim=0, unbiased=False)
                .tolist()
            ),
            "normalized_within_distance": float(within),
        }

    payload = {
        "schema": "exp4.dnn.real_state_logit_bank.v1",
        "date": "2026-07-31",
        "split": "train",
        "source_name": "dynamicnuclearnet",
        "n": len(dataset),
        "num_classes": 6,
        "id_to_label": id_to_label,
        "sample_ids": [str(row["sample_id"]) for row in dataset.rows],
        "logits": all_logits,
        "labels": all_labels,
        "global_mean": global_mean,
        "global_std": global_std,
        "class_within_distance": torch.stack(class_within_distance),
        "class_counts": {
            str(key): int(value)
            for key, value in sorted(class_counts.items())
        },
        "class_stats": class_stats,
        "judge_checkpoint": str(checkpoint),
        "judge_checkpoint_sha256": sha256_file(checkpoint),
        "base_model": str(model_dir),
        "base_model_sha256": sha256_file(model_dir / "model.safetensors"),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "preprocessing": (
            "percentile-normalized isolated cell -> resize 256 -> "
            "center crop 224 -> DINO ImageNet normalization"
        ),
        "val_images_opened": False,
        "test_images_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)

    summary = {
        key: value
        for key, value in payload.items()
        if key not in {
            "sample_ids",
            "logits",
            "labels",
            "global_mean",
            "global_std",
            "class_within_distance",
        }
    }
    summary.update(
        {
            "output": str(output),
            "output_sha256": sha256_file(output),
            "global_mean": global_mean.tolist(),
            "global_std": global_std.tolist(),
            "class_within_distance": (
                torch.stack(class_within_distance).tolist()
            ),
        }
    )
    summary_path = output.with_name("summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved {output}")
    print(f"saved {summary_path}")
    print("VAL IMAGES OPENED: NO")
    print("TEST IMAGES OPENED: NO")


if __name__ == "__main__":
    main()
