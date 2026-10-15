#!/usr/bin/env python3
"""Microscopy-adapt DINOv2-L/14 for six-class DNN state recognition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

_EXP4_ROOT = Path(__file__).resolve().parents[3]
if str(_EXP4_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXP4_ROOT))
if str(_EXP4_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_EXP4_ROOT / "scripts"))

from mscdms.workflows.support_decoded_dinov2_surrogate_capacity import (
    DINO_MEAN,
    DINO_STD,
    dino_descriptor,
)
from mscdms.workflows.support_latent_factor_probe_capacity import (
    FeatureProbe,
    balanced_class_weights,
    population_metrics,
)
from mscdms.workflows.dino_training_utils import (
    RandomQuarterTurn,
    configure_trainable_dino,
    trainable_state,
)
from mscdms.control_evaluator import (
    classification_metrics,
    read_jsonl,
    resolve_image_path,
    sha256_file,
)


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
        default=Path("outputs/v4_dnn_dinov2_state_surrogate_finetune_v1"),
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--last_vision_blocks", type=int, default=4)
    parser.add_argument("--vision_lr", type=float, default=1e-5)
    parser.add_argument("--probe_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--population_size", type=int, default=16)
    parser.add_argument("--population_reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="bf16",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (_EXP4_ROOT / path).resolve()


def is_dnn_state_row(row: dict[str, Any], split: str) -> bool:
    return (
        row.get("split") == split
        and row.get("source_name") == "dynamicnuclearnet"
        and row.get("state_class_id") is not None
    )


def gate_passes(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics["accuracy"]) >= 0.90
        and float(metrics["macro_f1"]) >= 0.90
    )


class DNNStateDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        split: str,
        data_dir: Path,
        train: bool,
    ) -> None:
        self.rows = sorted(
            (row for row in rows if is_dnn_state_row(row, split)),
            key=lambda row: str(row["sample_id"]),
        )
        self.data_dir = data_dir
        if train:
            self.transform = transforms.Compose(
                [
                    RandomQuarterTurn(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomResizedCrop(
                        (224, 224),
                        scale=(0.90, 1.0),
                        ratio=(1.0, 1.0),
                        interpolation=InterpolationMode.BICUBIC,
                        antialias=True,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(DINO_MEAN, DINO_STD),
                ]
            )
        else:
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image_path = resolve_image_path(str(row["image_path"]), self.data_dir)
        with Image.open(image_path) as image:
            pixels = self.transform(image.convert("RGB"))
        label = torch.tensor(int(row["state_class_id"]), dtype=torch.long)
        return pixels, label


class DinoStateModel(nn.Module):
    def __init__(self, backbone: nn.Module, probe: FeatureProbe) -> None:
        super().__init__()
        self.backbone = backbone
        self.probe = probe

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        output = self.backbone(pixel_values=pixels)
        return self.probe(dino_descriptor(output.last_hidden_state))


@torch.inference_mode()
def evaluate(
    model: DinoStateModel,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
    num_classes: int,
    population_size: int,
    population_reps: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    amp_enabled = device.type == "cuda" and precision != "fp32"
    for pixels, target in loader:
        pixels = pixels.to(device, non_blocking=True)
        with torch.amp.autocast(
            "cuda",
            enabled=amp_enabled,
            dtype=torch.bfloat16,
        ):
            prediction = model(pixels)
        logits.append(prediction.float().cpu())
        labels.append(target)
    all_logits = torch.cat(logits)
    all_labels = torch.cat(labels)
    single = classification_metrics(
        all_labels.tolist(),
        all_logits.argmax(dim=1).tolist(),
        num_classes=num_classes,
    )
    population = population_metrics(
        all_logits,
        all_labels,
        num_classes=num_classes,
        population_size=population_size,
        reps=population_reps,
        seed=seed,
    )
    return single, population


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model_dir = resolve(args.model_dir)
    data_dir = resolve(args.data_dir)
    manifest_path = resolve(args.manifest)
    vocab_path = resolve(args.vocab)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        if not (model_dir / name).is_file():
            raise FileNotFoundError(model_dir / name)
    rows = read_jsonl(manifest_path)
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    state_vocab = vocab["state"]["dynamicnuclearnet"]["id_to_label"]
    num_classes = len(state_vocab)
    if num_classes != 6:
        raise ValueError(f"expected 6 DNN state classes, got {num_classes}")

    train_dataset = DNNStateDataset(
        rows,
        split="train",
        data_dir=data_dir,
        train=True,
    )
    val_dataset = DNNStateDataset(
        rows,
        split="val",
        data_dir=data_dir,
        train=False,
    )
    if len(train_dataset) != 5000 or len(val_dataset) != 500:
        raise ValueError(
            f"unexpected DNN state counts: {len(train_dataset)}/{len(val_dataset)}"
        )
    train_labels = torch.tensor(
        [int(row["state_class_id"]) for row in train_dataset.rows],
        dtype=torch.long,
    )
    val_labels = torch.tensor(
        [int(row["state_class_id"]) for row in val_dataset.rows],
        dtype=torch.long,
    )
    if set(train_labels.tolist()) != set(range(num_classes)):
        raise ValueError("DNN train split does not cover all six state classes")
    if set(val_labels.tolist()) != set(range(num_classes)):
        raise ValueError("DNN val split does not cover all six state classes")

    weights = balanced_class_weights(train_labels, num_classes).to(device)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(
        str(model_dir),
        local_files_only=True,
    ).to(device)
    if int(backbone.config.hidden_size) != 1024:
        raise ValueError(
            f"expected DINOv2-L hidden size 1024, got {backbone.config.hidden_size}"
        )
    trainable_names = configure_trainable_dino(
        backbone,
        args.last_vision_blocks,
    )
    probe = FeatureProbe(
        feature_dim=3 * int(backbone.config.hidden_size),
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)
    model = DinoStateModel(backbone, probe).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    parameter
                    for parameter in backbone.parameters()
                    if parameter.requires_grad
                ],
                "lr": args.vision_lr,
            },
            {"params": probe.parameters(), "lr": args.probe_lr},
        ],
        weight_decay=args.weight_decay,
    )
    amp_enabled = device.type == "cuda" and args.precision != "fp32"
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    remaining_patience = args.patience
    checkpoint_path = output_dir / "checkpoint_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for pixels, target in train_loader:
            pixels = pixels.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda",
                enabled=amp_enabled,
                dtype=torch.bfloat16,
            ):
                logits = model(pixels)
                loss = F.cross_entropy(logits.float(), target, weight=weights)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(target)
            count += len(target)

        single, population = evaluate(
            model,
            val_loader,
            device=device,
            precision=args.precision,
            num_classes=num_classes,
            population_size=args.population_size,
            population_reps=args.population_reps,
            seed=args.seed,
        )
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, count),
            "val_single": single,
            "val_population16": population,
        }
        history.append(record)
        print(
            f"epoch={epoch} loss={record['train_loss']:.6f} "
            f"single={single['accuracy']:.4f}/{single['macro_f1']:.4f} "
            f"population16={population['accuracy']:.4f}/"
            f"{population['macro_f1']:.4f}",
            flush=True,
        )
        score = float(single["macro_f1"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            remaining_patience = args.patience
            torch.save(
                {
                    "schema": "exp4.dnn_dinov2_state_surrogate.v1",
                    "base_model_dir": str(model_dir),
                    "base_model_sha256": sha256_file(
                        model_dir / "model.safetensors"
                    ),
                    "vocab_sha256": sha256_file(vocab_path),
                    "state_labels": list(state_vocab),
                    "last_vision_blocks": args.last_vision_blocks,
                    "trainable_vision_parameter_names": trainable_names,
                    "trainable_vision_state": trainable_state(
                        backbone,
                        trainable_names,
                    ),
                    "probe_config": probe.config,
                    "probe_state": {
                        key: value.detach().cpu()
                        for key, value in probe.state_dict().items()
                    },
                    "best_epoch": epoch,
                    "single_metrics": single,
                    "population16_metrics": population,
                },
                checkpoint_path,
            )
        else:
            remaining_patience -= 1
            if remaining_patience <= 0:
                break

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    single = checkpoint["single_metrics"]
    population = checkpoint["population16_metrics"]
    passed = gate_passes(single)
    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema": "exp4.dnn_dinov2_state_surrogate.finetune.v1",
        "status": "pass" if passed else "fail",
        "date": "2026-07-31",
        "model": {
            "base": "DINOv2-L/14",
            "adapted": (
                f"last {args.last_vision_blocks} vision blocks + six-class head"
            ),
            "trainable_parameters": int(
                sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
            ),
            "independent_from_final_clip_l14_evaluator": True,
        },
        "splits": {
            "train_real_cells": len(train_dataset),
            "val_real_cells": len(val_dataset),
            "train_class_counts": torch.bincount(
                train_labels,
                minlength=num_classes,
            ).tolist(),
            "val_class_counts": torch.bincount(
                val_labels,
                minlength=num_classes,
            ).tolist(),
        },
        "selection": "highest validation single-image macro-F1",
        "best_epoch": int(best_epoch),
        "single_image": single,
        "population16_diagnostic": population,
        "gate": {
            "thresholds": {"accuracy": 0.90, "macro_f1": 0.90},
            "checks": {
                "accuracy": float(single["accuracy"]) >= 0.90,
                "macro_f1": float(single["macro_f1"]) >= 0.90,
            },
            "status": "pass" if passed else "fail",
            "pass_action": (
                "freeze the adapted DNN state surrogate and permit one "
                "final-decoded-x0 edit-only generator pilot"
            ),
            "failure_action": (
                "stop DNN state generator supervision; do not lower the "
                "gate or scan additional backbone depths"
            ),
        },
        "checkpoint": str(checkpoint_path),
        "test_accessed": False,
        "generator_training": False,
        "hyperparameter_scan": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# DNN DINOv2 state surrogate adaptation",
                "",
                f"- Status: **{summary['status'].upper()}**",
                f"- Best epoch: {best_epoch}",
                f"- Train/val real cells: {len(train_dataset)}/"
                f"{len(val_dataset)}",
                f"- Single-image Acc/F1: {single['accuracy']:.4f}/"
                f"{single['macro_f1']:.4f}",
                f"- 16-cell diagnostic Acc/F1: "
                f"{population['accuracy']:.4f}/"
                f"{population['macro_f1']:.4f}",
                "- Generator trained: no",
                "- Test accessed: no",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"saved {output_dir / 'summary.json'}")
    print(f"DNN STATE SURROGATE GATE: {summary['status'].upper()}")
    print("TEST DATA ACCESSED: NO")


if __name__ == "__main__":
    main()
