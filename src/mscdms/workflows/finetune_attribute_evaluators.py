#!/usr/bin/env python3
"""Finetune attribute evaluators."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mscdms.control_evaluator import (
    ControlProbe,
    classification_metrics,
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

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


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
        default=Path("models/clip-vit-large-patch14"),
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/v4_control_evaluators_finetuned_v2"),
    )
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--eval_batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--vision_lr", type=float, default=1e-5)
    ap.add_argument("--probe_lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--last_vision_blocks", type=int, default=4)
    ap.add_argument("--bbbc_moa_repeat", type=int, default=8)
    ap.add_argument("--identity_loss_weight", type=float, default=1.0)
    ap.add_argument("--state_loss_weight", type=float, default=2.0)
    ap.add_argument("--background_loss_weight", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument(
        "--source_filter",
        choices=STATE_GROUPS,
        help=(
            "Train a source-routed ICA/SCA expert. BCA remains the responsibility "
            "of the shared cross-source evaluator."
        ),
    )
    ap.add_argument("--distributed_backend", choices=["nccl", "gloo"], default="nccl")
    ap.add_argument("--ica_min_accuracy", type=float, default=0.80)
    ap.add_argument("--ica_min_macro_f1", type=float, default=0.75)
    ap.add_argument("--sca_min_accuracy", type=float, default=0.70)
    ap.add_argument("--sca_min_macro_f1", type=float, default=0.60)
    ap.add_argument("--bca_min_accuracy", type=float, default=0.90)
    ap.add_argument("--bca_min_macro_f1", type=float, default=0.90)
    return ap.parse_args()


def from_root(path: Path, root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


class RandomQuarterTurn:
    def __call__(self, image: Image.Image) -> Image.Image:
        turns = random.randrange(4)
        if turns == 1:
            return image.transpose(Image.Transpose.ROTATE_90)
        if turns == 2:
            return image.transpose(Image.Transpose.ROTATE_180)
        if turns == 3:
            return image.transpose(Image.Transpose.ROTATE_270)
        return image


class RealControlDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        dataset_root: Path,
        split: str,
        train: bool,
        bbbc_moa_repeat: int,
    ) -> None:
        selected = sorted(
            (row for row in rows if row["split"] == split),
            key=lambda row: row["sample_id"],
        )
        if train and bbbc_moa_repeat > 1:
            annotated_bbbc = [
                row
                for row in selected
                if row["source_name"] == "bbbc021"
                and row["state_class_id"] is not None
            ]
            selected.extend(annotated_bbbc * (bbbc_moa_repeat - 1))
        self.rows = selected
        self.dataset_root = dataset_root
        self.group_to_id = {
            group: index for index, group in enumerate(STATE_GROUPS)
        }
        transform_steps: list[Any] = []
        if train:
            transform_steps.extend(
                [
                    RandomQuarterTurn(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                ]
            )
        transform_steps.extend(
            [
                transforms.Resize(
                    (224, 224),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ]
        )
        self.transform = transforms.Compose(transform_steps)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        row = self.rows[index]
        path = resolve_image_path(str(row["image_path"]), self.dataset_root)
        with Image.open(path) as image:
            pixel_values = self.transform(image.convert("RGB"))
        return (
            pixel_values,
            torch.tensor(int(row["identity_class_id"]), dtype=torch.long),
            torch.tensor(int(row["background_class_id"]), dtype=torch.long),
            torch.tensor(self.group_to_id[row["state_group"]], dtype=torch.long),
            torch.tensor(
                -1
                if row["state_class_id"] is None
                else int(row["state_class_id"]),
                dtype=torch.long,
            ),
        )


class FineTunedControlModel(nn.Module):
    def __init__(self, clip_model: nn.Module, probe: ControlProbe) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.probe = probe

    def forward(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        output = self.clip_model.vision_model(pixel_values=pixel_values)
        features = self.clip_model.visual_projection(output.pooler_output)
        return self.probe(features)


def configure_trainable_vision(
    clip_model: nn.Module,
    last_blocks: int,
) -> list[str]:
    for parameter in clip_model.parameters():
        parameter.requires_grad_(False)
    layers = clip_model.vision_model.encoder.layers
    if not 1 <= last_blocks <= len(layers):
        raise ValueError(
            f"last_vision_blocks must be in [1,{len(layers)}], found {last_blocks}"
        )
    for layer in layers[-last_blocks:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    for module in [
        clip_model.vision_model.post_layernorm,
        clip_model.visual_projection,
    ]:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return [
        name
        for name, parameter in clip_model.named_parameters()
        if parameter.requires_grad
    ]


def class_weights(values: list[int], count: int, device: torch.device) -> torch.Tensor:
    labels = torch.tensor(
        [value for value in values if value >= 0],
        dtype=torch.long,
    )
    support = torch.bincount(labels, minlength=count).float()
    weights = torch.zeros_like(support)
    present = support > 0
    weights[present] = torch.sqrt(support[present].sum() / support[present])
    weights[present] /= weights[present].mean()
    return weights.to(device)


def build_weights(
    rows: list[dict[str, Any]],
    *,
    identity_classes: int,
    state_classes: dict[str, int],
    background_classes: int,
    device: torch.device,
) -> dict[str, Any]:
    train = [row for row in rows if row["split"] == "train"]
    result: dict[str, Any] = {
        "identity": class_weights(
            [int(row["identity_class_id"]) for row in train],
            identity_classes,
            device,
        ),
        "background": class_weights(
            [int(row["background_class_id"]) for row in train],
            background_classes,
            device,
        ),
        "state": {},
    }
    for group in STATE_GROUPS:
        values = [
            -1 if row["state_class_id"] is None else int(row["state_class_id"])
            for row in train
            if row["state_group"] == group
        ]
        result["state"][group] = class_weights(
            values, state_classes[group], device
        )
    return result


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
    model: FineTunedControlModel,
    loader: DataLoader,
    *,
    device: torch.device,
    state_classes: dict[str, int],
    identity_classes: int,
    background_classes: int,
) -> dict[str, Any]:
    model.eval()
    identity_targets: list[int] = []
    identity_predictions: list[int] = []
    background_targets: list[int] = []
    background_predictions: list[int] = []
    state_targets: list[int] = []
    state_predictions: list[int] = []
    state_groups: list[int] = []
    for pixel_values, identity, background, group, state in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        with torch.amp.autocast(
            "cuda",
            enabled=device.type == "cuda",
            dtype=torch.bfloat16,
        ):
            identity_logits, background_logits, states = model(pixel_values)
        identity_targets.extend(identity.tolist())
        identity_predictions.extend(identity_logits.argmax(1).cpu().tolist())
        background_targets.extend(background.tolist())
        background_predictions.extend(background_logits.argmax(1).cpu().tolist())
        for index in range(len(state)):
            target = int(state[index])
            if target < 0:
                continue
            group_id = int(group[index])
            group_name = STATE_GROUPS[group_id]
            state_groups.append(group_id)
            state_targets.append(target)
            state_predictions.append(
                int(states[group_name][index].argmax().cpu())
            )
    identity_metrics = classification_metrics(
        identity_targets,
        identity_predictions,
        num_classes=identity_classes,
    )
    background_metrics = classification_metrics(
        background_targets,
        background_predictions,
        num_classes=background_classes,
    )
    combined_targets: list[int] = []
    combined_predictions: list[int] = []
    by_group: dict[str, Any] = {}
    offset = 0
    for group_id, group in enumerate(STATE_GROUPS):
        indices = [
            index
            for index, value in enumerate(state_groups)
            if value == group_id
        ]
        targets = [state_targets[index] for index in indices]
        predictions = [state_predictions[index] for index in indices]
        by_group[group] = classification_metrics(
            targets,
            predictions,
            num_classes=state_classes[group],
        )
        combined_targets.extend(value + offset for value in targets)
        combined_predictions.extend(value + offset for value in predictions)
        offset += state_classes[group]
    state_metrics = classification_metrics(
        combined_targets,
        combined_predictions,
        num_classes=offset,
    )
    state_metrics["by_group"] = by_group
    return {
        "identity": identity_metrics,
        "state": state_metrics,
        "background": background_metrics,
        "selection_score_macro_f1_mean": float(
            np.mean(
                [
                    identity_metrics["macro_f1"],
                    state_metrics["macro_f1"],
                    background_metrics["macro_f1"],
                ]
            )
        ),
    }


def capture_parameter_delta(
    clip_model: nn.Module,
    trainable_names: list[str],
) -> dict[str, torch.Tensor]:
    parameters = dict(clip_model.named_parameters())
    return {
        name: parameters[name].detach().cpu().clone()
        for name in trainable_names
    }


def apply_parameter_delta(
    clip_model: nn.Module,
    delta: dict[str, torch.Tensor],
) -> None:
    parameters = dict(clip_model.named_parameters())
    with torch.no_grad():
        for name, value in delta.items():
            parameters[name].copy_(
                value.to(
                    device=parameters[name].device,
                    dtype=parameters[name].dtype,
                )
            )


def metric_summary(metrics: dict[str, Any]) -> str:
    return (
        f"ICA={metrics['identity']['accuracy']:.4f}/"
        f"{metrics['identity']['macro_f1']:.4f} "
        f"SCA={metrics['state']['accuracy']:.4f}/"
        f"{metrics['state']['macro_f1']:.4f} "
        f"BCA={metrics['background']['accuracy']:.4f}/"
        f"{metrics['background']['macro_f1']:.4f}"
    )


def gate_report(
    metrics: dict[str, Any],
    args: argparse.Namespace,
    *,
    include_background: bool = True,
) -> dict[str, Any]:
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
    if not include_background:
        thresholds.pop("background")
    checks = {
        f"{task}_{metric}": float(metrics[task][metric]) >= threshold
        for task, task_thresholds in thresholds.items()
        for metric, threshold in task_thresholds.items()
    }
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

    distributed = "RANK" in os.environ
    if distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.distributed_backend)
        device = torch.device("cuda", local_rank)
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    rows = read_jsonl(protocol_dir / "evaluator_manifest.jsonl")
    if args.source_filter is not None:
        rows = [
            row for row in rows if row["source_name"] == args.source_filter
        ]
        split_counts = {
            split: sum(row["split"] == split for row in rows)
            for split in ["train", "val", "test"]
        }
        if any(count == 0 for count in split_counts.values()):
            raise ValueError(
                f"source_filter={args.source_filter} has empty split: "
                f"{split_counts}"
            )
        if rank == 0:
            print(
                f"source_filter={args.source_filter} "
                f"split_counts={split_counts}"
            )
    vocab_path = protocol_dir / "label_vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    state_classes = {
        group: len(vocab["state"][group]["id_to_label"])
        for group in STATE_GROUPS
    }
    identity_classes = len(vocab["identity"]["id_to_label"])
    background_classes = len(vocab["background"]["id_to_label"])

    _, clip_model = load_clip_image_encoder(clip_model_dir, device)
    trainable_names = configure_trainable_vision(
        clip_model, args.last_vision_blocks
    )
    probe = ControlProbe(
        feature_dim=int(clip_model.config.projection_dim),
        hidden_dim=args.hidden_dim,
        identity_classes=identity_classes,
        state_classes=state_classes,
        background_classes=background_classes,
        dropout=args.dropout,
    ).to(device)
    core_model = FineTunedControlModel(clip_model, probe).to(device)
    model: nn.Module
    if distributed:
        model = DDP(
            core_model,
            device_ids=[local_rank],
            find_unused_parameters=True,
        )
    else:
        model = core_model

    dataset_root = protocol_dir.parent
    train_dataset = RealControlDataset(
        rows,
        dataset_root=dataset_root,
        split="train",
        train=True,
        bbbc_moa_repeat=args.bbbc_moa_repeat,
    )
    val_dataset = RealControlDataset(
        rows,
        dataset_root=dataset_root,
        split="val",
        train=False,
        bbbc_moa_repeat=1,
    )
    test_dataset = RealControlDataset(
        rows,
        dataset_root=dataset_root,
        split="test",
        train=False,
        bbbc_moa_repeat=1,
    )
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        if distributed
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    eval_loader_kwargs = {
        "batch_size": args.eval_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    val_loader = DataLoader(val_dataset, **eval_loader_kwargs) if rank == 0 else None
    test_loader = DataLoader(test_dataset, **eval_loader_kwargs) if rank == 0 else None
    weights = build_weights(
        rows,
        identity_classes=identity_classes,
        state_classes=state_classes,
        background_classes=background_classes,
        device=device,
    )
    vision_parameters = [
        parameter
        for parameter in core_model.clip_model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": vision_parameters, "lr": args.vision_lr},
            {"params": core_model.probe.parameters(), "lr": args.probe_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )

    checkpoint_path = output_dir / "checkpoint_best.pt"
    best_score = -math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        losses: list[float] = []
        for pixel_values, identity, background, group, state in train_loader:
            pixel_values = pixel_values.to(device, non_blocking=True)
            identity = identity.to(device, non_blocking=True)
            background = background.to(device, non_blocking=True)
            group = group.to(device, non_blocking=True)
            state = state.to(device, non_blocking=True)
            with torch.amp.autocast(
                "cuda",
                enabled=device.type == "cuda",
                dtype=torch.bfloat16,
            ):
                identity_logits, background_logits, state_logits = model(
                    pixel_values
                )
                identity_loss = F.cross_entropy(
                    identity_logits, identity, weight=weights["identity"]
                )
                background_loss = F.cross_entropy(
                    background_logits,
                    background,
                    weight=weights["background"],
                )
                current_state_loss = state_loss(
                    state_logits, group, state, weights["state"]
                )
                loss = (
                    args.identity_loss_weight * identity_loss
                    + args.state_loss_weight * current_state_loss
                    + args.background_loss_weight * background_loss
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                5.0,
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        if distributed:
            dist.barrier()
        if rank == 0:
            assert val_loader is not None
            val_metrics = evaluate(
                core_model,
                val_loader,
                device=device,
                state_classes=state_classes,
                identity_classes=identity_classes,
                background_classes=background_classes,
            )
            score = val_metrics["selection_score_macro_f1_mean"]
            mean_loss = float(np.mean(losses))
            history.append(
                {
                    "epoch": epoch,
                    "rank0_train_loss": mean_loss,
                    "vision_lr": float(optimizer.param_groups[0]["lr"]),
                    "probe_lr": float(optimizer.param_groups[1]["lr"]),
                    "validation": val_metrics,
                }
            )
            print(
                f"epoch={epoch}/{args.epochs} loss={mean_loss:.6f} "
                f"score={score:.4f} {metric_summary(val_metrics)}"
            )
            if score > best_score + 1e-6:
                best_score = score
                best_epoch = epoch
                stale = 0
                torch.save(
                    {
                        "schema": "exp4.control_evaluator.finetuned.v2",
                        "model": core_model.probe.state_dict(),
                        "model_config": core_model.probe.config(),
                        "vision_parameter_delta": capture_parameter_delta(
                            core_model.clip_model, trainable_names
                        ),
                        "trainable_vision_parameters": trainable_names,
                        "vocab_sha256": sha256_file(vocab_path),
                        "vocab": vocab,
                        "clip_model_dir": str(clip_model_dir),
                        "source_filter": args.source_filter,
                        "best_epoch": best_epoch,
                        "best_val_metrics": val_metrics,
                    },
                    checkpoint_path,
                )
            else:
                stale += 1
            score_state = torch.tensor(
                [score, float(stale)], device=device, dtype=torch.float64
            )
        else:
            score_state = torch.zeros(2, device=device, dtype=torch.float64)
        if distributed:
            dist.broadcast(score_state, src=0)
            dist.barrier()
        if rank != 0:
            score = float(score_state[0])
            stale = int(score_state[1])
        if stale >= args.patience:
            if rank == 0:
                print(f"early stopping after {stale} stale epochs")
            break

    if distributed:
        dist.barrier()
    if rank == 0:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        core_model.probe.load_state_dict(checkpoint["model"])
        apply_parameter_delta(
            core_model.clip_model, checkpoint["vision_parameter_delta"]
        )
        assert val_loader is not None and test_loader is not None
        validation = evaluate(
            core_model,
            val_loader,
            device=device,
            state_classes=state_classes,
            identity_classes=identity_classes,
            background_classes=background_classes,
        )
        test = evaluate(
            core_model,
            test_loader,
            device=device,
            state_classes=state_classes,
            identity_classes=identity_classes,
            background_classes=background_classes,
        )
        gate = gate_report(
            test,
            args,
            include_background=args.source_filter is None,
        )
        report = {
            "schema": "exp4.control_evaluator.real_test_report.v2",
            "status": gate["status"],
            "best_epoch": best_epoch,
            "policy": {
                "real_train_only": True,
                "real_validation_checkpoint_selection": True,
                "real_test_after_selection": True,
                "generated_images_used_for_training": False,
                "clip_text_tower_frozen": True,
                "early_vision_blocks_frozen": True,
                "last_vision_blocks_finetuned": args.last_vision_blocks,
                "bbbc021_state_ground_truth": "official 12-class MoA subset",
                "source_filter": args.source_filter,
                "background_gate_included": args.source_filter is None,
            },
            "validation": validation,
            "test": test,
            "gate": gate,
        }
        write_json(output_dir / "training_history.json", history)
        write_json(output_dir / "real_test_report.json", report)
        report_scope = (
            "shared ICA/SCA/BCA"
            if args.source_filter is None
            else f"{args.source_filter} routed ICA/SCA expert"
        )
        markdown = f"""# Fine-tuned real-image evaluator report

Status: **{gate['status'].upper()}**

- Evaluation scope: {report_scope}

| Evaluator | Accuracy | Macro-F1 |
|---|---:|---:|
| ICA | {test['identity']['accuracy']:.4f} | {test['identity']['macro_f1']:.4f} |
| SCA | {test['state']['accuracy']:.4f} | {test['state']['macro_f1']:.4f} |
| BCA | {test['background']['accuracy']:.4f} | {test['background']['macro_f1']:.4f} |

- Best validation epoch: {best_epoch}
- Last CLIP vision blocks fine-tuned: {args.last_vision_blocks}
- BBBC021 SCA: official 12-class MoA subset
- Generated images used for evaluator training: **NO**
- BCA gate included: **{'YES' if args.source_filter is None else 'NO; use shared BCA evaluator'}**
- Failed gates: {', '.join(gate['failed_checks']) or 'none'}
"""
        (output_dir / "real_test_report.md").write_text(
            markdown, encoding="utf-8"
        )
        print(f"saved {checkpoint_path}")
        print(f"saved {output_dir / 'real_test_report.json'}")
        print(f"REAL TEST EVALUATOR GATE V2: {gate['status'].upper()}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
