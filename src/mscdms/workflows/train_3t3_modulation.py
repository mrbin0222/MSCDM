#!/usr/bin/env python3
"""Train 3t3 modulation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mscdms.control_evaluator import sha256_file
from mscdms.distributed import cleanup_distributed, get_device, setup_distributed
from mscdms.latent_foreground_diffusion import (
    MaskConditionedLatentForegroundDiffusion,
    initialize_latent_foreground_target_state_response,
    latent_calibrations_match,
    load_latent_foreground_diffusion,
)
from mscdms.workflows.train_foreground_renderer import reduce_totals


DNN_LABELS = ("3t3", "hek293", "hela", "pc-3", "raw264", "rpe")
TARGET_LABEL = "3t3"
TARGET_CLASS_ID = DNN_LABELS.index(TARGET_LABEL)
CHECKPOINT_SCHEMA = "exp4.latent_foreground_diffusion.stage42a.checkpoint.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--manifest_audit", type=Path, required=True)
    parser.add_argument("--base_checkpoint", type=Path, required=True)
    parser.add_argument("--teacher_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def state_label(row: dict[str, Any]) -> str:
    key = str(row["state_key"])
    source, separator, label = key.partition("::")
    if source != "dynamicnuclearnet" or separator != "::" or label not in DNN_LABELS:
        raise ValueError(f"invalid DNN state key: {key}")
    return label


class FixedMask3T3ResponseDataset(Dataset[dict[str, torch.Tensor]]):
    """Use each factual anchor's own latent, background and fixed mask."""

    def __init__(
        self,
        anchors: list[dict[str, Any]],
        *,
        cache_dir: Path,
        augment: bool,
    ) -> None:
        if not anchors or any(state_label(row) == TARGET_LABEL for row in anchors):
            raise ValueError("3T3 response anchors must be non-3T3 DNN records")
        self.anchors = anchors
        self.cache_dir = cache_dir
        self.augment = bool(augment)
        self._target: np.ndarray | None = None
        self._background: np.ndarray | None = None
        self._masks: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.anchors)

    def _open(self) -> None:
        if self._target is None:
            self._target = np.load(
                self.cache_dir / "target_latents.npy", mmap_mode="r"
            )
            self._background = np.load(
                self.cache_dir / "background_latents.npy", mmap_mode="r"
            )
            self._masks = np.load(self.cache_dir / "masks.npy", mmap_mode="r")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        self._open()
        assert self._target is not None
        assert self._background is not None
        assert self._masks is not None
        row = self.anchors[index]
        cache_index = int(row["cache_index"])
        clean = torch.from_numpy(
            np.array(self._target[cache_index], dtype=np.float32, copy=True)
        )
        background = torch.from_numpy(
            np.array(self._background[cache_index], dtype=np.float32, copy=True)
        )
        mask = torch.from_numpy(
            np.array(self._masks[cache_index], dtype=np.float32, copy=True)
        )
        if self.augment:
            if bool(torch.rand(()) < 0.5):
                clean, background, mask = clean.flip(-1), background.flip(-1), mask.flip(-1)
            if bool(torch.rand(()) < 0.5):
                clean, background, mask = clean.flip(-2), background.flip(-2), mask.flip(-2)
        return {
            "clean": clean,
            "background": background,
            "mask": mask,
            "source_id": torch.tensor(row["source_id"], dtype=torch.long),
            "anchor_class_id": torch.tensor(
                DNN_LABELS.index(state_label(row)), dtype=torch.long
            ),
            "cache_index": torch.tensor(cache_index, dtype=torch.long),
        }


class FixedMask3T3ResponseDistiller(nn.Module):
    def __init__(
        self,
        student: MaskConditionedLatentForegroundDiffusion,
        teacher: MaskConditionedLatentForegroundDiffusion,
        *,
        target_state_id: int,
    ) -> None:
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.target_state_id = int(target_state_id)

    def forward(
        self,
        clean: torch.Tensor,
        background: torch.Tensor,
        mask: torch.Tensor,
        source_ids: torch.Tensor,
        anchor_class_ids: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if mask.shape[-2:] != clean.shape[-2:]:
            mask = F.interpolate(mask, size=clean.shape[-2:], mode="nearest")
        mask = (mask >= 0.5).to(dtype=clean.dtype)
        target_state_ids = torch.full_like(source_ids, self.target_state_id)
        target_class_ids = torch.full_like(source_ids, TARGET_CLASS_ID)
        response_route = target_class_ids.ne(anchor_class_ids)
        noisy = self.student.diffusion.q_sample(clean, timesteps, noise)
        with torch.no_grad():
            teacher_condition = self.teacher.conditioner(
                source_ids, target_state_ids
            )
            teacher_prediction = self.teacher._predict(
                noisy,
                timesteps,
                background,
                mask,
                teacher_condition,
            )
        student_condition = self.student.conditioner(source_ids, target_state_ids)
        student_prediction = self.student._predict(
            noisy,
            timesteps,
            background,
            mask,
            student_condition,
            state_target_class_ids=target_class_ids,
            state_anchor_class_ids=anchor_class_ids,
            # Preserve the frozen Stage-36c RPE response exactly.
            state_delta_route_mask=torch.zeros_like(response_route),
            target_state_response_route_mask=response_route,
        )
        support = F.max_pool2d(mask, kernel_size=7, stride=1, padding=3)
        squared = (student_prediction.float() - teacher_prediction.float()).square()
        per_pixel = squared.mean(dim=1, keepdim=True)
        inside = (per_pixel * support).sum() / support.sum().clamp_min(1.0)
        weights = support + 0.05 * (1.0 - support)
        loss = (per_pixel * weights).sum() / weights.sum().clamp_min(1.0)
        return loss, {
            "masked_teacher_mse": inside.detach(),
            "prediction_abs_delta": (
                student_prediction.float() - teacher_prediction.float()
            ).abs().mean().detach(),
        }


def fixed_validation_inputs(
    shape: torch.Size,
    cache_indices: torch.Tensor,
    *,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    noises: list[torch.Tensor] = []
    timesteps: list[int] = []
    for cache_index in cache_indices.tolist():
        item_seed = seed + int(cache_index) * 104729
        generator = torch.Generator(device=device).manual_seed(item_seed)
        noises.append(torch.randn(shape[1:], generator=generator, device=device))
        timesteps.append(item_seed % 1000)
    return torch.stack(noises), torch.tensor(timesteps, device=device)


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = setup_distributed("nccl")
    device = get_device(local_rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)

    cache_audit = json.loads(
        (args.cache_dir / "cache_audit.json").read_text(encoding="utf-8")
    )
    manifest_audit = json.loads(args.manifest_audit.read_text(encoding="utf-8"))
    if cache_audit.get("status") != "pass" or cache_audit.get("test_accessed") is not False:
        raise ValueError("latent cache audit did not pass")
    if manifest_audit.get("status") != "pass" or manifest_audit.get("test_accessed") is not False:
        raise ValueError("manifest audit did not pass")
    rows = [
        row
        for row in read_jsonl(args.cache_dir / "index.jsonl")
        if row.get("source_name") == "dynamicnuclearnet"
    ]
    train = [row for row in rows if row.get("split") == "train"]
    val = [row for row in rows if row.get("split") == "val"]
    train_anchors = [row for row in train if state_label(row) != TARGET_LABEL]
    val_anchors = [row for row in val if state_label(row) != TARGET_LABEL]
    if len(train) != 5000 or len(val) != 500:
        raise ValueError("Stage-42a expects 5,000/500 DNN train/val cache rows")
    if len(train_anchors) != 4167 or len(val_anchors) != 417:
        raise ValueError("Stage-42a non-3T3 anchor counts changed")

    student, base_payload = initialize_latent_foreground_target_state_response(
        args.base_checkpoint, target_class=TARGET_CLASS_ID
    )
    teacher, teacher_payload = load_latent_foreground_diffusion(
        args.teacher_checkpoint
    )
    if teacher_payload.get("schema") != "exp4.latent_foreground_diffusion.stage28a.checkpoint.v1":
        raise ValueError("Stage-42a teacher must be Stage-28a")
    for key in ("source_to_id", "state_to_id", "latent_calibration"):
        left = base_payload[key]
        right = teacher_payload[key]
        equal = latent_calibrations_match(left, right) if key == "latent_calibration" else left == right
        if not equal:
            raise ValueError(f"Stage-42a base/teacher metadata differ: {key}")
    target_state_key = f"dynamicnuclearnet::{TARGET_LABEL}"
    target_state_id = int(base_payload["state_to_id"][target_state_key])
    if int(teacher_payload["state_to_id"][target_state_key]) != target_state_id:
        raise ValueError("3T3 state ID differs between teacher and student")

    student.requires_grad_(False)
    trainable = student.target_state_response_parameters()
    for parameter in trainable:
        parameter.requires_grad_(True)
    teacher.requires_grad_(False).eval()
    parameter_count = sum(parameter.numel() for parameter in trainable)
    if parameter_count != 3840:
        raise ValueError(f"Stage-42a parameter budget changed: {parameter_count}")
    bundle = FixedMask3T3ResponseDistiller(
        student.to(device), teacher.to(device), target_state_id=target_state_id
    ).to(device)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=0.0
    )

    train_dataset = FixedMask3T3ResponseDataset(
        train_anchors, cache_dir=args.cache_dir, augment=True
    )
    val_dataset = FixedMask3T3ResponseDataset(
        val_anchors, cache_dir=args.cache_dir, augment=False
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    val_indices = list(range(rank, len(val_dataset), world_size))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_indices,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    if world_size > 1:
        bundle = DistributedDataParallel(bundle, device_ids=[local_rank])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_path = args.output_dir / "checkpoint_last.pt"
    best_path = args.output_dir / "checkpoint_best.pt"
    history_path = args.output_dir / "metrics.jsonl"
    start_epoch = 0
    best_val = float("inf")
    if args.resume and last_path.is_file():
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("unsupported Stage-42a resume checkpoint")
        module = bundle.module if isinstance(bundle, DistributedDataParallel) else bundle
        module.student.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"])
        best_val = float(payload["best_val_loss"])

    autocast_enabled = args.precision == "bf16" and device.type == "cuda"
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        bundle.train()
        module = bundle.module if isinstance(bundle, DistributedDataParallel) else bundle
        module.teacher.eval()
        train_totals = [0.0, 0.0, 0.0, 0.0]
        for batch in train_loader:
            clean = batch["clean"].to(device, non_blocking=True)
            background = batch["background"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            source = batch["source_id"].to(device, non_blocking=True)
            anchor_class = batch["anchor_class_id"].to(device, non_blocking=True)
            timesteps = torch.randint(0, 1000, (len(clean),), device=device)
            noise = torch.randn_like(clean)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=autocast_enabled, dtype=torch.bfloat16
            ):
                loss, diagnostics = bundle(
                    clean,
                    background,
                    mask,
                    source,
                    anchor_class,
                    timesteps,
                    noise,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            count = len(clean)
            train_totals[0] += float(loss.detach()) * count
            train_totals[1] += float(diagnostics["masked_teacher_mse"]) * count
            train_totals[2] += float(diagnostics["prediction_abs_delta"]) * count
            train_totals[3] += count

        bundle.eval()
        eval_bundle = bundle.module if isinstance(bundle, DistributedDataParallel) else bundle
        val_totals = [0.0, 0.0, 0.0]
        with torch.inference_mode():
            for batch in val_loader:
                clean = batch["clean"].to(device, non_blocking=True)
                background = batch["background"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                source = batch["source_id"].to(device, non_blocking=True)
                anchor_class = batch["anchor_class_id"].to(device, non_blocking=True)
                noise, timesteps = fixed_validation_inputs(
                    clean.shape,
                    batch["cache_index"],
                    device=device,
                    seed=args.seed + 200000,
                )
                with torch.amp.autocast(
                    "cuda", enabled=autocast_enabled, dtype=torch.bfloat16
                ):
                    loss, diagnostics = eval_bundle(
                        clean,
                        background,
                        mask,
                        source,
                        anchor_class,
                        timesteps,
                        noise,
                    )
                count = len(clean)
                val_totals[0] += float(loss) * count
                val_totals[1] += float(diagnostics["masked_teacher_mse"]) * count
                val_totals[2] += count

        totals = reduce_totals(train_totals + val_totals, device)
        train_count = max(1.0, totals[3])
        val_count = max(1.0, totals[6])
        metrics = {
            "epoch": epoch,
            "train_response_distillation_loss": totals[0] / train_count,
            "train_masked_teacher_mse": totals[1] / train_count,
            "train_prediction_abs_delta": totals[2] / train_count,
            "val_response_distillation_loss": totals[4] / val_count,
            "val_masked_teacher_mse": totals[5] / val_count,
        }
        if rank == 0:
            local_module = bundle.module if isinstance(bundle, DistributedDataParallel) else bundle
            payload = {
                "schema": CHECKPOINT_SCHEMA,
                "epoch": epoch,
                "best_val_loss": min(best_val, metrics["val_response_distillation_loss"]),
                "training_role": "fixed_mask_3t3_target_only_response_distillation",
                "model_config": local_module.student.model_config,
                "model": local_module.student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "source_to_id": base_payload["source_to_id"],
                "state_to_id": base_payload["state_to_id"],
                "latent_calibration": base_payload["latent_calibration"],
                "vae_checkpoint": base_payload["vae_checkpoint"],
                "vae_checkpoint_sha256": base_payload["vae_checkpoint_sha256"],
                "initial_checkpoint": str(args.base_checkpoint.resolve()),
                "initial_checkpoint_sha256": sha256_file(args.base_checkpoint),
                "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
                "teacher_checkpoint_sha256": sha256_file(args.teacher_checkpoint),
                "target_state": TARGET_LABEL,
                "target_state_class": TARGET_CLASS_ID,
                "target_state_response_class": TARGET_CLASS_ID,
                "frozen_stage36c_feature_gate": 5,
                "trainable_scope": "target_state_response_affines_only",
                "trainable_parameters": parameter_count,
                "training_objective": "fixed_anchor_mask_raw_v_teacher_distillation",
                "timestep_range": [0, 999],
                "train_anchor_records": len(train_anchors),
                "val_anchor_records": len(val_anchors),
                "seed": args.seed,
                "metrics": metrics,
                "test_accessed": False,
            }
            torch.save(payload, last_path)
            if metrics["val_response_distillation_loss"] < best_val:
                best_val = metrics["val_response_distillation_loss"]
                payload["best_val_loss"] = best_val
                torch.save(payload, best_path)
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")
            print(
                f"epoch={epoch} train={metrics['train_response_distillation_loss']:.6f} "
                f"val={metrics['val_response_distillation_loss']:.6f}",
                flush=True,
            )
        if dist.is_initialized():
            dist.barrier()

    if rank == 0:
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        meta_keys = (
            "schema",
            "epoch",
            "best_val_loss",
            "training_role",
            "initial_checkpoint",
            "initial_checkpoint_sha256",
            "teacher_checkpoint",
            "teacher_checkpoint_sha256",
            "target_state",
            "target_state_class",
            "target_state_response_class",
            "frozen_stage36c_feature_gate",
            "trainable_scope",
            "trainable_parameters",
            "training_objective",
            "timestep_range",
            "train_anchor_records",
            "val_anchor_records",
            "seed",
            "test_accessed",
        )
        (args.output_dir / "best_meta.json").write_text(
            json.dumps({key: best[key] for key in meta_keys}, indent=2) + "\n",
            encoding="utf-8",
        )
    cleanup_distributed()


if __name__ == "__main__":
    main()
