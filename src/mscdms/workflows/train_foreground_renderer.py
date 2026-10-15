#!/usr/bin/env python3
"""Train foreground renderer."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mscdms.distributed import cleanup_distributed, get_device, setup_distributed
from mscdms.foreground_renderer import (
    ForegroundRendererUNet,
    compose_exact_background,
    foreground_reconstruction_loss,
)
from mscdms.mask_surrogate import FrozenSourceMaskSurrogates, mask_surrogate_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest_audit", type=Path, required=True)
    parser.add_argument("--bbbc021_checkpoint", type=Path, required=True)
    parser.add_argument("--dynamicnuclearnet_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--condition_dim", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mask_loss_weight", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_image(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(
            image.convert("RGB").resize(
                (image_size, image_size), Image.Resampling.BICUBIC
            ),
            dtype=np.float32,
        ) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    flat = tensor.view(3, -1)
    low = torch.quantile(flat, 0.01, dim=1).view(3, 1, 1)
    high = torch.quantile(flat, 0.99, dim=1).view(3, 1, 1)
    high = torch.maximum(high, low + 1e-6)
    return ((tensor - low) / (high - low)).clamp(0.0, 1.0) * 2.0 - 1.0


def load_mask(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(
            image.convert("L").resize(
                (image_size, image_size), Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        )
    mask = torch.from_numpy((array > 0).astype(np.float32)).unsqueeze(0)
    if not bool(mask.any()):
        raise ValueError(f"empty foreground mask: {path}")
    return mask


class ForegroundRendererDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        data_dir: Path,
        image_size: int,
        augment: bool,
    ) -> None:
        self.rows = rows
        self.data_dir = data_dir
        self.image_size = image_size
        self.augment = augment
        self.balance_keys = [
            (
                str(row["source_name"]),
                str(row["foreground_renderer"]["state_key"]),
                str(row["foreground_renderer"]["morphology"]),
            )
            for row in rows
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        target = normalize_image(self.data_dir / row["image_path"], self.image_size)
        mask = load_mask(
            self.data_dir / row["target_mask_path"], self.image_size
        )
        if self.augment:
            if torch.rand(()) < 0.5:
                target = target.flip(-1)
                mask = mask.flip(-1)
            if torch.rand(()) < 0.5:
                target = target.flip(-2)
                mask = mask.flip(-2)
        # Inside-mask pixels deliberately carry no exemplar RGB.  This keeps
        # the renderer from copying a factual foreground and matches inference.
        background = target * (1.0 - mask)
        payload = row["foreground_renderer"]
        return {
            "target": target,
            "background": background,
            "mask": mask,
            "source_id": torch.tensor(payload["source_id"], dtype=torch.long),
            "state_id": torch.tensor(payload["state_id"], dtype=torch.long),
            "sample_id": str(row["sample_id"]),
        }


class BalancedDistributedSampler(Sampler[int]):
    def __init__(
        self,
        keys: list[tuple[str, str, str]],
        *,
        rank: int,
        world_size: int,
        seed: int,
    ) -> None:
        if len(keys) % world_size:
            raise ValueError("dataset size must be divisible by world size")
        counts = Counter(keys)
        self.weights = torch.tensor(
            [1.0 / counts[key] for key in keys], dtype=torch.double
        )
        self.size = len(keys)
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.size // self.world_size

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        global_indices = torch.multinomial(
            self.weights,
            self.size,
            replacement=True,
            generator=generator,
        ).tolist()
        return iter(global_indices[self.rank :: self.world_size])


class RankSliceSampler(Sampler[int]):
    def __init__(self, size: int, *, rank: int, world_size: int) -> None:
        if size % world_size:
            raise ValueError("validation size must be divisible by world size")
        self.indices = list(range(rank, size, world_size))

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)


def reduce_totals(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    module = model.module if isinstance(model, DistributedDataParallel) else model
    return module.state_dict()


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = setup_distributed("nccl")
    device = get_device(local_rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)

    rows = read_jsonl(args.manifest)
    audit = json.loads(args.manifest_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or audit.get("test_accessed") is not False:
        raise ValueError("foreground-renderer manifest audit did not pass")
    train_rows = [row for row in rows if row.get("split") == "train"]
    val_rows = [row for row in rows if row.get("split") == "val"]
    if len(train_rows) != 10000 or len(val_rows) != 1000:
        raise ValueError("expected 10,000 train and 1,000 val records")

    train_dataset = ForegroundRendererDataset(
        train_rows,
        data_dir=args.data_dir,
        image_size=args.image_size,
        augment=True,
    )
    val_dataset = ForegroundRendererDataset(
        val_rows,
        data_dir=args.data_dir,
        image_size=args.image_size,
        augment=False,
    )
    train_sampler = BalancedDistributedSampler(
        train_dataset.balance_keys,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
    )
    val_sampler = RankSliceSampler(len(val_dataset), rank=rank, world_size=world_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    model = ForegroundRendererUNet(
        num_sources=len(audit["source_to_id"]),
        num_states=len(audit["state_to_id"]),
        base_channels=args.base_channels,
        condition_dim=args.condition_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    surrogate = FrozenSourceMaskSurrogates(
        bbbc021_checkpoint=args.bbbc021_checkpoint,
        dynamicnuclearnet_checkpoint=args.dynamicnuclearnet_checkpoint,
    ).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_path = args.output_dir / "checkpoint_last.pt"
    best_path = args.output_dir / "checkpoint_best.pt"
    history_path = args.output_dir / "metrics.jsonl"
    start_epoch = 0
    best_val = float("inf")
    if args.resume and last_path.is_file():
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        target = model.module if isinstance(model, DistributedDataParallel) else model
        target.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"])
        best_val = float(payload["best_val_loss"])
        if rank == 0:
            print(f"resuming completed epoch {start_epoch}", flush=True)

    autocast_enabled = args.precision == "bf16" and device.type == "cuda"
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        train_reconstruction = 0.0
        train_mask = 0.0
        train_count = 0
        for batch in train_loader:
            target = batch["target"].to(device, non_blocking=True)
            background = batch["background"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            source_ids = batch["source_id"].to(device, non_blocking=True)
            state_ids = batch["state_id"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda",
                enabled=autocast_enabled,
                dtype=torch.bfloat16,
            ):
                foreground = model(background, mask, source_ids, state_ids)
                reconstruction, _ = foreground_reconstruction_loss(
                    foreground, target, mask
                )
                composite = compose_exact_background(background, foreground, mask)
                surrogate_logits = surrogate(composite, source_ids)
                surrogate_target = F.interpolate(
                    mask,
                    size=surrogate_logits.shape[-2:],
                    mode="nearest",
                )
                mask_loss, _ = mask_surrogate_loss(
                    surrogate_logits, surrogate_target
                )
                loss = reconstruction + args.mask_loss_weight * mask_loss
            loss.backward()
            optimizer.step()
            count = len(target)
            train_loss += float(loss.detach()) * count
            train_reconstruction += float(reconstruction.detach()) * count
            train_mask += float(mask_loss.detach()) * count
            train_count += count

        model.eval()
        val_loss = 0.0
        val_reconstruction = 0.0
        val_mask = 0.0
        val_count = 0
        with torch.inference_mode():
            for batch in val_loader:
                target = batch["target"].to(device, non_blocking=True)
                background = batch["background"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                source_ids = batch["source_id"].to(device, non_blocking=True)
                state_ids = batch["state_id"].to(device, non_blocking=True)
                with torch.amp.autocast(
                    "cuda",
                    enabled=autocast_enabled,
                    dtype=torch.bfloat16,
                ):
                    foreground = model(background, mask, source_ids, state_ids)
                    reconstruction, _ = foreground_reconstruction_loss(
                        foreground, target, mask
                    )
                    composite = compose_exact_background(
                        background, foreground, mask
                    )
                    logits = surrogate(composite, source_ids)
                    surrogate_target = F.interpolate(
                        mask, size=logits.shape[-2:], mode="nearest"
                    )
                    mask_loss, _ = mask_surrogate_loss(logits, surrogate_target)
                    loss = reconstruction + args.mask_loss_weight * mask_loss
                count = len(target)
                val_loss += float(loss) * count
                val_reconstruction += float(reconstruction) * count
                val_mask += float(mask_loss) * count
                val_count += count

        totals = reduce_totals(
            [
                train_loss,
                train_reconstruction,
                train_mask,
                float(train_count),
                val_loss,
                val_reconstruction,
                val_mask,
                float(val_count),
            ],
            device,
        )
        train_count_global = max(1.0, totals[3])
        val_count_global = max(1.0, totals[7])
        metrics = {
            "epoch": epoch,
            "train_loss": totals[0] / train_count_global,
            "train_reconstruction": totals[1] / train_count_global,
            "train_mask_loss": totals[2] / train_count_global,
            "val_loss": totals[4] / val_count_global,
            "val_reconstruction": totals[5] / val_count_global,
            "val_mask_loss": totals[6] / val_count_global,
        }
        if rank == 0:
            payload = {
                "schema": "exp4.foreground_renderer.stage26.checkpoint.v1",
                "epoch": epoch,
                "best_val_loss": min(best_val, metrics["val_loss"]),
                "model_config": {
                    "num_sources": len(audit["source_to_id"]),
                    "num_states": len(audit["state_to_id"]),
                    "base_channels": args.base_channels,
                    "condition_dim": args.condition_dim,
                },
                "model": model_state(model),
                "optimizer": optimizer.state_dict(),
                "source_to_id": audit["source_to_id"],
                "state_to_id": audit["state_to_id"],
                "image_size": args.image_size,
                "mask_loss_weight": args.mask_loss_weight,
                "seed": args.seed,
                "train_records": len(train_rows),
                "val_records": len(val_rows),
                "test_accessed": False,
                "metrics": metrics,
            }
            torch.save(payload, last_path)
            if metrics["val_loss"] < best_val:
                best_val = metrics["val_loss"]
                payload["best_val_loss"] = best_val
                torch.save(payload, best_path)
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")
            print(
                f"epoch={epoch} train={metrics['train_loss']:.6f} "
                f"val={metrics['val_loss']:.6f} best={best_val:.6f}",
                flush=True,
            )
        if dist.is_initialized():
            dist.barrier()

    if rank == 0:
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        (args.output_dir / "best_meta.json").write_text(
            json.dumps(
                {
                    key: best[key]
                    for key in (
                        "schema",
                        "epoch",
                        "best_val_loss",
                        "model_config",
                        "source_to_id",
                        "state_to_id",
                        "image_size",
                        "mask_loss_weight",
                        "seed",
                        "train_records",
                        "val_records",
                        "test_accessed",
                    )
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    cleanup_distributed()


