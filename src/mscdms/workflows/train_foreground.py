#!/usr/bin/env python3
"""Train foreground."""

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
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mscdms.distributed import cleanup_distributed, get_device, setup_distributed
from mscdms.latent_foreground_diffusion import (
    MaskConditionedLatentForegroundDiffusion,
)
from mscdms.workflows.train_foreground_renderer import (
    BalancedDistributedSampler,
    RankSliceSampler,
    model_state,
    reduce_totals,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--manifest_audit", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--condition_dim", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class LatentCacheDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        cache_dir: Path,
        augment: bool,
    ) -> None:
        self.rows = rows
        self.cache_dir = cache_dir
        self.augment = augment
        self._target: np.ndarray | None = None
        self._background: np.ndarray | None = None
        self._masks: np.ndarray | None = None
        self.balance_keys = [
            (row["source_name"], row["state_key"], row["morphology"])
            for row in rows
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def _open(self) -> None:
        if self._target is None:
            self._target = np.load(
                self.cache_dir / "target_latents.npy", mmap_mode="r"
            )
            self._background = np.load(
                self.cache_dir / "background_latents.npy", mmap_mode="r"
            )
            self._masks = np.load(self.cache_dir / "masks.npy", mmap_mode="r")

    def __getitem__(self, index: int) -> dict[str, Any]:
        self._open()
        assert self._target is not None
        assert self._background is not None
        assert self._masks is not None
        row = self.rows[index]
        cache_index = int(row["cache_index"])
        target = torch.from_numpy(
            np.array(self._target[cache_index], dtype=np.float32, copy=True)
        )
        background = torch.from_numpy(
            np.array(self._background[cache_index], dtype=np.float32, copy=True)
        )
        mask = torch.from_numpy(
            np.array(self._masks[cache_index], dtype=np.float32, copy=True)
        )
        if self.augment:
            if torch.rand(()) < 0.5:
                target = target.flip(-1)
                background = background.flip(-1)
                mask = mask.flip(-1)
            if torch.rand(()) < 0.5:
                target = target.flip(-2)
                background = background.flip(-2)
                mask = mask.flip(-2)
        return {
            "target": target,
            "background": background,
            "mask": mask,
            "source_id": torch.tensor(row["source_id"], dtype=torch.long),
            "state_id": torch.tensor(row["state_id"], dtype=torch.long),
            "cache_index": torch.tensor(cache_index, dtype=torch.long),
        }


def fixed_validation_noise(
    shape: torch.Size,
    indices: torch.Tensor,
    *,
    device: torch.device,
    seed: int,
    timesteps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    noise: list[torch.Tensor] = []
    timestep_values: list[int] = []
    for index in indices.tolist():
        generator = torch.Generator(device=device).manual_seed(seed + int(index))
        noise.append(torch.randn(shape[1:], device=device, generator=generator))
        timestep_values.append((seed + int(index) * 7919) % timesteps)
    return torch.stack(noise), torch.tensor(timestep_values, device=device)


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
    rows = read_jsonl(args.cache_dir / "index.jsonl")
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    if len(train_rows) != 10000 or len(val_rows) != 1000:
        raise ValueError("expected 10,000 train and 1,000 validation cache rows")
    train_dataset = LatentCacheDataset(
        train_rows, cache_dir=args.cache_dir, augment=True
    )
    val_dataset = LatentCacheDataset(
        val_rows, cache_dir=args.cache_dir, augment=False
    )
    train_sampler = BalancedDistributedSampler(
        train_dataset.balance_keys,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
    )
    val_sampler = RankSliceSampler(
        len(val_dataset), rank=rank, world_size=world_size
    )
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
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    model = MaskConditionedLatentForegroundDiffusion(
        num_sources=len(manifest_audit["source_to_id"]),
        num_states=len(manifest_audit["state_to_id"]),
        latent_channels=4,
        condition_dim=args.condition_dim,
        base_channels=args.base_channels,
        input_resolution=64,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks=1,
        num_heads=4,
        num_train_timesteps=1000,
        prediction_type="v",
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "checkpoint_best.pt"
    last_path = args.output_dir / "checkpoint_last.pt"
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
        train_totals = [0.0, 0.0, 0.0, 0.0]
        for batch in train_loader:
            clean = batch["target"].to(device, non_blocking=True)
            background = batch["background"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            source = batch["source_id"].to(device, non_blocking=True)
            state = batch["state_id"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=autocast_enabled, dtype=torch.bfloat16
            ):
                loss, diagnostics = model(
                    clean, background, mask, source, state
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = len(clean)
            train_totals[0] += float(loss.detach()) * count
            train_totals[1] += float(diagnostics["inside_diffusion_mse"]) * count
            train_totals[2] += float(diagnostics["outside_diffusion_mse"]) * count
            train_totals[3] += count

        model.eval()
        val_totals = [0.0, 0.0, 0.0, 0.0]
        with torch.inference_mode():
            for batch in val_loader:
                clean = batch["target"].to(device, non_blocking=True)
                background = batch["background"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                source = batch["source_id"].to(device, non_blocking=True)
                state = batch["state_id"].to(device, non_blocking=True)
                indices = batch["cache_index"].to(device)
                noise, timesteps = fixed_validation_noise(
                    clean.shape,
                    indices,
                    device=device,
                    seed=args.seed + 100000,
                    timesteps=1000,
                )
                with torch.amp.autocast(
                    "cuda", enabled=autocast_enabled, dtype=torch.bfloat16
                ):
                    loss, diagnostics = model(
                        clean,
                        background,
                        mask,
                        source,
                        state,
                        timesteps=timesteps,
                        noise=noise,
                    )
                count = len(clean)
                val_totals[0] += float(loss) * count
                val_totals[1] += float(diagnostics["inside_diffusion_mse"]) * count
                val_totals[2] += float(diagnostics["outside_diffusion_mse"]) * count
                val_totals[3] += count
        totals = reduce_totals(train_totals + val_totals, device)
        train_count = max(1.0, totals[3])
        val_count = max(1.0, totals[7])
        metrics = {
            "epoch": epoch,
            "train_loss": totals[0] / train_count,
            "train_inside_mse": totals[1] / train_count,
            "train_outside_mse": totals[2] / train_count,
            "val_loss": totals[4] / val_count,
            "val_inside_mse": totals[5] / val_count,
            "val_outside_mse": totals[6] / val_count,
        }
        if rank == 0:
            module = model.module if isinstance(model, DistributedDataParallel) else model
            payload = {
                "schema": "exp4.latent_foreground_diffusion.stage27a.checkpoint.v1",
                "epoch": epoch,
                "best_val_loss": min(best_val, metrics["val_loss"]),
                "model_config": module.model_config,
                "model": model_state(model),
                "optimizer": optimizer.state_dict(),
                "source_to_id": manifest_audit["source_to_id"],
                "state_to_id": manifest_audit["state_to_id"],
                "latent_calibration": cache_audit["latent_calibration"],
                "vae_checkpoint": cache_audit["vae_checkpoint"],
                "vae_checkpoint_sha256": cache_audit["vae_checkpoint_sha256"],
                "cache_audit": str((args.cache_dir / "cache_audit.json").resolve()),
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
        keys = (
            "schema",
            "epoch",
            "best_val_loss",
            "model_config",
            "source_to_id",
            "state_to_id",
            "latent_calibration",
            "vae_checkpoint",
            "vae_checkpoint_sha256",
            "seed",
            "train_records",
            "val_records",
            "test_accessed",
        )
        (args.output_dir / "best_meta.json").write_text(
            json.dumps({key: best[key] for key in keys}, indent=2) + "\n",
            encoding="utf-8",
        )
    cleanup_distributed()


if __name__ == "__main__":
    main()
