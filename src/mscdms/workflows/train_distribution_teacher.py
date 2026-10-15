#!/usr/bin/env python3
"""Train distribution teacher."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mscdms.control_evaluator import sha256_file
from mscdms.distributed import cleanup_distributed, get_device, setup_distributed
from mscdms.factor_surrogate import (
    AdaptedDinoStateJudge,
    population_energy_distance_to_bank,
)
from mscdms.foreground_renderer import compose_exact_background
from mscdms.latent_foreground_diffusion import (
    denormalize_high_latent,
    initialize_latent_foreground_state_attention_delta,
    initialize_latent_foreground_state_feature_delta,
    latent_calibrations_match,
    load_latent_foreground_diffusion,
    load_original_v4_high_vae,
)
from mscdms.workflows.train_foreground_renderer import (
    RankSliceSampler,
    load_mask,
    normalize_image,
    reduce_totals,
)


DNN_LABELS = ("3t3", "hek293", "hela", "pc-3", "raw264", "rpe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--manifest_audit", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--initial_checkpoint", type=Path, required=True)
    parser.add_argument("--vae_checkpoint", type=Path, required=True)
    parser.add_argument("--dino_model_dir", type=Path, required=True)
    parser.add_argument("--dnn_state_judge", type=Path, required=True)
    parser.add_argument("--state_distribution_bank", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--state_weight", type=float, default=0.05)
    parser.add_argument("--population_size", type=int, default=2)
    parser.add_argument("--timestep_min", type=int, default=0)
    parser.add_argument("--timestep_max", type=int, default=249)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--enable_state_distribution", action="store_true")
    parser.add_argument("--state_attention_delta_rank", type=int, default=0)
    parser.add_argument("--train_state_attention_delta_only", action="store_true")
    parser.add_argument("--state_feature_delta_classes", type=int, default=0)
    parser.add_argument("--train_state_feature_delta_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class DNNLatentCacheDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        cache_dir: Path,
        data_dir: Path,
        augment: bool,
        load_rgb: bool,
        state_to_class: dict[int, int],
        target_state_ids: tuple[int, ...],
    ) -> None:
        self.rows = rows
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.augment = augment
        self.load_rgb = load_rgb
        self.state_to_class = state_to_class
        self.target_state_ids = target_state_ids
        self._target: np.ndarray | None = None
        self._background: np.ndarray | None = None
        self._masks: np.ndarray | None = None
        self.factual_classes = [
            self.state_to_class[int(row["state_id"])] for row in rows
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

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        if isinstance(index, tuple):
            row_index, target_class = map(int, index)
        else:
            row_index, target_class = int(index), -1
        self._open()
        assert self._target is not None
        assert self._background is not None
        assert self._masks is not None
        row = self.rows[row_index]
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
        rgb_background = torch.empty(0)
        mask_rgb = torch.empty(0)
        if self.load_rgb:
            image = normalize_image(
                self.data_dir / str(row["image_path"]), 512
            )
            mask_rgb = load_mask(
                self.data_dir / str(row["target_mask_path"]), 512
            )
            rgb_background = image * (1.0 - mask_rgb)
        flip_x = self.augment and bool(torch.rand(()) < 0.5)
        flip_y = self.augment and bool(torch.rand(()) < 0.5)
        for dimension, enabled in ((-1, flip_x), (-2, flip_y)):
            if enabled:
                target = target.flip(dimension)
                background = background.flip(dimension)
                mask = mask.flip(dimension)
                if self.load_rgb:
                    rgb_background = rgb_background.flip(dimension)
                    mask_rgb = mask_rgb.flip(dimension)
        factual_class = self.state_to_class[int(row["state_id"])]
        if target_class >= 0 and target_class == factual_class:
            raise ValueError("counterfactual state target equals factual state")
        target_state_id = (
            self.target_state_ids[target_class] if target_class >= 0 else -1
        )
        return {
            "target": target,
            "background": background,
            "mask": mask,
            "rgb_background": rgb_background,
            "mask_rgb": mask_rgb,
            "source_id": torch.tensor(row["source_id"], dtype=torch.long),
            "state_id": torch.tensor(row["state_id"], dtype=torch.long),
            "factual_class": torch.tensor(factual_class, dtype=torch.long),
            "target_class": torch.tensor(target_class, dtype=torch.long),
            "target_state_id": torch.tensor(target_state_id, dtype=torch.long),
            "cache_index": torch.tensor(cache_index, dtype=torch.long),
        }


class CounterfactualPopulationSampler(Sampler[tuple[int, int]]):
    """Emit local, same-target populations while excluding factual matches."""

    def __init__(
        self,
        factual_classes: list[int],
        *,
        population_size: int,
        num_classes: int,
        rank: int,
        world_size: int,
        seed: int,
    ) -> None:
        if population_size <= 1:
            raise ValueError("population_size must exceed one")
        self.factual_classes = factual_classes
        self.population_size = int(population_size)
        self.num_classes = int(num_classes)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch = 0
        self.groups_per_rank = len(factual_classes) // (
            population_size * world_size
        )
        if self.groups_per_rank <= 0:
            raise ValueError("dataset is too small for grouped DDP sampling")
        self.allowed = {
            target: [
                index
                for index, factual in enumerate(factual_classes)
                if factual != target
            ]
            for target in range(num_classes)
        }
        if any(len(values) < population_size for values in self.allowed.values()):
            raise ValueError("a target state has too few counterfactual anchors")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.groups_per_rank * self.population_size

    def __iter__(self) -> Iterator[tuple[int, int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        target_offset = int(
            torch.randint(self.num_classes, (1,), generator=generator).item()
        )
        groups: list[list[tuple[int, int]]] = []
        for group_index in range(self.groups_per_rank):
            target = (
                target_offset + self.rank + group_index
            ) % self.num_classes
            allowed = self.allowed[target]
            chosen = torch.randint(
                len(allowed),
                (self.population_size,),
                generator=generator,
            ).tolist()
            groups.append([(allowed[value], target) for value in chosen])
        return iter([item for group in groups for item in group])


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


def decoded_dino_pixels(decoded_rgb: torch.Tensor) -> torch.Tensor:
    pixels = (decoded_rgb.float() * 0.5 + 0.5).clamp(0.0, 1.0)
    pixels = F.interpolate(
        pixels,
        size=(256, 256),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )[:, :, 16:240, 16:240]
    mean = pixels.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = pixels.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (pixels - mean) / std


def validate_bank(
    path: Path,
    *,
    judge_checkpoint: Path,
    model_dir: Path,
) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "exp4.dnn.real_state_logit_bank.v1":
        raise ValueError("unsupported DNN state distribution bank")
    if payload.get("split") != "train" or payload.get("id_to_label") != list(
        DNN_LABELS
    ):
        raise ValueError("DNN state bank split or class order mismatch")
    if payload.get("judge_checkpoint_sha256") != sha256_file(judge_checkpoint):
        raise ValueError("DNN state bank judge hash mismatch")
    if payload.get("base_model_sha256") != sha256_file(
        model_dir / "model.safetensors"
    ):
        raise ValueError("DNN state bank backbone hash mismatch")
    result = {
        "logits": torch.as_tensor(payload["logits"], dtype=torch.float32),
        "labels": torch.as_tensor(payload["labels"], dtype=torch.long),
        "mean": torch.as_tensor(payload["global_mean"], dtype=torch.float32),
        "std": torch.as_tensor(payload["global_std"], dtype=torch.float32),
        "within": torch.as_tensor(
            payload["class_within_distance"], dtype=torch.float32
        ),
    }
    if result["logits"].ndim != 2 or result["logits"].shape[1] != 6:
        raise ValueError("DNN state bank logits must have shape [n,6]")
    if set(result["labels"].tolist()) != set(range(6)):
        raise ValueError("DNN state bank does not cover all six states")
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size != args.population_size:
        raise ValueError("local batch size must equal population size")
    if not 0 <= args.timestep_min <= args.timestep_max < 1000:
        raise ValueError("invalid counterfactual timestep range")
    if args.enable_state_distribution and args.state_weight <= 0:
        raise ValueError("enabled state distribution requires positive weight")
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
    target_state_ids = tuple(
        int(manifest_audit["state_to_id"][f"dynamicnuclearnet::{label}"])
        for label in DNN_LABELS
    )
    state_to_class = {
        state_id: class_id
        for class_id, state_id in enumerate(target_state_ids)
    }
    rows = read_jsonl(args.cache_dir / "index.jsonl")
    train_rows = [
        row
        for row in rows
        if row["split"] == "train" and row["source_name"] == "dynamicnuclearnet"
    ]
    val_rows = [
        row
        for row in rows
        if row["split"] == "val" and row["source_name"] == "dynamicnuclearnet"
    ]
    if len(train_rows) != 5000 or len(val_rows) != 500:
        raise ValueError("expected 5,000 DNN train and 500 DNN validation rows")
    if Counter(int(row["state_id"]) for row in train_rows).keys() != set(
        target_state_ids
    ):
        raise ValueError("DNN train cache does not cover all formal states")
    train_dataset = DNNLatentCacheDataset(
        train_rows,
        cache_dir=args.cache_dir,
        data_dir=args.data_dir,
        augment=True,
        load_rgb=args.enable_state_distribution,
        state_to_class=state_to_class,
        target_state_ids=target_state_ids,
    )
    val_dataset = DNNLatentCacheDataset(
        val_rows,
        cache_dir=args.cache_dir,
        data_dir=args.data_dir,
        augment=False,
        load_rgb=False,
        state_to_class=state_to_class,
        target_state_ids=target_state_ids,
    )
    train_sampler = CounterfactualPopulationSampler(
        train_dataset.factual_classes,
        population_size=args.population_size,
        num_classes=len(DNN_LABELS),
        rank=rank,
        world_size=world_size,
        seed=args.seed,
    )
    # Unlike the original 1,000-row validation, the DNN-only 500-row slice is
    # not divisible by eight. Uneven, non-repeating rank slices are valid here.
    val_sampler = list(range(rank, len(val_dataset), world_size))
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

    adapter_modes = int(args.train_state_attention_delta_only) + int(
        args.train_state_feature_delta_only
    )
    if adapter_modes > 1:
        raise ValueError("state attention and feature delta modes are mutually exclusive")
    if adapter_modes and not args.enable_state_distribution:
        raise ValueError("state-delta-only training requires state distribution")
    if args.train_state_attention_delta_only:
        if args.state_attention_delta_rank <= 0:
            raise ValueError("state-attention-only training requires a positive rank")
        model, initial = initialize_latent_foreground_state_attention_delta(
            args.initial_checkpoint,
            rank=args.state_attention_delta_rank,
        )
        model.requires_grad_(False)
        trainable_parameters = model.state_attention_delta_parameters()
        if not trainable_parameters:
            raise ValueError("state attention delta has no trainable parameters")
        for parameter in trainable_parameters:
            parameter.requires_grad_(True)
    elif args.train_state_feature_delta_only:
        if args.state_feature_delta_classes != len(DNN_LABELS):
            raise ValueError("state feature delta must use the six frozen DNN classes")
        model, initial = initialize_latent_foreground_state_feature_delta(
            args.initial_checkpoint,
            num_classes=args.state_feature_delta_classes,
        )
        model.requires_grad_(False)
        trainable_parameters = model.state_feature_delta_parameters()
        if not trainable_parameters:
            raise ValueError("state feature delta has no trainable parameters")
        for parameter in trainable_parameters:
            parameter.requires_grad_(True)
    else:
        model, initial = load_latent_foreground_diffusion(args.initial_checkpoint)
        trainable_parameters = list(model.parameters())
    if initial.get("schema") != "exp4.latent_foreground_diffusion.stage27a.checkpoint.v1":
        raise ValueError("Stage-28a must initialize from frozen Stage-27a")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    vae = None
    calibration = initial["latent_calibration"]
    judge = None
    bank = None
    if args.enable_state_distribution:
        vae, vae_calibration = load_original_v4_high_vae(
            args.vae_checkpoint, device=device
        )
        if not latent_calibrations_match(vae_calibration, calibration):
            raise ValueError("Stage-28a VAE calibration differs from Stage-27a")
        judge = AdaptedDinoStateJudge(
            checkpoint=args.dnn_state_judge,
            dino_model_dir=args.dino_model_dir,
        ).to(device).eval()
        bank = {
            key: value.to(device)
            for key, value in validate_bank(
                args.state_distribution_bank,
                judge_checkpoint=args.dnn_state_judge,
                model_dir=args.dino_model_dir,
            ).items()
        }
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_path = args.output_dir / "checkpoint_last.pt"
    best_path = args.output_dir / "checkpoint_best.pt"
    history_path = args.output_dir / "metrics.jsonl"
    start_epoch = 0
    best_val = float("inf")
    role = (
        "state_attention_delta"
        if args.train_state_attention_delta_only
        else (
            "state_feature_delta"
            if args.train_state_feature_delta_only
            else ("state_distribution" if args.enable_state_distribution else "matched_control")
        )
    )
    checkpoint_schema = (
        "exp4.latent_foreground_diffusion.stage28b.checkpoint.v1"
        if args.train_state_attention_delta_only
        else (
            "exp4.latent_foreground_diffusion.stage28c.checkpoint.v1"
            if args.train_state_feature_delta_only
            else "exp4.latent_foreground_diffusion.stage28a.checkpoint.v1"
        )
    )
    if args.resume and last_path.is_file():
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        if payload.get("schema") != checkpoint_schema:
            raise ValueError("unsupported Stage-28 resume checkpoint")
        if payload.get("training_role") != role:
            raise ValueError("Stage-28a resume role mismatch")
        target_model = model.module if isinstance(model, DistributedDataParallel) else model
        target_model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"])
        best_val = float(payload["best_val_loss"])
        if rank == 0:
            print(f"resuming completed epoch {start_epoch}", flush=True)

    autocast_enabled = args.precision == "bf16" and device.type == "cuda"
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        train_totals = [0.0, 0.0, 0.0, 0.0, 0.0]
        target_counts = Counter()
        for batch in train_loader:
            clean = batch["target"].to(device, non_blocking=True)
            background = batch["background"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            source = batch["source_id"].to(device, non_blocking=True)
            factual_state = batch["state_id"].to(device, non_blocking=True)
            target_class = batch["target_class"].to(device, non_blocking=True)
            target_state = batch["target_state_id"].to(device, non_blocking=True)
            if not bool((target_class == target_class[:1]).all()):
                raise ValueError("local population does not share one target state")
            if bool((target_state == factual_state).any()):
                raise ValueError("counterfactual target leaked factual state")
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=autocast_enabled, dtype=torch.bfloat16
            ):
                counterfactual_kwargs: dict[str, torch.Tensor] = {}
                if args.enable_state_distribution:
                    counterfactual_kwargs = {
                        "counterfactual_state_ids": target_state,
                        "counterfactual_anchor_state_ids": factual_state,
                        "counterfactual_target_state_class_ids": target_class,
                        "counterfactual_anchor_state_class_ids": batch[
                            "factual_class"
                        ].to(device, non_blocking=True),
                        "counterfactual_timesteps": torch.randint(
                            args.timestep_min,
                            args.timestep_max + 1,
                            (len(clean),),
                            device=device,
                        ),
                        "counterfactual_noise": torch.randn_like(clean),
                    }
                factual_loss, diagnostics = model(
                    clean,
                    background,
                    mask,
                    source,
                    factual_state,
                    **counterfactual_kwargs,
                )
                state_loss = factual_loss.new_zeros(())
                state_accuracy = factual_loss.new_zeros(())
                if args.enable_state_distribution:
                    predicted_x0 = diagnostics["counterfactual_x0"]
                    assert vae is not None and judge is not None and bank is not None
                    decoded = vae.decode(
                        denormalize_high_latent(predicted_x0, calibration)
                    ).clamp(-1.0, 1.0)
                    composite = compose_exact_background(
                        batch["rgb_background"].to(device, non_blocking=True),
                        decoded,
                        batch["mask_rgb"].to(device, non_blocking=True),
                    )
                    logits = judge(decoded_dino_pixels(composite))
                    state_loss = population_energy_distance_to_bank(
                        logits,
                        target_class,
                        population_size=args.population_size,
                        bank_logits=bank["logits"],
                        bank_labels=bank["labels"],
                        bank_mean=bank["mean"],
                        bank_std=bank["std"],
                        bank_within_distance=bank["within"],
                    )
                    state_accuracy = (
                        logits.float().mean(dim=0).argmax()
                        == target_class[0]
                    ).float()
                total_loss = factual_loss + args.state_weight * state_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            optimizer.step()
            count = len(clean)
            train_totals[0] += float(total_loss.detach()) * count
            train_totals[1] += float(factual_loss.detach()) * count
            train_totals[2] += float(state_loss.detach()) * count
            train_totals[3] += float(state_accuracy.detach()) * count
            train_totals[4] += count
            target_counts[int(target_class[0])] += count

        model.eval()
        val_totals = [0.0, 0.0]
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
                    val_loss, _ = model(
                        clean,
                        background,
                        mask,
                        source,
                        state,
                        timesteps=timesteps,
                        noise=noise,
                    )
                val_totals[0] += float(val_loss) * len(clean)
                val_totals[1] += len(clean)
        totals = reduce_totals(train_totals + val_totals, device)
        train_count = max(1.0, totals[4])
        val_count = max(1.0, totals[6])
        metrics = {
            "epoch": epoch,
            "training_role": role,
            "train_total_loss": totals[0] / train_count,
            "train_factual_loss": totals[1] / train_count,
            "train_state_distribution_loss": totals[2] / train_count,
            "train_state_population_accuracy": totals[3] / train_count,
            "val_factual_loss": totals[5] / val_count,
        }
        if rank == 0:
            module = model.module if isinstance(model, DistributedDataParallel) else model
            payload = {
                "schema": checkpoint_schema,
                "epoch": epoch,
                "best_val_loss": min(best_val, metrics["val_factual_loss"]),
                "training_role": role,
                "model_config": module.model_config,
                "model": module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "source_to_id": manifest_audit["source_to_id"],
                "state_to_id": manifest_audit["state_to_id"],
                "latent_calibration": calibration,
                "vae_checkpoint": initial["vae_checkpoint"],
                "vae_checkpoint_sha256": initial["vae_checkpoint_sha256"],
                "initial_checkpoint": str(args.initial_checkpoint.resolve()),
                "initial_checkpoint_sha256": sha256_file(args.initial_checkpoint),
                "state_distribution_bank": str(
                    args.state_distribution_bank.resolve()
                ),
                "state_distribution_bank_sha256": sha256_file(
                    args.state_distribution_bank
                ),
                "dnn_state_judge": str(args.dnn_state_judge.resolve()),
                "dnn_state_judge_sha256": sha256_file(args.dnn_state_judge),
                "population_size": args.population_size,
                "state_weight": (
                    args.state_weight if args.enable_state_distribution else 0.0
                ),
                "trainable_scope": (
                    "state_attention_delta_only"
                    if args.train_state_attention_delta_only
                    else (
                        "state_feature_delta_only"
                        if args.train_state_feature_delta_only
                        else "complete_foreground_model"
                    )
                ),
                "state_attention_delta_rank": (
                    args.state_attention_delta_rank
                    if args.train_state_attention_delta_only
                    else 0
                ),
                "state_feature_delta_classes": (
                    args.state_feature_delta_classes
                    if args.train_state_feature_delta_only
                    else 0
                ),
                "trainable_parameters": sum(
                    parameter.numel() for parameter in trainable_parameters
                ),
                "timestep_range": [args.timestep_min, args.timestep_max],
                "seed": args.seed,
                "train_records": len(train_rows),
                "val_records": len(val_rows),
                "test_accessed": False,
                "metrics": metrics,
                "target_counts_rank0": dict(sorted(target_counts.items())),
            }
            torch.save(payload, last_path)
            if metrics["val_factual_loss"] < best_val:
                best_val = metrics["val_factual_loss"]
                payload["best_val_loss"] = best_val
                torch.save(payload, best_path)
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")
            print(
                f"epoch={epoch} role={role} "
                f"train={metrics['train_total_loss']:.6f} "
                f"factual={metrics['train_factual_loss']:.6f} "
                f"state={metrics['train_state_distribution_loss']:.6f} "
                f"state_acc={metrics['train_state_population_accuracy']:.4f} "
                f"val={metrics['val_factual_loss']:.6f}",
                flush=True,
            )
        if dist.is_initialized():
            dist.barrier()

    if rank == 0:
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        keys = (
            "schema",
            "epoch",
            "best_val_loss",
            "training_role",
            "model_config",
            "source_to_id",
            "state_to_id",
            "initial_checkpoint",
            "initial_checkpoint_sha256",
            "state_distribution_bank",
            "state_distribution_bank_sha256",
            "dnn_state_judge",
            "dnn_state_judge_sha256",
            "population_size",
            "state_weight",
            "trainable_scope",
            "state_attention_delta_rank",
            "state_feature_delta_classes",
            "trainable_parameters",
            "timestep_range",
            "seed",
            "train_records",
            "val_records",
            "test_accessed",
        )
        (args.output_dir / "last_meta.json").write_text(
            json.dumps({key: last[key] for key in keys}, indent=2) + "\n",
            encoding="utf-8",
        )
    cleanup_distributed()


if __name__ == "__main__":
    main()
