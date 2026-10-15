#!/usr/bin/env python3
"""Resynthesize foreground."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mscdms.foreground_renderer import compose_exact_background
from mscdms.latent_foreground_diffusion import (
    denormalize_high_latent,
    load_latent_foreground_diffusion,
    load_original_v4_high_vae,
    normalize_high_latent,
)
from mscdms.mask_surrogate import FrozenSourceMaskSurrogates
from mscdms.workflows.support_foreground_exemplar import (
    classify_ratio,
    largest_component,
    mask_geometry,
)
from mscdms.workflows.support_foreground_renderer_factual import (
    dice_iou,
    summarize,
)
from mscdms.workflows.train_foreground_renderer import (
    ForegroundRendererDataset,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae_checkpoint", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bbbc021_checkpoint", type=Path, required=True)
    parser.add_argument("--dynamicnuclearnet_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class FactualLatentDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        data_dir: Path,
        cache_dir: Path,
        *,
        split: str,
    ):
        self.rows = rows
        self.split = split
        self.rgb = ForegroundRendererDataset(
            rows, data_dir=data_dir, image_size=512, augment=False
        )
        cache_rows = read_jsonl(cache_dir / "index.jsonl")
        self.cache_index = {
            str(row["sample_id"]): int(row["cache_index"])
            for row in cache_rows
            if row["split"] == split
        }
        self.background_path = cache_dir / "background_latents.npy"
        self._background: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._background is None:
            self._background = np.load(self.background_path, mmap_mode="r")
        item = self.rgb[index]
        if item["sample_id"] in self.cache_index:
            cache_index = self.cache_index[item["sample_id"]]
            item["background_latent"] = torch.from_numpy(
                np.array(self._background[cache_index], dtype=np.float32, copy=True)
            )
            item["background_latent_cached"] = True
        else:
            # Final-test images were deliberately excluded from the train/val
            # latent cache.  Encode only the explicit masked background at
            # inference time; no target foreground pixels enter this tensor.
            item["background_latent"] = torch.empty(0, dtype=torch.float32)
            item["background_latent_cached"] = False
        return item


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    array = (
        ((tensor.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5)
        .round()
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    Image.fromarray(array, mode="RGB").save(path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, checkpoint = load_latent_foreground_diffusion(args.checkpoint)
    model = model.to(device).eval()
    vae, calibration = load_original_v4_high_vae(
        args.vae_checkpoint, device=device
    )
    trained_calibration = checkpoint["latent_calibration"]
    if (
        bool(calibration["enabled"]) != bool(trained_calibration["enabled"])
        or tuple(calibration["mean"]) != tuple(trained_calibration["mean"])
        or tuple(calibration["std"]) != tuple(trained_calibration["std"])
    ):
        raise ValueError("generation VAE calibration differs from training cache")
    rows = [
        row for row in read_jsonl(args.manifest) if row.get("split") == args.split
    ]
    expected_count = 1000 if args.split == "val" else 4000
    if len(rows) != expected_count:
        raise ValueError(
            f"expected {expected_count:,} factual {args.split} rows, got {len(rows)}"
        )
    row_by_id = {str(row["sample_id"]): row for row in rows}
    loader = DataLoader(
        FactualLatentDataset(
            rows,
            args.data_dir,
            args.cache_dir,
            split=args.split,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    surrogate = FrozenSourceMaskSurrogates(
        bbbc021_checkpoint=args.bbbc021_checkpoint,
        dynamicnuclearnet_checkpoint=args.dynamicnuclearnet_checkpoint,
    ).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    completed = 0
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader):
            target = batch["target"].to(device, non_blocking=True)
            background_rgb = batch["background"].to(device, non_blocking=True)
            cached_flags = batch["background_latent_cached"]
            if bool(cached_flags.all()):
                background_latent = batch["background_latent"].to(
                    device, non_blocking=True
                )
            elif bool((~cached_flags).all()):
                with torch.amp.autocast(
                    "cuda", enabled=device.type == "cuda", dtype=torch.bfloat16
                ):
                    background_mean, _ = vae.encode(background_rgb)
                background_latent = normalize_high_latent(
                    background_mean.float(), calibration
                )
            else:
                raise ValueError("a factual batch mixes cached and uncached latents")
            mask = batch["mask"].to(device, non_blocking=True)
            source_ids = batch["source_id"].to(device, non_blocking=True)
            state_ids = batch["state_id"].to(device, non_blocking=True)
            generator = torch.Generator(device=device).manual_seed(
                args.seed + batch_number
            )
            with torch.amp.autocast(
                "cuda", enabled=device.type == "cuda", dtype=torch.bfloat16
            ):
                sampled = model.sample(
                    background_latent,
                    mask,
                    source_ids,
                    state_ids,
                    steps=args.steps,
                    generator=generator,
                )
                foreground = vae.decode(
                    denormalize_high_latent(sampled, calibration)
                ).clamp(-1.0, 1.0)
                composite = compose_exact_background(
                    background_rgb, foreground, mask
                )
                logits = surrogate(composite, source_ids)
            predicted = logits.float().sigmoid() >= 0.5
            exact = F.interpolate(mask, size=predicted.shape[-2:], mode="nearest") >= 0.5
            for index, sample_id_value in enumerate(batch["sample_id"]):
                sample_id = str(sample_id_value)
                row = row_by_id[sample_id]
                generated_id = (
                    f"{sample_id}__stage27a_factual"
                    if args.split == "val"
                    else f"{sample_id}__stage42b_final_factual"
                )
                output_path = image_dir / f"{generated_id}.png"
                save_tensor_image(composite[index], output_path)
                manifest_rows.append(
                    {
                        "generated_id": generated_id,
                        "sample_id": sample_id,
                        "generated_image": str(output_path.resolve()),
                        "reference_image": str(
                            (args.data_dir / str(row["image_path"])).resolve()
                        ),
                        "subject": str(row["subject"]),
                        "conditioning_subject": str(row["subject"]),
                        "state": str(row["state"]),
                        "background": str(row["background"]),
                        "source_name": str(row["source_name"]),
                        "object_type": str(row["object_type"]),
                        "checkpoint": str(args.checkpoint.resolve()),
                        "factual_conditioning": True,
                        "stochastic_foreground": True,
                        "steps": args.steps,
                        "paired_counterfactual_target_used": False,
                    }
                )
                target_mask = exact[index, 0].cpu().numpy()
                predicted_mask = predicted[index, 0].cpu().numpy()
                dice, iou = dice_iou(predicted_mask, target_mask)
                target_morphology = str(row["foreground_renderer"]["morphology"])
                predicted_empty = not bool(predicted_mask.any())
                predicted_morphology = None
                if not predicted_empty:
                    predicted_morphology = mask_geometry(
                        largest_component(predicted_mask)
                    )["morphology"]
                per_image.append(
                    {
                        "generated_id": generated_id,
                        "sample_id": sample_id,
                        "source_name": str(row["source_name"]),
                        "target_morphology": target_morphology,
                        "predicted_morphology": predicted_morphology,
                        "morphology_correct": predicted_morphology == target_morphology,
                        "predicted_empty": predicted_empty,
                        "dice": dice,
                        "iou": iou,
                    }
                )
            completed += len(target)
            print(f"stochastic factual generation {completed}/{len(rows)}", flush=True)
    manifest_rows.sort(key=lambda row: row["generated_id"])
    per_image.sort(key=lambda row: row["generated_id"])
    for path, values in (
        (args.output_dir / "generated_manifest.jsonl", manifest_rows),
        (args.output_dir / "mask_audit_per_image.jsonl", per_image),
    ):
        with path.open("w", encoding="utf-8") as handle:
            for row in values:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = {
        "schema": "exp4.latent_foreground_diffusion.stage27a.factual_audit.v1",
        "status": "pass",
        "overall": summarize(per_image),
        "by_source": {
            source: summarize(
                [row for row in per_image if row["source_name"] == source]
            )
            for source in ("bbbc021", "dynamicnuclearnet")
        },
        "source_counts": dict(Counter(row["source_name"] for row in manifest_rows)),
        "n_generated": len(manifest_rows),
        "stochastic_foreground": True,
        "exact_background_outside_mask": True,
        "explicit_split_masks_used": True,
        "paired_counterfactual_targets_used": False,
        "split": args.split,
        "test_accessed": args.split == "test",
    }
    (args.output_dir / "mask_audit_summary.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("saved", args.output_dir / "generated_manifest.jsonl")
    print("saved", args.output_dir / "mask_audit_summary.json")


if __name__ == "__main__":
    main()
