#!/usr/bin/env python3
"""Support foreground renderer factual."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mscdms.foreground_renderer import (
    compose_exact_background,
    load_foreground_renderer,
)
from mscdms.mask_surrogate import FrozenSourceMaskSurrogates
from mscdms.workflows.support_foreground_exemplar import (
    classify_ratio,
    largest_component,
    mask_geometry,
)
from mscdms.workflows.train_foreground_renderer import (
    ForegroundRendererDataset,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bbbc021_checkpoint", type=Path, required=True)
    parser.add_argument("--dynamicnuclearnet_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    array = (
        ((tensor.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5)
        .round()
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    Image.fromarray(array, mode="RGB").save(path)


def dice_iou(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    intersection = float((left & right).sum())
    union = float((left | right).sum())
    dice = (2.0 * intersection + 1.0) / (left.sum() + right.sum() + 1.0)
    iou = (intersection + 1.0) / (union + 1.0)
    return float(dice), float(iou)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "mean_dice": float(np.mean([row["dice"] for row in rows])),
        "mean_iou": float(np.mean([row["iou"] for row in rows])),
        "morphology_accuracy": float(
            np.mean([row["morphology_correct"] for row in rows])
        ),
        "empty_prediction_rate": float(
            np.mean([row["predicted_empty"] for row in rows])
        ),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    renderer, checkpoint = load_foreground_renderer(args.checkpoint)
    renderer = renderer.to(device).eval()
    image_size = int(checkpoint["image_size"])
    if image_size != 512:
        raise ValueError("formal factual audit requires 512x512 renderer")
    rows = [row for row in read_jsonl(args.manifest) if row.get("split") == "val"]
    if len(rows) != 1000:
        raise ValueError(f"expected 1,000 validation rows, got {len(rows)}")
    row_by_id = {str(row["sample_id"]): row for row in rows}
    dataset = ForegroundRendererDataset(
        rows, data_dir=args.data_dir, image_size=512, augment=False
    )
    loader = DataLoader(
        dataset,
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
        for batch in loader:
            target = batch["target"].to(device, non_blocking=True)
            background = batch["background"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            source_ids = batch["source_id"].to(device, non_blocking=True)
            state_ids = batch["state_id"].to(device, non_blocking=True)
            with torch.amp.autocast(
                "cuda", enabled=device.type == "cuda", dtype=torch.bfloat16
            ):
                foreground = renderer(background, mask, source_ids, state_ids)
                composite = compose_exact_background(background, foreground, mask)
                logits = surrogate(composite, source_ids)
            predicted = logits.float().sigmoid() >= 0.5
            exact = F.interpolate(mask, size=predicted.shape[-2:], mode="nearest") >= 0.5
            for index, sample_id in enumerate(batch["sample_id"]):
                sample_id = str(sample_id)
                row = row_by_id[sample_id]
                output_path = image_dir / f"{sample_id}__stage26b_factual.png"
                save_tensor_image(composite[index], output_path)
                generated_id = f"{sample_id}__stage26b_factual"
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
                        "factual_reconstruction": True,
                        "paired_counterfactual_target_used": False,
                    }
                )
                target_mask = exact[index, 0].cpu().numpy()
                predicted_mask = predicted[index, 0].cpu().numpy()
                dice, iou = dice_iou(predicted_mask, target_mask)
                target_morphology = str(
                    row["foreground_renderer"]["morphology"]
                )
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
            print(f"factual reconstruction {completed}/{len(rows)}", flush=True)

    manifest_rows.sort(key=lambda row: row["generated_id"])
    per_image.sort(key=lambda row: row["generated_id"])
    manifest_path = args.output_dir / "generated_manifest.jsonl"
    per_image_path = args.output_dir / "mask_audit_per_image.jsonl"
    for path, values in ((manifest_path, manifest_rows), (per_image_path, per_image)):
        with path.open("w", encoding="utf-8") as handle:
            for row in values:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {
        "schema": "exp4.foreground_renderer.stage26b.factual_audit.v1",
        "status": "pass",
        "overall": summarize(per_image),
        "by_source": {
            source: summarize(
                [row for row in per_image if row["source_name"] == source]
            )
            for source in ("bbbc021", "dynamicnuclearnet")
        },
        "by_morphology": {
            morphology: summarize(
                [
                    row
                    for row in per_image
                    if row["target_morphology"] == morphology
                ]
            )
            for morphology in ("elongated", "oval", "round")
        },
        "n_generated": len(manifest_rows),
        "real_validation_masks_used": True,
        "real_validation_images_used_as_factual_input_and_reference": True,
        "paired_counterfactual_targets_used": False,
        "test_accessed": False,
    }
    (args.output_dir / "mask_audit_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("saved", manifest_path)
    print("saved", args.output_dir / "mask_audit_summary.json")


