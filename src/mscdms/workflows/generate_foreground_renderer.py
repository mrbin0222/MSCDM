#!/usr/bin/env python3
"""Generate foreground renderer."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mscdms.foreground_renderer import (
    compose_exact_background,
    load_foreground_renderer,
)
from mscdms.workflows.build_spatial_morphology_oracle import (
    nearest_background_fill,
    predict_masks,
    sha256_file,
)
from mscdms.workflows.prepare_mask_geometry import largest_component


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline_manifest", type=Path, required=True)
    parser.add_argument("--stage25_manifest", type=Path, required=True)
    parser.add_argument("--training_manifest", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--bbbc021_checkpoint", type=Path, required=True)
    parser.add_argument("--dynamicnuclearnet_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--mask_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_array(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32) / 255.0
    flattened = values.reshape(-1, 3)
    low = np.quantile(flattened, 0.01, axis=0).reshape(1, 1, 3)
    high = np.quantile(flattened, 0.99, axis=0).reshape(1, 1, 3)
    high = np.maximum(high, low + 1e-6)
    return np.clip((values - low) / (high - low), 0.0, 1.0) * 2.0 - 1.0


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(
            image.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC),
            dtype=np.uint8,
        )


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(
            image.convert("L").resize((512, 512), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
    mask = largest_component(array > 0)
    if not bool(mask.any()):
        raise ValueError(f"empty target mask: {path}")
    return mask


def tensor_image(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array, dtype=np.float32)).permute(2, 0, 1)


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
    model, checkpoint = load_foreground_renderer(args.checkpoint)
    model = model.to(device).eval()
    image_size = int(checkpoint["image_size"])
    if image_size != 512:
        raise ValueError("Stage-26 formal inference requires 512x512 renderer")

    baseline_all = read_jsonl(args.baseline_manifest)
    baseline = [row for row in baseline_all if row.get("target_attribute") == "identity"]
    selected = read_jsonl(args.stage25_manifest)
    if len(baseline_all) != 4800 or len(baseline) != 1600 or len(selected) != 1600:
        raise ValueError("expected 4,800 baseline and 1,600 identity records")
    baseline_by_id = {str(row["generated_id"]): row for row in baseline}
    selection_by_id = {str(row["generated_id"]): row for row in selected}
    if set(baseline_by_id) != set(selection_by_id):
        raise ValueError("baseline and Stage-25 selection IDs differ")

    training_by_id = {
        str(row["sample_id"]): row
        for row in read_jsonl(args.training_manifest)
        if row.get("split") == "train"
    }
    ordered_ids = sorted(baseline_by_id)
    ordered_baseline = [baseline_by_id[key] for key in ordered_ids]
    anchor_masks = predict_masks(
        ordered_baseline,
        bbbc_checkpoint=args.bbbc021_checkpoint,
        dnn_checkpoint=args.dynamicnuclearnet_checkpoint,
        batch_size=args.mask_batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []

    for generated_id, baseline_row, anchor_mask in zip(
        ordered_ids, ordered_baseline, anchor_masks
    ):
        selection = selection_by_id[generated_id]
        factorization = selection.get("foreground_background_factorization")
        if not isinstance(factorization, dict) or factorization.get("template_split") != "train":
            raise ValueError(f"invalid Stage-25 selection: {generated_id}")
        template_id = str(factorization["template_sample_id"])
        template = training_by_id.get(template_id)
        if template is None:
            raise ValueError(f"missing train template: {template_id}")
        source = str(baseline_row["source_name"])
        state_key = f"{source}::{factorization['target_state_label']}"
        if state_key not in checkpoint["state_to_id"]:
            raise ValueError(f"unknown renderer state: {state_key}")
        anchor_mask = largest_component(np.asarray(anchor_mask, dtype=bool))
        payloads.append(
            {
                "generated_id": generated_id,
                "baseline": baseline_row,
                "factorization": factorization,
                "template_id": template_id,
                "template_mask_path": str(
                    args.data_dir / str(template["target_mask_path"])
                ),
                "state_key": state_key,
                "anchor_mask_fraction": float(anchor_mask.mean()),
                "anchor_mask": anchor_mask,
                "source_id": int(checkpoint["source_to_id"][source]),
                "state_id": int(checkpoint["state_to_id"][state_key]),
            }
        )

    autocast_enabled = device.type == "cuda"
    for start in range(0, len(payloads), args.batch_size):
        end = min(len(payloads), start + args.batch_size)
        batch_items = payloads[start:end]
        batch_backgrounds: list[np.ndarray] = []
        batch_masks: list[np.ndarray] = []
        for item in batch_items:
            anchor = normalize_array(
                load_rgb(Path(str(item["baseline"]["generated_image"])))
            )
            target_mask = load_mask(Path(item["template_mask_path"]))
            filled = nearest_background_fill(
                anchor, item["anchor_mask"]
            ).astype(np.float32)
            # Remove all RGB evidence inside the requested support. The
            # renderer must infer content from source/state plus target mask.
            filled[target_mask] = 0.0
            batch_backgrounds.append(filled)
            batch_masks.append(target_mask)
        background_batch = torch.stack(
            [tensor_image(value) for value in batch_backgrounds]
        ).to(device)
        mask_batch = torch.stack(
            [
                torch.from_numpy(value.astype(np.float32)).unsqueeze(0)
                for value in batch_masks
            ]
        ).to(device)
        source_batch = torch.tensor(
            [item["source_id"] for item in batch_items],
            dtype=torch.long,
            device=device,
        )
        state_batch = torch.tensor(
            [item["state_id"] for item in batch_items],
            dtype=torch.long,
            device=device,
        )
        with torch.inference_mode(), torch.amp.autocast(
            "cuda", enabled=autocast_enabled, dtype=torch.bfloat16
        ):
            foreground = model(background_batch, mask_batch, source_batch, state_batch)
            composite = compose_exact_background(background_batch, foreground, mask_batch)
        for local_index, item in enumerate(batch_items):
            output_path = image_dir / f"{item['generated_id']}.png"
            save_tensor_image(composite[local_index], output_path)
            baseline_row = item["baseline"]
            output = dict(baseline_row)
            output["generated_image"] = str(output_path.resolve())
            output["checkpoint"] = f"{args.checkpoint.resolve()}::stage26_learned_foreground_renderer"
            output["foreground_background_factorization"] = {
                "schema": "exp4.foreground_renderer.stage26.inference.v1",
                "anchor_foreground": "frozen_mask_surrogate_removed",
                "background": "matched_control_anchor_exact_outside_target_mask",
                "foreground": "learned_mask_conditioned_renderer",
                "template_sample_id": item["template_id"],
                "template_split": "train",
                "target_state_label": item["factorization"]["target_state_label"],
                "target_morphology": item["factorization"]["target_morphology"],
                "state_key": item["state_key"],
                "selection_mode": item["factorization"]["selection_mode"],
                "paired_validation_reference_used": False,
                "target_mask_is_hard_support": True,
            }
            output_rows.append(output)
            diagnostics.append(
                {
                    "generated_id": item["generated_id"],
                    "source_name": baseline_row["source_name"],
                    "target_state": item["factorization"]["target_state_label"],
                    "target_morphology": item["factorization"]["target_morphology"],
                    "template_sample_id": item["template_id"],
                    "template_split": "train",
                    "anchor_mask_fraction": item["anchor_mask_fraction"],
                    "target_mask_fraction": float(batch_masks[local_index].mean()),
                    "background_outside_target_sha256": hashlib.sha256(
                        np.ascontiguousarray(
                            batch_backgrounds[local_index][~batch_masks[local_index]]
                        ).tobytes()
                    ).hexdigest(),
                }
            )
        print(f"rendered {end}/{len(payloads)}", flush=True)

    output_rows.sort(key=lambda row: str(row["generated_id"]))
    diagnostics.sort(key=lambda row: str(row["generated_id"]))
    manifest_path = args.output_dir / "generated_manifest.jsonl"
    diagnostic_path = args.output_dir / "rendering_diagnostics.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with diagnostic_path.open("w", encoding="utf-8") as handle:
        for row in diagnostics:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = {
        "schema": "exp4.foreground_renderer.stage26.inference_audit.v1",
        "status": "pass",
        "n_images": len(output_rows),
        "n_conditions": len({row["sample_id"] for row in output_rows}),
        "source_counts": dict(Counter(row["source_name"] for row in output_rows)),
        "all_target_masks_train_only": True,
        "paired_validation_reference_used": False,
        "hard_target_mask_support": True,
        "exact_background_outside_target_mask": True,
        "renderer_checkpoint": str(args.checkpoint.resolve()),
        "renderer_checkpoint_sha256": sha256_file(args.checkpoint),
        "generated_manifest_sha256": sha256_file(manifest_path),
        "diagnostics_sha256": sha256_file(diagnostic_path),
        "test_accessed": False,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("saved", manifest_path)
    print("saved", args.output_dir / "audit.json")


