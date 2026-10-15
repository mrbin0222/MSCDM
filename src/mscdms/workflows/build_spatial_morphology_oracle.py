#!/usr/bin/env python3
"""Build spatial morphology oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mscdms.mask_surrogate import FrozenSourceMaskSurrogates
from mscdms.workflows.prepare_mask_geometry import (
    counterfactual_mask,
    geometry,
    largest_component,
)


MORPHOLOGY_RE = re.compile(r"\b(elongated|oval|round) morphology\b", re.I)
SOURCE_TO_ID = {"bbbc021": 0, "dynamicnuclearnet": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor_manifest", type=Path, required=True)
    parser.add_argument("--identity_edits", type=Path, required=True)
    parser.add_argument("--stage13_summary", type=Path, required=True)
    parser.add_argument("--bbbc021_checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dynamicnuclearnet_checkpoint", type=Path, required=True
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--canvas_margin", type=int, default=8)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_morphology(value: str) -> str:
    match = MORPHOLOGY_RE.search(str(value))
    if match is None:
        raise ValueError(f"cannot parse morphology: {value!r}")
    return match.group(1).lower()


def normalize_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize(
            (512, 512), Image.Resampling.BICUBIC
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    flattened = tensor.view(3, -1)
    low = torch.quantile(flattened, 0.01, dim=1).view(3, 1, 1)
    high = torch.quantile(flattened, 0.99, dim=1).view(3, 1, 1)
    high = torch.maximum(high, low + 1e-6)
    return ((tensor - low) / (high - low)).clamp(0.0, 1.0) * 2.0 - 1.0


class AnchorDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.rows[index]
        path = Path(row["generated_image"])
        if not path.is_file():
            raise FileNotFoundError(path)
        source = str(row["source_name"])
        return (
            normalize_image(path),
            torch.tensor(SOURCE_TO_ID[source], dtype=torch.long),
            index,
        )


def nearest_background_fill(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    expanded = ndimage.binary_dilation(mask, iterations=3)
    if bool(expanded.all()):
        raise ValueError("foreground mask occupies the full canvas")
    indices = ndimage.distance_transform_edt(
        expanded,
        return_distances=False,
        return_indices=True,
    )
    return image[indices[0], indices[1]]


def warp_foreground(
    image: np.ndarray,
    source_mask: np.ndarray,
    *,
    target_ratio: float,
    canvas_margin: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Warp one foreground component while filling its old support."""
    image = np.asarray(image, dtype=np.uint8)
    source_mask = largest_component(source_mask)
    target_mask, transform = counterfactual_mask(
        source_mask,
        target_ratio=target_ratio,
        canvas_margin=canvas_margin,
    )
    before = geometry(source_mask)
    after = geometry(target_mask)
    sx, sy, sw, sh = before["bbox_xywh"]
    tx, ty, tw, th = after["bbox_xywh"]

    source_patch = Image.fromarray(image[sy : sy + sh, sx : sx + sw])
    source_patch = source_patch.resize((tw, th), Image.Resampling.BICUBIC)
    patch = np.asarray(source_patch, dtype=np.float32)

    target_alpha = (
        target_mask[ty : ty + th, tx : tx + tw].astype(np.uint8) * 255
    )
    alpha = Image.fromarray(target_alpha, mode="L").filter(
        ImageFilter.GaussianBlur(radius=0.75)
    )
    alpha_array = np.asarray(alpha, dtype=np.float32)[..., None] / 255.0

    background = nearest_background_fill(image, source_mask).astype(np.float32)
    region = background[ty : ty + th, tx : tx + tw]
    background[ty : ty + th, tx : tx + tw] = (
        alpha_array * patch + (1.0 - alpha_array) * region
    )
    output = np.clip(np.round(background), 0, 255).astype(np.uint8)
    diagnostics = {
        "source_geometry": before,
        "target_geometry": after,
        "transform": transform,
        "foreground_area_fraction": float(source_mask.mean()),
        "reference_target_mask_accessed": False,
    }
    return output, target_mask, diagnostics


def predict_masks(
    rows: list[dict[str, Any]],
    *,
    bbbc_checkpoint: Path,
    dnn_checkpoint: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> list[np.ndarray]:
    bundle = FrozenSourceMaskSurrogates(
        bbbc021_checkpoint=bbbc_checkpoint,
        dynamicnuclearnet_checkpoint=dnn_checkpoint,
    ).to(device)
    loader = DataLoader(
        AnchorDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    predictions: list[np.ndarray | None] = [None] * len(rows)
    for images, source_ids, indices in loader:
        images = images.to(device, non_blocking=True)
        source_ids = source_ids.to(device, non_blocking=True)
        with torch.inference_mode(), torch.amp.autocast(
            "cuda",
            enabled=device.type == "cuda",
            dtype=torch.bfloat16,
        ):
            masks = bundle(images, source_ids).sigmoid() >= 0.5
        masks = torch.nn.functional.interpolate(
            masks.float(),
            size=(512, 512),
            mode="nearest",
        ).bool()
        for local_index, global_index in enumerate(indices.tolist()):
            predictions[global_index] = masks[local_index, 0].cpu().numpy()
    if any(mask is None for mask in predictions):
        raise RuntimeError("mask inference did not cover every anchor image")
    return [np.asarray(mask, dtype=bool) for mask in predictions]


def main() -> None:
    args = parse_args()
    anchor_path = resolve(args.anchor_manifest)
    edits_path = resolve(args.identity_edits)
    stage13_path = resolve(args.stage13_summary)
    output_dir = resolve(args.output_dir)
    image_dir = output_dir / "images"
    all_anchor_rows = read_jsonl(anchor_path)
    if len(all_anchor_rows) != 4800:
        raise ValueError(
            f"expected 4,800 schedule-preserving anchor images, got "
            f"{len(all_anchor_rows)}"
        )
    anchor_rows = [
        row for row in all_anchor_rows if row.get("target_attribute") == "identity"
    ]
    edits = read_jsonl(edits_path)
    if len(anchor_rows) != 1600:
        raise ValueError(f"expected 1,600 anchor images, got {len(anchor_rows)}")
    if len(edits) != 100:
        raise ValueError(f"expected 100 identity edits, got {len(edits)}")
    edit_by_id = {str(row["edit_id"]): row for row in edits}
    if set(str(row["sample_id"]) for row in anchor_rows) != set(edit_by_id):
        raise ValueError("anchor generation and identity edits are not aligned")
    if any(row.get("target_attribute") != "identity" for row in anchor_rows):
        raise ValueError("invalid identity anchor target attributes")

    stage13 = json.loads(stage13_path.read_text(encoding="utf-8"))
    if not stage13.get("gate", {}).get("passed"):
        raise ValueError("Stage-13 geometry audit did not pass")
    target_ratios = stage13["target_ratios"]

    device = torch.device(args.device)
    masks = predict_masks(
        anchor_rows,
        bbbc_checkpoint=resolve(args.bbbc021_checkpoint),
        dnn_checkpoint=resolve(args.dynamicnuclearnet_checkpoint),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    image_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for row, source_mask in zip(anchor_rows, masks):
        edit = edit_by_id[str(row["sample_id"])]
        source = str(row["source_name"])
        target_morphology = parse_morphology(edit["subject"])
        target_ratio = float(target_ratios[source][target_morphology])
        generated_path = Path(row["generated_image"])
        with Image.open(generated_path) as image:
            image_array = np.asarray(
                image.convert("RGB").resize(
                    (512, 512), Image.Resampling.BICUBIC
                ),
                dtype=np.uint8,
            )
        output, _, item_diagnostics = warp_foreground(
            image_array,
            source_mask,
            target_ratio=target_ratio,
            canvas_margin=args.canvas_margin,
        )
        output_path = image_dir / f"{row['generated_id']}.png"
        Image.fromarray(output, mode="RGB").save(output_path)

        output_row = dict(row)
        output_row.update(
            {
                "generated_image": str(output_path),
                "subject": str(edit["subject"]),
                "conditioning_subject": str(edit["subject"]),
                "state": str(edit["state"]),
                "background": str(edit["background"]),
                "target_attribute": "identity",
                "anchor_prompt": {
                    "subject": str(row["subject"]),
                    "state": str(row["state"]),
                    "background": str(row["background"]),
                },
                "checkpoint": (
                    f"{row['checkpoint']}::stage17_spatial_morphology_oracle"
                ),
                "spatial_intervention": {
                    "schema": "exp4.spatial_morphology.oracle.stage17.v1",
                    "target_morphology": target_morphology,
                    "target_ratio": target_ratio,
                    "source_mask": "frozen_source_surrogate_prediction",
                    "paired_target_mask_used": False,
                },
            }
        )
        output_rows.append(output_row)
        diagnostics.append(
            {
                "generated_id": str(row["generated_id"]),
                "edit_id": str(row["sample_id"]),
                "source_name": source,
                "target_morphology": target_morphology,
                **item_diagnostics,
            }
        )

    manifest_path = output_dir / "generated_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    diagnostics_path = output_dir / "spatial_intervention_diagnostics.jsonl"
    with diagnostics_path.open("w", encoding="utf-8") as handle:
        for row in diagnostics:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = {
        "schema": "exp4.spatial_morphology.oracle.stage17.audit.v1",
        "status": "pass",
        "n_images": len(output_rows),
        "n_schedule_preserving_generated_images": len(all_anchor_rows),
        "n_conditions": len({row["sample_id"] for row in output_rows}),
        "source_counts": dict(Counter(row["source_name"] for row in output_rows)),
        "target_morphology_counts": dict(
            Counter(
                row["spatial_intervention"]["target_morphology"]
                for row in output_rows
            )
        ),
        "paired_target_masks_accessed": False,
        "reference_images_used_for_intervention": False,
        "generator_trained": False,
        "test_accessed": False,
        "inputs": {
            "anchor_manifest_sha256": sha256_file(anchor_path),
            "identity_edits_sha256": sha256_file(edits_path),
            "stage13_summary_sha256": sha256_file(stage13_path),
        },
        "outputs": {"generated_manifest_sha256": sha256_file(manifest_path)},
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved {manifest_path}")
    print(f"saved {output_dir / 'audit.json'}")


