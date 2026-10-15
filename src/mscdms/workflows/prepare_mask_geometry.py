#!/usr/bin/env python3
"""Prepare mask geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[3]
SOURCES = ("bbbc021", "dynamicnuclearnet")
SOURCE_TO_ID = {source: index for index, source in enumerate(SOURCES)}
MORPHOLOGY_RE = re.compile(r"\b(elongated|oval|round) morphology\b", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--source_manifest",
        default="manifest_control_factor_surrogate_v1.jsonl",
    )
    parser.add_argument(
        "--identity_edits",
        default="control_disentangled_v1/val_identity_edits_stage10_v1.jsonl",
    )
    parser.add_argument(
        "--stage13_summary",
        type=Path,
        default=Path(
            "outputs/v4_counterfactual_mask_geometry_stage13_v1/summary.json"
        ),
    )
    parser.add_argument(
        "--output_manifest",
        default="manifest_control_mask_geometry_stage15_v1.jsonl",
    )
    parser.add_argument(
        "--asset_dir",
        default="control_mask_geometry_stage15_v1",
    )
    parser.add_argument(
        "--audit_output",
        type=Path,
        default=Path(
            "configs/evaluation/v4_mask_geometry_stage15_v1/manifest_audit.json"
        ),
    )
    parser.add_argument("--canvas_margin", type=int, default=2)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


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


def classify_ratio(ratio: float) -> str:
    if ratio >= 1.75:
        return "elongated"
    if ratio <= 1.20:
        return "round"
    return "oval"


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(np.asarray(mask, dtype=bool))
    if count == 0:
        raise ValueError("empty target mask")
    support = np.bincount(labels.ravel())
    support[0] = 0
    return labels == int(support.argmax())


def geometry(mask: np.ndarray) -> dict[str, Any]:
    coordinates = np.argwhere(np.asarray(mask, dtype=bool))
    if not len(coordinates):
        raise ValueError("empty geometry mask")
    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0)
    width = int(x1 - x0 + 1)
    height = int(y1 - y0 + 1)
    ratio = float(max(width, height) / max(1, min(width, height)))
    center_y, center_x = coordinates.mean(axis=0)
    return {
        "area": int(len(coordinates)),
        "bbox_xywh": [int(x0), int(y0), width, height],
        "ratio": ratio,
        "morphology": classify_ratio(ratio),
        "orientation": "horizontal" if width >= height else "vertical",
        "centroid_xy": [float(center_x), float(center_y)],
    }


def counterfactual_mask(
    mask: np.ndarray,
    *,
    target_ratio: float,
    canvas_margin: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = largest_component(mask)
    before = geometry(mask)
    x0, y0, width, height = before["bbox_xywh"]
    crop = mask[y0 : y0 + height, x0 : x0 + width].astype(np.uint8)
    bbox_area = float(width * height)
    major = math.sqrt(bbox_area * target_ratio)
    minor = math.sqrt(bbox_area / target_ratio)
    if before["orientation"] == "horizontal":
        new_width, new_height = major, minor
    else:
        new_width, new_height = minor, major
    canvas_height, canvas_width = mask.shape
    fit_scale = min(
        1.0,
        (canvas_width - 2 * canvas_margin) / max(new_width, 1.0),
        (canvas_height - 2 * canvas_margin) / max(new_height, 1.0),
    )
    new_width = max(1, int(round(new_width * fit_scale)))
    new_height = max(1, int(round(new_height * fit_scale)))
    resized = np.asarray(
        Image.fromarray(crop * 255, mode="L").resize(
            (new_width, new_height),
            Image.Resampling.NEAREST,
        )
    ) > 0
    coordinates = np.argwhere(resized)
    resized_y, resized_x = coordinates.mean(axis=0)
    center_x, center_y = before["centroid_xy"]
    requested_paste_x = int(round(center_x - resized_x))
    requested_paste_y = int(round(center_y - resized_y))
    min_paste_x = canvas_margin
    min_paste_y = canvas_margin
    max_paste_x = canvas_width - canvas_margin - new_width
    max_paste_y = canvas_height - canvas_margin - new_height
    if max_paste_x < min_paste_x or max_paste_y < min_paste_y:
        raise ValueError("counterfactual morphology mask cannot fit in canvas")
    # Some valid source cells sit close to an image boundary. Preserve the
    # transformed mask exactly and translate it by the minimum distance needed
    # to fit the canvas instead of clipping its geometry.
    paste_x = min(max(requested_paste_x, min_paste_x), max_paste_x)
    paste_y = min(max(requested_paste_y, min_paste_y), max_paste_y)
    placement_shift_pixels = math.dist(
        (requested_paste_x, requested_paste_y),
        (paste_x, paste_y),
    )
    output = np.zeros_like(mask, dtype=bool)
    output[paste_y : paste_y + new_height, paste_x : paste_x + new_width] = resized
    after = geometry(output)
    return output, {
        "area_relative_change": float(
            (after["area"] - before["area"]) / before["area"]
        ),
        "centroid_shift_pixels": float(
            math.dist(before["centroid_xy"], after["centroid_xy"])
        ),
        "target_ratio": float(target_ratio),
        "fit_scale": float(fit_scale),
        "boundary_adjusted": bool(placement_shift_pixels > 0.0),
        "placement_shift_pixels": float(placement_shift_pixels),
    }


def save_target(
    *,
    data_dir: Path,
    row: dict[str, Any],
    target_morphology: str,
    target_ratio: float,
    relative_output: Path,
    canvas_margin: int,
) -> dict[str, Any]:
    mask_path = data_dir / str(row["target_mask_path"])
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert("L")) > 0
    transformed, diagnostics = counterfactual_mask(
        mask,
        target_ratio=target_ratio,
        canvas_margin=canvas_margin,
    )
    after = geometry(transformed)
    if after["morphology"] != target_morphology:
        raise ValueError(
            f"{row['sample_id']}: target morphology was not achieved"
        )
    output_path = data_dir / relative_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(transformed.astype(np.uint8) * 255, mode="L").save(
        output_path
    )
    return {
        "target_mask_path": relative_output.as_posix(),
        "target_morphology": target_morphology,
        "after": after,
        "diagnostics": diagnostics,
    }


def main() -> None:
    args = parse_args()
    data_dir = resolve(args.data_dir)
    source_manifest = data_dir / args.source_manifest
    identity_edits_path = data_dir / args.identity_edits
    stage13_path = resolve(args.stage13_summary)
    output_manifest = data_dir / args.output_manifest
    asset_root = Path(args.asset_dir)
    audit_path = resolve(args.audit_output)

    rows = read_jsonl(source_manifest)
    if len(rows) != 27500:
        raise ValueError(f"expected 27,500 train/val rows, got {len(rows)}")
    stage13 = json.loads(stage13_path.read_text(encoding="utf-8"))
    if not bool(stage13.get("gate", {}).get("passed")):
        raise ValueError("Stage-13 geometry gate did not pass")
    target_ratios = stage13["target_ratios"]

    output_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    row_by_id = {str(row["sample_id"]): row for row in rows}
    for index, row in enumerate(rows, 1):
        normalized = dict(row)
        enabled = str(row["source_name"]) in SOURCES
        rank_enabled = dict(row["control_rank_enabled"])
        rank_enabled.update(subject=enabled, state=False, background=False)
        normalized["control_rank_enabled"] = rank_enabled
        if enabled:
            source = str(row["source_name"])
            target_morphology = parse_morphology(
                row["control_negatives"]["subject"]
            )
            anchor_morphology = str(
                row["attributes"]["morphology"]
            ).lower()
            if target_morphology == anchor_morphology:
                raise ValueError(
                    f"{row['sample_id']}: morphology target did not change"
                )
            relative_output = (
                asset_root
                / "training_targets"
                / str(row["split"])
                / source
                / f"{row['sample_id']}.png"
            )
            saved = save_target(
                data_dir=data_dir,
                row=row,
                target_morphology=target_morphology,
                target_ratio=float(target_ratios[source][target_morphology]),
                relative_output=relative_output,
                canvas_margin=args.canvas_margin,
            )
            normalized["morphology_mask_surrogate"] = {
                "enabled": True,
                "source_id": SOURCE_TO_ID[source],
                **saved,
            }
            diagnostics.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "split": str(row["split"]),
                    "source_name": source,
                    **saved,
                }
            )
        else:
            normalized["morphology_mask_surrogate"] = {
                "enabled": False,
                "source_id": -1,
                "target_mask_path": "",
            }
        output_rows.append(normalized)
        if index % 1000 == 0 or index == len(rows):
            print(f"training geometry {index}/{len(rows)}", flush=True)

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    evaluation_targets: list[dict[str, Any]] = []
    edits = read_jsonl(identity_edits_path)
    if len(edits) != 100:
        raise ValueError(f"expected 100 identity edits, got {len(edits)}")
    for edit in edits:
        source = str(edit["source_name"])
        anchor_id = str(edit["anchor_sample_id"])
        row = row_by_id[anchor_id]
        target_morphology = parse_morphology(edit["target_labels"]["identity"])
        anchor_morphology = str(row["attributes"]["morphology"]).lower()
        if target_morphology == anchor_morphology:
            raise ValueError(f"{edit['edit_id']}: evaluation target did not change")
        relative_output = (
            asset_root
            / "evaluation_targets"
            / source
            / f"{edit['edit_id']}.png"
        )
        saved = save_target(
            data_dir=data_dir,
            row=row,
            target_morphology=target_morphology,
            target_ratio=float(target_ratios[source][target_morphology]),
            relative_output=relative_output,
            canvas_margin=args.canvas_margin,
        )
        evaluation_targets.append(
            {
                "edit_id": str(edit["edit_id"]),
                "anchor_sample_id": anchor_id,
                "source_name": source,
                "source_id": SOURCE_TO_ID[source],
                **saved,
            }
        )
    evaluation_path = data_dir / asset_root / "val_identity_geometry_targets.jsonl"
    with evaluation_path.open("w", encoding="utf-8") as handle:
        for row in evaluation_targets:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    split_source_counts = Counter(
        (row["split"], row["source_name"]) for row in diagnostics
    )
    area_errors = [
        abs(float(row["diagnostics"]["area_relative_change"]))
        for row in diagnostics
    ]
    center_shifts = [
        float(row["diagnostics"]["centroid_shift_pixels"])
        for row in diagnostics
    ]
    placement_shifts = [
        float(row["diagnostics"]["placement_shift_pixels"])
        for row in diagnostics
    ]
    boundary_adjusted_count = sum(
        bool(row["diagnostics"]["boundary_adjusted"])
        for row in diagnostics
    )
    audit = {
        "schema": "exp4.mask_geometry.stage15.manifest_audit.v1",
        "status": "pass",
        "output_manifest": str(output_manifest),
        "output_manifest_sha256": sha256_file(output_manifest),
        "evaluation_targets": str(evaluation_path),
        "evaluation_targets_sha256": sha256_file(evaluation_path),
        "counts": {
            "all_train_val": len(output_rows),
            "geometry_train_val": len(diagnostics),
            "evaluation_edits": len(evaluation_targets),
            "by_split_source": {
                f"{split}|{source}": count
                for (split, source), count in sorted(split_source_counts.items())
            },
        },
        "geometry": {
            "median_absolute_area_change": float(np.median(area_errors)),
            "p95_absolute_area_change": float(np.quantile(area_errors, 0.95)),
            "max_centroid_shift_pixels": float(max(center_shifts)),
            "boundary_adjusted_count": int(boundary_adjusted_count),
            "boundary_adjusted_fraction": float(
                boundary_adjusted_count / len(diagnostics)
            ),
            "p95_placement_shift_pixels": float(
                np.quantile(placement_shifts, 0.95)
            ),
            "max_placement_shift_pixels": float(max(placement_shifts)),
            "all_target_morphologies_achieved": True,
        },
        "invariants": {
            "only_bbbc021_and_dynamicnuclearnet_enabled": True,
            "only_subject_factor_enabled": True,
            "state_and_background_prompts_preserved": True,
            "train_and_validation_only": True,
            "test_accessed": False,
        },
        "inputs": {
            "source_manifest_sha256": sha256_file(source_manifest),
            "identity_edits_sha256": sha256_file(identity_edits_path),
            "stage13_summary_sha256": sha256_file(stage13_path),
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved {output_manifest}")
    print(f"saved {evaluation_path}")
    print(f"saved {audit_path}")


