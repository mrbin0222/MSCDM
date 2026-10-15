#!/usr/bin/env python3
"""Support foreground exemplar."""

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
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mscdms.workflows.build_spatial_morphology_oracle import predict_masks


def classify_ratio(ratio: float) -> str:
    if ratio >= 1.75:
        return "elongated"
    if ratio <= 1.20:
        return "round"
    return "oval"


def largest_component(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    labels, components = ndimage.label(binary)
    if components == 0:
        raise ValueError("mask is empty")
    support = np.bincount(labels.ravel())
    support[0] = 0
    return labels == int(support.argmax())


def mask_geometry(mask: np.ndarray) -> dict[str, Any]:
    binary = np.asarray(mask, dtype=bool)
    coordinates = np.argwhere(binary)
    if not len(coordinates):
        raise ValueError("mask is empty")
    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0)
    height = int(y1 - y0 + 1)
    width = int(x1 - x0 + 1)
    ratio = float(max(width, height) / max(1, min(width, height)))
    return {
        "area": int(binary.sum()),
        "bbox_xywh": [int(x0), int(y0), width, height],
        "bbox_aspect_ratio": ratio,
        "morphology": classify_ratio(ratio),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle_manifest", type=Path, required=True)
    parser.add_argument("--training_manifest", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--bbbc021_checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dynamicnuclearnet_checkpoint", type=Path, required=True
    )
    parser.add_argument("--raw_conditions", type=Path, required=True)
    parser.add_argument("--composite_conditions", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(
            image.convert("L").resize((512, 512), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
    return largest_component(array > 0)


def dice(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    intersection = float((left & right).sum())
    return (2.0 * intersection + 1.0) / (left.sum() + right.sum() + 1.0)


def condition_majority(rows: list[dict[str, Any]], key: str) -> float:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["edit_id"])].append(row)
    successes = []
    for values in groups.values():
        votes = Counter(str(row[key]) for row in values)
        predicted = votes.most_common(1)[0][0]
        successes.append(predicted == str(values[0]["target_morphology"]))
    return float(np.mean(np.asarray(successes, dtype=np.float64)))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_images": len(rows),
        "n_conditions": len({row["edit_id"] for row in rows}),
        "exact_mask_target_success": float(
            np.mean([row["exact_mask_target_success"] for row in rows])
        ),
        "raw_surrogate_target_success": float(
            np.mean([row["raw_surrogate_target_success"] for row in rows])
        ),
        "composite_surrogate_target_success": float(
            np.mean([row["composite_surrogate_target_success"] for row in rows])
        ),
        "raw_surrogate_condition_success": condition_majority(
            rows, "raw_predicted_morphology"
        ),
        "composite_surrogate_condition_success": condition_majority(
            rows, "composite_predicted_morphology"
        ),
        "mean_raw_surrogate_dice_to_exact": float(
            np.mean([row["raw_surrogate_dice_to_exact"] for row in rows])
        ),
        "mean_composite_surrogate_dice_to_exact": float(
            np.mean([row["composite_surrogate_dice_to_exact"] for row in rows])
        ),
        "raw_to_composite_morphology_agreement": float(
            np.mean(
                [
                    row["raw_predicted_morphology"]
                    == row["composite_predicted_morphology"]
                    for row in rows
                ]
            )
        ),
    }


def summarize_formal_conditions(path: Path) -> dict[str, Any]:
    rows = [
        row
        for row in read_jsonl(path)
        if row.get("target_attribute") == "identity"
    ]
    if len(rows) != 100:
        raise ValueError(f"expected 100 formal identity conditions: {path}")

    def block(values: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n_conditions": len(values),
            "target_success": float(
                np.mean([bool(row["target_success"]) for row in values])
            ),
            "preservation": float(
                np.mean([bool(row["preservation_success"]) for row in values])
            ),
            "acsr": float(np.mean([bool(row["acsr_success"]) for row in values])),
        }

    return {
        "overall": block(rows),
        "by_source": {
            source: block([row for row in rows if row.get("source") == source])
            for source in ("bbbc021", "dynamicnuclearnet")
        },
        "by_target_morphology": {
            morphology: block(
                [
                    row
                    for row in rows
                    if str(row.get("target_labels", {}).get("identity", "")).endswith(
                        f"|{morphology}"
                    )
                ]
            )
            for morphology in ("elongated", "oval", "round")
        },
    }


def diagnose(
    overall: dict[str, Any],
    raw_formal: dict[str, Any] | None = None,
    composite_formal: dict[str, Any] | None = None,
) -> tuple[str, str]:
    exact = float(overall["exact_mask_target_success"])
    raw = float(overall["raw_surrogate_target_success"])
    composite = float(overall["composite_surrogate_target_success"])
    if exact < 0.95:
        return (
            "template_label_geometry_mismatch",
            "rebuild the train-exemplar selection and morphology label protocol",
        )
    if raw < 0.80:
        return (
            "mask_surrogate_not_valid_for_real_morphology_classification",
            "remove surrogate morphology success from the architecture gate and retain it only for overlap diagnostics",
        )
    if raw_formal is not None and composite_formal is not None:
        raw_target = float(raw_formal["overall"]["target_success"])
        composite_target = float(composite_formal["overall"]["target_success"])
        if composite_target < raw_target - 0.15:
            return (
                "hard_composition_domain_shift",
                "replace hard exemplar pasting with a learned mask-conditioned foreground generator and keep exact background bypass",
            )
    if composite < raw - 0.15:
        return (
            "hard_composition_domain_shift",
            "replace hard exemplar pasting with a learned mask-conditioned foreground generator and keep exact background bypass",
        )
    return (
        "geometry_assumption_supported",
        "proceed to the learned mask-conditioned foreground generator",
    )


def main() -> None:
    args = parse_args()
    oracle_path = resolve(args.oracle_manifest)
    training_path = resolve(args.training_manifest)
    data_dir = resolve(args.data_dir)
    output_dir = resolve(args.output_dir)
    oracle_rows = read_jsonl(oracle_path)
    if len(oracle_rows) != 1600:
        raise ValueError(f"expected 1600 oracle rows, got {len(oracle_rows)}")
    train_by_id = {
        str(row["sample_id"]): row
        for row in read_jsonl(training_path)
        if row.get("split") == "train"
    }
    raw_rows: list[dict[str, Any]] = []
    exact_masks: list[np.ndarray] = []
    for row in oracle_rows:
        payload = row.get("foreground_background_factorization")
        if not isinstance(payload, dict) or payload.get("template_split") != "train":
            raise ValueError(f"invalid oracle row: {row.get('generated_id')}")
        template_id = str(payload["template_sample_id"])
        train = train_by_id.get(template_id)
        if train is None:
            raise ValueError(f"missing train template: {template_id}")
        image_path = data_dir / str(train["image_path"])
        mask_path = data_dir / str(train["target_mask_path"])
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"missing train assets: {template_id}")
        raw_rows.append(
            {
                "generated_image": str(image_path),
                "source_name": str(row["source_name"]),
            }
        )
        exact_masks.append(read_mask(mask_path))

    device = torch.device(args.device)
    kwargs = {
        "bbbc_checkpoint": resolve(args.bbbc021_checkpoint),
        "dnn_checkpoint": resolve(args.dynamicnuclearnet_checkpoint),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": device,
    }
    raw_predictions = predict_masks(raw_rows, **kwargs)
    composite_predictions = predict_masks(oracle_rows, **kwargs)

    per_image: list[dict[str, Any]] = []
    for row, exact, raw_pred, composite_pred in zip(
        oracle_rows,
        exact_masks,
        raw_predictions,
        composite_predictions,
    ):
        payload = row["foreground_background_factorization"]
        target = str(payload["target_morphology"])
        exact_morphology = str(mask_geometry(exact)["morphology"])
        raw_morphology = str(mask_geometry(largest_component(raw_pred))["morphology"])
        composite_morphology = str(
            mask_geometry(largest_component(composite_pred))["morphology"]
        )
        per_image.append(
            {
                "generated_id": str(row["generated_id"]),
                "edit_id": str(row["sample_id"]),
                "source_name": str(row["source_name"]),
                "target_morphology": target,
                "template_sample_id": str(payload["template_sample_id"]),
                "selection_mode": str(payload["selection_mode"]),
                "exact_mask_morphology": exact_morphology,
                "raw_predicted_morphology": raw_morphology,
                "composite_predicted_morphology": composite_morphology,
                "exact_mask_target_success": exact_morphology == target,
                "raw_surrogate_target_success": raw_morphology == target,
                "composite_surrogate_target_success": composite_morphology == target,
                "raw_surrogate_dice_to_exact": dice(raw_pred, exact),
                "composite_surrogate_dice_to_exact": dice(composite_pred, exact),
            }
        )

    overall = summarize(per_image)
    by_source = {
        source: summarize([row for row in per_image if row["source_name"] == source])
        for source in ("bbbc021", "dynamicnuclearnet")
    }
    by_target = {
        morphology: summarize(
            [row for row in per_image if row["target_morphology"] == morphology]
        )
        for morphology in ("elongated", "oval", "round")
    }
    raw_formal = summarize_formal_conditions(resolve(args.raw_conditions))
    composite_formal = summarize_formal_conditions(
        resolve(args.composite_conditions)
    )
    formal_delta = {
        key: float(composite_formal["overall"][key])
        - float(raw_formal["overall"][key])
        for key in ("target_success", "preservation", "acsr")
    }
    finding, next_action = diagnose(overall, raw_formal, composite_formal)
    summary = {
        "schema": "exp4.foreground_exemplar_oracle.stage25b.audit.v1",
        "status": "pass",
        "overall": overall,
        "by_source": by_source,
        "by_target_morphology": by_target,
        "formal_control": {
            "raw_train_exemplars": raw_formal,
            "composites": composite_formal,
            "composite_minus_raw_overall": formal_delta,
        },
        "diagnosis": {
            "finding": finding,
            "next_action": next_action,
        },
        "all_templates_train_only": True,
        "paired_validation_reference_used": False,
        "test_accessed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_image.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_image:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Stage-25b foreground-exemplar geometry audit",
        "",
        "| Measure | Value |",
        "|---|---:|",
    ]
    for key, value in overall.items():
        if key.startswith("n_"):
            continue
        lines.append(f"| {key} | {value:.4f} |")
    lines.extend(
        [
            "",
            "## Formal control: raw exemplar versus composite",
            "",
            "| Measure | Raw train exemplar | Composite | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in ("target_success", "preservation", "acsr"):
        lines.append(
            f"| {key} | {raw_formal['overall'][key]:.4f} | "
            f"{composite_formal['overall'][key]:.4f} | "
            f"{formal_delta[key]:+.4f} |"
        )
    lines.extend(
        [
            "",
            f"- Finding: `{finding}`",
            f"- Next action: `{next_action}`",
            "- Test accessed: no",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("saved", output_dir / "summary.json")
    print("DIAGNOSIS:", finding)


