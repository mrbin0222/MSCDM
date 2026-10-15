#!/usr/bin/env python3
"""Edit generated geometry."""

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
from mscdms.workflows.build_spatial_morphology_oracle import (
    nearest_background_fill,
    predict_masks,
    sha256_file,
)
from mscdms.workflows.generate_foreground_renderer import (
    load_mask,
    load_rgb,
    normalize_array,
    save_tensor_image,
    tensor_image,
)


TARGET_ATTRIBUTES = ("identity", "state")
DNN_LABELS = ("3t3", "hek293", "hela", "pc-3", "raw264", "rpe")
DNN_LABEL_TO_CLASS = {label: index for index, label in enumerate(DNN_LABELS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae_checkpoint", type=Path, required=True)
    parser.add_argument("--baseline_manifest", type=Path, required=True)
    parser.add_argument("--control_edits", type=Path, required=True)
    parser.add_argument("--stage25_manifest", type=Path, required=True)
    parser.add_argument("--training_manifest", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--bbbc021_checkpoint", type=Path, required=True)
    parser.add_argument("--dynamicnuclearnet_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--mask_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--state_feature_timestep_max", type=int)
    parser.add_argument(
        "--state_feature_target_class_gate",
        type=int,
        help=(
            "Restrict an initialized state-feature delta to one target DNN "
            "class. Stage-36c fixes this to RPE class 5."
        ),
    )
    parser.add_argument(
        "--target_attributes",
        nargs="+",
        choices=TARGET_ATTRIBUTES,
        default=list(TARGET_ATTRIBUTES),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    args = parse_args()
    if args.state_feature_target_class_gate is not None and not (
        0 <= args.state_feature_target_class_gate < len(DNN_LABELS)
    ):
        raise ValueError("state-feature target-class gate is outside DNN vocabulary")
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
        raise ValueError("Stage-27b VAE calibration differs from Stage-27a")
    if checkpoint["source_to_id"].get("dynamicnuclearnet") is None:
        raise ValueError("Stage-27a checkpoint has no DynamicNuclearNet source")

    baseline_all = read_jsonl(args.baseline_manifest)
    baseline = [
        row
        for row in baseline_all
        if row.get("source_name") == "dynamicnuclearnet"
        and row.get("target_attribute") in set(args.target_attributes)
    ]
    edits = {str(row["edit_id"]): row for row in read_jsonl(args.control_edits)}
    requested = set(args.target_attributes)
    expected_expert = 16 * sum(
        row.get("source_name") == "dynamicnuclearnet"
        and row.get("target_attribute") in requested
        for row in edits.values()
    )
    if len(baseline) != expected_expert:
        raise ValueError(
            f"expected {expected_expert} requested DNN expert rows, "
            f"got {len(baseline)} from {len(baseline_all)} baseline rows"
        )
    baseline.sort(key=lambda row: str(row["generated_id"]))
    generated_ids = {str(row["generated_id"]) for row in baseline}
    if len(generated_ids) != len(baseline):
        raise ValueError("duplicate DNN Stage-27b generated IDs")
    if not edits:
        raise ValueError("control-edit manifest is empty")
    edit_splits = {str(row.get("split")) for row in edits.values()}
    if not edit_splits <= {"val", "test"} or len(edit_splits) != 1:
        raise ValueError(f"control edits mix unsupported splits: {edit_splits}")
    test_accessed = edit_splits == {"test"}
    identity_selection = {
        str(row["generated_id"]): row
        for row in read_jsonl(args.stage25_manifest)
        if row.get("source_name") == "dynamicnuclearnet"
    }
    if "identity" in args.target_attributes and len(identity_selection) != 800:
        raise ValueError("expected 800 DNN train-mask identity selections")
    training = {
        str(row["sample_id"]): row
        for row in read_jsonl(args.training_manifest)
        if row.get("split") == "train"
    }
    anchor_masks = predict_masks(
        baseline,
        bbbc_checkpoint=args.bbbc021_checkpoint,
        dnn_checkpoint=args.dynamicnuclearnet_checkpoint,
        batch_size=args.mask_batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    payloads: list[dict[str, Any]] = []
    for row, anchor_mask_value in zip(baseline, anchor_masks):
        generated_id = str(row["generated_id"])
        edit_id = str(row["sample_id"])
        edit = edits.get(edit_id)
        if edit is None or edit.get("source_name") != "dynamicnuclearnet":
            raise ValueError(f"missing DNN edit definition: {edit_id}")
        attribute = str(row["target_attribute"])
        anchor_mask = np.asarray(anchor_mask_value, dtype=bool)
        template_id = None
        if attribute == "identity":
            selection = identity_selection.get(generated_id)
            if selection is None:
                raise ValueError(f"missing train-mask selection: {generated_id}")
            factorization = selection["foreground_background_factorization"]
            if factorization.get("template_split") != "train":
                raise ValueError(f"identity template is not train-only: {generated_id}")
            template_id = str(factorization["template_sample_id"])
            template = training.get(template_id)
            if template is None:
                raise ValueError(f"missing train template: {template_id}")
            target_mask = load_mask(
                args.data_dir / str(template["target_mask_path"])
            )
        else:
            target_mask = anchor_mask
        target_state = str(edit["target_labels"]["state"]).lower()
        state_key = f"dynamicnuclearnet::{target_state}"
        if state_key not in checkpoint["state_to_id"]:
            raise ValueError(f"unknown DNN target state: {state_key}")
        anchor_state = str(edit["original_labels"]["state"]).lower()
        anchor_state_key = f"dynamicnuclearnet::{anchor_state}"
        if anchor_state_key not in checkpoint["state_to_id"]:
            raise ValueError(f"unknown DNN anchor state: {anchor_state_key}")
        if target_state not in DNN_LABEL_TO_CLASS or anchor_state not in DNN_LABEL_TO_CLASS:
            raise ValueError("DNN state is outside the frozen six-class vocabulary")
        payloads.append(
            {
                "baseline": row,
                "generated_id": generated_id,
                "attribute": attribute,
                "anchor_mask": anchor_mask,
                "target_mask": np.asarray(target_mask, dtype=bool),
                "target_state": target_state,
                "state_key": state_key,
                "state_id": int(checkpoint["state_to_id"][state_key]),
                "anchor_state": anchor_state,
                "anchor_state_key": anchor_state_key,
                "anchor_state_id": int(
                    checkpoint["state_to_id"][anchor_state_key]
                ),
                "target_state_class_id": DNN_LABEL_TO_CLASS[target_state],
                "anchor_state_class_id": DNN_LABEL_TO_CLASS[anchor_state],
                "source_id": int(checkpoint["source_to_id"]["dynamicnuclearnet"]),
                "template_id": template_id,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(payloads), args.batch_size):
            items = payloads[start : start + args.batch_size]
            backgrounds: list[np.ndarray] = []
            masks: list[np.ndarray] = []
            for item in items:
                anchor = normalize_array(
                    load_rgb(Path(str(item["baseline"]["generated_image"])))
                )
                filled = nearest_background_fill(anchor, item["anchor_mask"]).astype(
                    np.float32
                )
                filled[item["target_mask"]] = 0.0
                backgrounds.append(filled)
                masks.append(item["target_mask"])
            background_rgb = torch.stack(
                [tensor_image(value) for value in backgrounds]
            ).to(device)
            mask = torch.stack(
                [
                    torch.from_numpy(value.astype(np.float32)).unsqueeze(0)
                    for value in masks
                ]
            ).to(device)
            source = torch.tensor(
                [item["source_id"] for item in items], device=device
            )
            state = torch.tensor(
                [item["state_id"] for item in items], device=device
            )
            anchor_state = torch.tensor(
                [item["anchor_state_id"] for item in items], device=device
            )
            target_state_class = torch.tensor(
                [item["target_state_class_id"] for item in items], device=device
            )
            anchor_state_class = torch.tensor(
                [item["anchor_state_class_id"] for item in items], device=device
            )
            with torch.amp.autocast(
                "cuda", enabled=device.type == "cuda", dtype=torch.bfloat16
            ):
                background_mean, _ = vae.encode(background_rgb)
                background_latent = normalize_high_latent(
                    background_mean, calibration
                )
                sampled = model.sample(
                    background_latent,
                    mask,
                    source,
                    state,
                    steps=args.steps,
                    generator=torch.Generator(device=device).manual_seed(
                        args.seed + start
                    ),
                    anchor_state_ids=anchor_state,
                    target_state_class_ids=target_state_class,
                    anchor_state_class_ids=anchor_state_class,
                    state_feature_timestep_max=args.state_feature_timestep_max,
                    state_feature_target_class_gate=(
                        args.state_feature_target_class_gate
                        if model.has_state_feature_delta
                        else None
                    ),
                )
                foreground = vae.decode(
                    denormalize_high_latent(sampled, calibration)
                ).clamp(-1.0, 1.0)
                composite = compose_exact_background(background_rgb, foreground, mask)
            for local, item in enumerate(items):
                output_path = image_dir / f"{item['generated_id']}.png"
                save_tensor_image(composite[local], output_path)
                output = dict(item["baseline"])
                output["generated_image"] = str(output_path.resolve())
                output["checkpoint"] = (
                    f"{args.checkpoint.resolve()}::stage27b_dnn_"
                    f"{item['attribute']}_latent_foreground"
                )
                output["latent_foreground_factorization"] = {
                    "schema": "exp4.latent_foreground_diffusion.stage27b_dnn.v1",
                    "source": "dynamicnuclearnet",
                    "target_attribute": item["attribute"],
                    "target_state": item["target_state"],
                    "state_key": item["state_key"],
                    "anchor_state": item["anchor_state"],
                    "anchor_state_key": item["anchor_state_key"],
                    "state_attention_delta_active": bool(
                        model.has_state_attention_delta
                        and item["state_id"] != item["anchor_state_id"]
                    ),
                    "state_feature_delta_active": bool(
                        model.has_state_feature_delta
                        and item["target_state_class_id"]
                        != item["anchor_state_class_id"]
                    ),
                    "state_feature_timestep_max": args.state_feature_timestep_max,
                    "state_feature_target_class_gate": (
                        args.state_feature_target_class_gate
                        if model.has_state_feature_delta
                        else None
                    ),
                    "target_state_response_active": bool(
                        model.has_target_state_response
                        and item["target_state_class_id"]
                        == model.target_state_response_class
                        and item["target_state_class_id"]
                        != item["anchor_state_class_id"]
                    ),
                    "target_state_response_class": (
                        model.target_state_response_class
                        if model.has_target_state_response
                        else None
                    ),
                    "target_mask": (
                        "train-only morphology template"
                        if item["attribute"] == "identity"
                        else "frozen anchor mask"
                    ),
                    "template_sample_id": item["template_id"],
                    "paired_validation_reference_used": False,
                    "stochastic_foreground": True,
                    "hard_mask_support": True,
                }
                output_rows.append(output)
                diagnostics.append(
                    {
                        "generated_id": item["generated_id"],
                        "target_attribute": item["attribute"],
                        "target_state": item["target_state"],
                        "template_sample_id": item["template_id"],
                        "anchor_mask_fraction": float(item["anchor_mask"].mean()),
                        "target_mask_fraction": float(item["target_mask"].mean()),
                        "background_sha256": stable_array_sha256(backgrounds[local]),
                    }
                )
            print(f"generated DNN counterfactuals {start + len(items)}/{len(payloads)}", flush=True)
    output_rows.sort(key=lambda row: str(row["generated_id"]))
    diagnostics.sort(key=lambda row: str(row["generated_id"]))
    manifest_path = args.output_dir / "generated_manifest.jsonl"
    diagnostics_path = args.output_dir / "diagnostics.jsonl"
    for path, values in ((manifest_path, output_rows), (diagnostics_path, diagnostics)):
        with path.open("w", encoding="utf-8") as handle:
            for row in values:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = {
        "schema": "exp4.latent_foreground_diffusion.stage27b_dnn.audit.v1",
        "status": "pass",
        "n_images": len(output_rows),
        "target_attributes": sorted(set(args.target_attributes)),
        "attribute_counts": dict(Counter(row["target_attribute"] for row in output_rows)),
        "all_identity_masks_train_only": True,
        "all_state_masks_from_frozen_anchor_predictions": True,
        "paired_validation_reference_used": False,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "state_feature_target_class_gate": args.state_feature_target_class_gate,
        "target_state_response_class": model.target_state_response_class,
        "manifest_sha256": sha256_file(manifest_path),
        "test_accessed": test_accessed,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("saved", manifest_path)
    print("saved", args.output_dir / "audit.json")


if __name__ == "__main__":
    main()
