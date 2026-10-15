#!/usr/bin/env python3
"""Train foreground evaluator."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mscdms.workflows.support_dnn_3t3_hek293_factor_boundary import (
    BoundaryDataset,
    extract_dino,
    normalize_state,
)
from mscdms.workflows.support_dnn_factor_isolated_state import (
    FrozenProbe,
    GeneratedFactorDataset,
    evaluate_run,
    parse_runs,
)
from mscdms.workflows.support_latent_factor_probe_capacity import train_probe
from mscdms.control_evaluator import read_jsonl, sha256_file


SOURCE = "dynamicnuclearnet"
STATES = ("3t3", "hek293", "hela", "pc-3", "raw264", "rpe")
VIEWS = ("full", "foreground_only", "background_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", type=Path, default=Path("models/dinov2-large"))
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--evaluator_manifest",
        type=Path,
        default=Path(
            "data/processed/"
            "control_metrics_v2/evaluator_manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--mask_manifest",
        type=Path,
        default=Path(
            "data/processed/"
            "mask_manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--control_edits",
        type=Path,
        default=Path(
            "data/processed/"
            "control_disentangled_v1/val_control_edits.jsonl"
        ),
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="LABEL=existing run directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/v4_dnn_sixclass_factor_isolated_stage38a_v1"),
    )
    parser.add_argument("--extract_batch_size", type=int, default=64)
    parser.add_argument("--probe_batch_size", type=int, default=256)
    parser.add_argument("--score_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--population_size", type=int, default=16)
    parser.add_argument("--population_reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def select_rows(
    evaluator_rows: list[dict[str, Any]],
    mask_rows: list[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    mask_by_id = {
        str(row["sample_id"]): row
        for row in mask_rows
        if row.get("source_name") == SOURCE and row.get("split") == split
    }
    selected: list[dict[str, Any]] = []
    for row in evaluator_rows:
        state = normalize_state(row.get("state_label"))
        if (
            row.get("source_name") != SOURCE
            or row.get("split") != split
            or state not in STATES
        ):
            continue
        sample_id = str(row["sample_id"])
        mask_row = mask_by_id.get(sample_id)
        if mask_row is None:
            raise ValueError(f"missing target mask row: {sample_id}")
        merged = dict(row)
        # BoundaryDataset keeps this historical field name.  Here it stores
        # the six-class DNN state ID rather than a binary ID.
        merged["binary_state_id"] = STATES.index(state)
        merged["target_mask_path"] = mask_row.get("target_mask_path")
        selected.append(merged)
    selected.sort(key=lambda row: str(row["sample_id"]))
    expected = {"train": 5000, "val": 500}[split]
    if len(selected) != expected:
        raise ValueError(f"{split}: expected {expected} DNN rows, got {len(selected)}")
    counts = Counter(int(row["binary_state_id"]) for row in selected)
    if set(counts) != set(range(len(STATES))) or min(counts.values()) < 1:
        raise ValueError(f"{split}: incomplete six-state coverage: {counts}")
    return selected


def add_recalls(metrics: dict[str, Any]) -> dict[str, Any]:
    output = dict(metrics)
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    support = confusion.sum(axis=1)
    recalls = np.divide(
        np.diag(confusion),
        support,
        out=np.zeros(len(STATES), dtype=np.float64),
        where=support > 0,
    )
    output["per_class_recall"] = {
        label: float(recalls[index]) for index, label in enumerate(STATES)
    }
    return output


def decide(
    real_results: dict[str, Any], generated_results: dict[str, Any]
) -> tuple[str, dict[str, bool]]:
    real_foreground = real_results["foreground_only"]["selection"]
    generated = generated_results["stage36c"]["foreground_target"]
    real_recalls = real_foreground["per_class_recall"]
    generated_recalls = generated["per_class_recall"]
    checks = {
        "real_foreground_macro_f1": float(real_foreground["macro_f1"]) >= 0.90,
        "real_foreground_min_recall": min(real_recalls.values()) >= 0.75,
        "generated_target_accuracy": float(generated["accuracy"]) >= 0.80,
        "generated_target_macro_f1": float(generated["macro_f1"]) >= 0.65,
        "generated_3t3_recall": generated_recalls["3t3"] >= 0.75,
        "generated_hek293_recall": generated_recalls["hek293"] >= 0.75,
        "generated_min_recall": min(generated_recalls.values()) >= 0.50,
    }
    evaluator_ready = checks["real_foreground_macro_f1"] and checks[
        "real_foreground_min_recall"
    ]
    generator_ready = all(
        value
        for key, value in checks.items()
        if key.startswith("generated_")
    )
    if not evaluator_ready:
        decision = "stop_formal_dnn_sca_factor_evaluator_not_reliable"
    elif generator_ready:
        decision = "freeze_stage36c_with_sixclass_foreground_isolated_dnn_sca"
    else:
        decision = "retain_stage36c_quality_and_localize_remaining_foreground_state_failures"
    return decision, checks


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    model_dir = resolve(args.model_dir)
    data_dir = resolve(args.data_dir)
    evaluator_path = resolve(args.evaluator_manifest)
    mask_path = resolve(args.mask_manifest)
    edits_path = resolve(args.control_edits)
    output_dir = resolve(args.output_dir)
    cache_dir = output_dir / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    runs = parse_runs(args.run)
    for path in (
        model_dir / "model.safetensors",
        evaluator_path,
        mask_path,
        edits_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    evaluator_rows = read_jsonl(evaluator_path)
    mask_rows = read_jsonl(mask_path)
    rows = {
        split: select_rows(evaluator_rows, mask_rows, split)
        for split in ("train", "val")
    }
    model_hash = sha256_file(model_dir / "model.safetensors")
    device = torch.device(args.device)

    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(
        str(model_dir), local_files_only=True
    ).to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    for view in VIEWS:
        for split in ("train", "val"):
            cache_path = cache_dir / f"{view}_{split}.pt"
            expected_ids = [str(row["sample_id"]) for row in rows[split]]
            if cache_path.is_file():
                cached = torch.load(
                    cache_path, map_location="cpu", weights_only=False
                )
                if (
                    cached.get("model_sha256") == model_hash
                    and cached.get("sample_ids") == expected_ids
                ):
                    print("reuse", cache_path, flush=True)
                    continue
            loader = DataLoader(
                BoundaryDataset(rows[split], data_dir, view),
                batch_size=args.extract_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            cache = extract_dino(
                backbone,
                loader,
                device=device,
                precision=args.precision,
                view=view,
                split=split,
                model_hash=model_hash,
            )
            cache["schema"] = "exp4.dnn_sixclass.factor_view_cache.stage38a.v1"
            torch.save(cache, cache_path)

    real_results: dict[str, Any] = {}
    for view in VIEWS:
        train_cache = torch.load(
            cache_dir / f"{view}_train.pt", map_location="cpu", weights_only=False
        )
        val_cache = torch.load(
            cache_dir / f"{view}_val.pt", map_location="cpu", weights_only=False
        )
        result = train_probe(
            task_name=f"dnn6_{view}",
            branch=view,
            train_features=train_cache["features"].float(),
            train_labels=train_cache["labels"].long(),
            val_features=val_cache["features"].float(),
            val_labels=val_cache["labels"].long(),
            num_classes=len(STATES),
            args=args,
            device=device,
            output_dir=output_dir,
            population=True,
        )
        result["single_image"] = add_recalls(result["single_image"])
        result["selection"] = add_recalls(result["selection"])
        real_results[view] = result

    foreground_probe_path = (
        output_dir / "probes/dnn6_foreground_only/checkpoint_best.pt"
    )
    background_probe_path = (
        output_dir / "probes/dnn6_background_only/checkpoint_best.pt"
    )
    foreground_probe = FrozenProbe(foreground_probe_path).to(device).eval()
    background_probe = FrozenProbe(background_probe_path).to(device).eval()
    training_masks = {
        str(row["sample_id"]): str(row["target_mask_path"])
        for row in mask_rows
        if row.get("source_name") == SOURCE and row.get("split") == "train"
    }
    edits = {str(row["edit_id"]): row for row in read_jsonl(edits_path)}
    generated_results: dict[str, Any] = {}
    for label, run_dir in runs.items():
        dataset = GeneratedFactorDataset(
            run_dir,
            data_dir=data_dir,
            training_masks=training_masks,
            edits=edits,
            states=STATES,
        )
        result = evaluate_run(
            label=label,
            dataset=dataset,
            backbone=backbone,
            foreground_probe=foreground_probe,
            background_probe=background_probe,
            device=device,
            batch_size=args.score_batch_size,
            num_workers=args.num_workers,
            precision=args.precision,
            states=STATES,
        )
        generated_results[label] = result
        with (output_dir / f"{label}_conditions.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in result["conditions"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if "stage36c" not in generated_results:
        raise ValueError("a run labelled stage36c is required")
    decision, checks = decide(real_results, generated_results)
    status = "pass" if decision.startswith("freeze_stage36c") else "fail"
    summary = {
        "schema": "exp4.dnn.sixclass_factor_isolated.stage38a.v1",
        "status": status,
        "decision": decision,
        "states": list(STATES),
        "real_factor_evaluators": real_results,
        "generated_runs": generated_results,
        "checks": checks,
        "provenance": {
            "source": SOURCE,
            "fit_split": "real train only",
            "selection_split": "real validation only",
            "train_rows": len(rows["train"]),
            "validation_rows": len(rows["val"]),
            "population_size": args.population_size,
            "generator_trained": False,
            "images_regenerated": False,
            "dino_model_sha256": model_hash,
            "foreground_probe_sha256": sha256_file(foreground_probe_path),
            "background_probe_sha256": sha256_file(background_probe_path),
            "test_accessed": False,
        },
        "test_accessed": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Stage-38a six-state factor-isolated DNN audit",
        "",
        f"- Status: **{status.upper()}**",
        f"- Decision: `{decision}`",
        "- Generator trained: no",
        "- Images regenerated: no",
        "- Test accessed: no",
        "",
        "## Real validation evaluator gate",
        "",
        "| View | Single Acc/F1 | Population-16 Acc/F1 | Min class recall |",
        "|---|---:|---:|---:|",
    ]
    for view in VIEWS:
        single = real_results[view]["single_image"]
        population = real_results[view]["selection"]
        lines.append(
            f"| {view} | {single['accuracy']:.4f}/{single['macro_f1']:.4f} | "
            f"{population['accuracy']:.4f}/{population['macro_f1']:.4f} | "
            f"{min(population['per_class_recall'].values()):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Existing generated runs",
            "",
            "| Run | Foreground target Acc/F1 | 3T3 | HEK293 | Min recall | Background anchor agreement |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, result in generated_results.items():
        foreground = result["foreground_target"]
        recalls = foreground["per_class_recall"]
        agreement = result["background_anchor"]["agreement"]
        agreement_text = "N/A" if agreement is None else f"{agreement:.4f}"
        lines.append(
            f"| {label} | {foreground['accuracy']:.4f}/{foreground['macro_f1']:.4f} | "
            f"{recalls['3t3']:.4f} | {recalls['hek293']:.4f} | "
            f"{min(recalls.values()):.4f} | {agreement_text} |"
        )
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("saved", summary_path)
    print("saved", report_path)
    print("STAGE-38A DECISION:", decision)
    print("TEST DATA ACCESSED: NO")


if __name__ == "__main__":
    main()
