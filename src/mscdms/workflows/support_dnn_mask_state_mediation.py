#!/usr/bin/env python3
"""Support dnn mask state mediation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mscdms.workflows.support_latent_factor_probe_capacity import (
    FeatureProbe,
    population_metrics,
    predict,
)
from mscdms.control_evaluator import classification_metrics, read_jsonl, sha256_file
from mscdms.workflows.prepare_mask_geometry import largest_component


SOURCE = "dynamicnuclearnet"
STATES = ("3t3", "hek293", "hela", "pc-3", "raw264", "rpe")
MORPHOLOGIES = ("round", "oval", "elongated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--training_manifest",
        type=Path,
        default=Path(
            "data/processed/"
            "manifest_control_mask_geometry_stage15_v1.jsonl"
        ),
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
        "--output_dir",
        type=Path,
        default=Path("outputs/v4_dnn_mask_state_mediation_stage31a_v1"),
    )
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--population_size", type=int, default=16)
    parser.add_argument("--population_reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def normalize_state(value: Any) -> str:
    label = str(value or "").strip().lower()
    suffix = " cell-line state"
    if label.endswith(suffix):
        label = label[: -len(suffix)].strip()
    return label


def geometry_features(mask: np.ndarray, grid_size: int = 32) -> torch.Tensor:
    mask = largest_component(np.asarray(mask, dtype=bool))
    if not bool(mask.any()) or bool(mask.all()):
        raise ValueError("mask must be non-empty and non-full")
    height, width = mask.shape
    y, x = np.nonzero(mask)
    y0, y1 = int(y.min()), int(y.max() + 1)
    x0, x1 = int(x.min()), int(x.max() + 1)
    bbox_h = y1 - y0
    bbox_w = x1 - x0
    area = float(mask.sum())
    centered_x = x.astype(np.float64) - float(x.mean())
    centered_y = y.astype(np.float64) - float(y.mean())
    covariance = np.cov(np.stack([centered_x, centered_y]), bias=True)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    major = float(max(eigenvalues[0], 1e-8))
    minor = float(max(eigenvalues[1], 0.0))
    eccentricity = math.sqrt(max(0.0, 1.0 - minor / major))
    boundary = mask ^ ndimage.binary_erosion(mask)
    perimeter = float(boundary.sum())
    circularity = 4.0 * math.pi * area / max(perimeter * perimeter, 1.0)
    radial = np.sqrt(centered_x * centered_x + centered_y * centered_y)
    radius_scale = max(height, width)
    scalars = np.asarray(
        [
            area / (height * width),
            bbox_h / height,
            bbox_w / width,
            math.log(max(bbox_w / max(bbox_h, 1), 1e-6)),
            area / max(bbox_h * bbox_w, 1),
            float(x.mean()) / width,
            float(y.mean()) / height,
            major / (radius_scale * radius_scale),
            minor / (radius_scale * radius_scale),
            eccentricity,
            perimeter / (height + width),
            circularity,
            float(np.quantile(radial, 0.25)) / radius_scale,
            float(np.quantile(radial, 0.50)) / radius_scale,
            float(np.quantile(radial, 0.75)) / radius_scale,
            float(radial.std()) / radius_scale,
        ],
        dtype=np.float32,
    )
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    grid = F.interpolate(
        tensor,
        size=(grid_size, grid_size),
        mode="area",
    ).flatten()
    result = torch.cat([grid, torch.from_numpy(scalars)])
    if not torch.isfinite(result).all():
        raise ValueError("mask geometry features contain non-finite values")
    return result


def load_mask_feature(path: Path, grid_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
    return geometry_features(mask, grid_size)


def extract_features(
    *,
    training_rows: list[dict[str, Any]],
    evaluator_rows: list[dict[str, Any]],
    data_dir: Path,
    grid_size: int,
) -> dict[str, dict[str, Any]]:
    evaluator_by_id = {
        str(row["sample_id"]): row
        for row in evaluator_rows
        if row.get("source_name") == SOURCE
        and row.get("split") in {"train", "val"}
    }
    state_to_id = {label: index for index, label in enumerate(STATES)}
    morphology_to_id = {
        label: index for index, label in enumerate(MORPHOLOGIES)
    }
    outputs: dict[str, dict[str, Any]] = {}
    for split, expected in (("train", 5000), ("val", 500)):
        selected = [
            row
            for row in training_rows
            if row.get("source_name") == SOURCE and row.get("split") == split
        ]
        selected.sort(key=lambda row: str(row["sample_id"]))
        if len(selected) != expected:
            raise ValueError(f"{split}: expected {expected} DNN rows, got {len(selected)}")
        features: list[torch.Tensor] = []
        states: list[int] = []
        morphologies: list[int] = []
        sample_ids: list[str] = []
        for index, row in enumerate(selected, 1):
            sample_id = str(row["sample_id"])
            evaluator = evaluator_by_id.get(sample_id)
            if evaluator is None:
                raise ValueError(f"missing evaluator row: {sample_id}")
            state = normalize_state(evaluator.get("state_label"))
            morphology = str(row.get("attributes", {}).get("morphology") or "").lower()
            if state not in state_to_id or morphology not in morphology_to_id:
                raise ValueError(
                    f"invalid DNN labels: {sample_id}/{state}/{morphology}"
                )
            mask_path = data_dir / str(row.get("target_mask_path") or "")
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)
            features.append(load_mask_feature(mask_path, grid_size))
            states.append(state_to_id[state])
            morphologies.append(morphology_to_id[morphology])
            sample_ids.append(sample_id)
            if index % 1000 == 0:
                print(f"mask features {split}: {index}/{expected}", flush=True)
        outputs[split] = {
            "features": torch.stack(features),
            "state": torch.tensor(states, dtype=torch.long),
            "morphology": torch.tensor(morphologies, dtype=torch.long),
            "sample_ids": sample_ids,
        }
    return outputs


def evaluate_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    morphologies: torch.Tensor,
    *,
    population_size: int,
    population_reps: int,
    seed: int,
) -> dict[str, Any]:
    single = classification_metrics(
        labels.tolist(), logits.argmax(dim=1).tolist(), num_classes=len(STATES)
    )
    overall = population_metrics(
        logits,
        labels,
        num_classes=len(STATES),
        population_size=population_size,
        reps=population_reps,
        seed=seed,
    )
    by_morphology: dict[str, Any] = {}
    for morphology_id, morphology in enumerate(MORPHOLOGIES):
        selected = morphologies == morphology_id
        counts = torch.bincount(labels[selected], minlength=len(STATES))
        if bool((counts == 0).any()):
            by_morphology[morphology] = {
                "eligible": False,
                "class_counts": counts.tolist(),
                "reason": "at least one state class is absent",
            }
            continue
        metrics = population_metrics(
            logits[selected],
            labels[selected],
            num_classes=len(STATES),
            population_size=population_size,
            reps=population_reps,
            seed=seed + 100 + morphology_id,
        )
        by_morphology[morphology] = {
            "eligible": True,
            "class_counts": counts.tolist(),
            "population16": metrics,
        }
    return {
        "single_image": single,
        "population16": overall,
        "population16_by_morphology": by_morphology,
    }


def main() -> None:
    args = parse_args()
    data_dir = resolve(args.data_dir)
    training_path = resolve(args.training_manifest)
    evaluator_path = resolve(args.evaluator_manifest)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "mask_features.pt"
    if cache_path.is_file():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache.get("schema") != "exp4.dnn_mask_state_mediation.features.v1":
            raise ValueError("unsupported Stage-31a feature cache")
        splits = cache["splits"]
        print("reusing", cache_path)
    else:
        splits = extract_features(
            training_rows=read_jsonl(training_path),
            evaluator_rows=read_jsonl(evaluator_path),
            data_dir=data_dir,
            grid_size=args.grid_size,
        )
        torch.save(
            {
                "schema": "exp4.dnn_mask_state_mediation.features.v1",
                "grid_size": args.grid_size,
                "splits": splits,
                "training_manifest_sha256": sha256_file(training_path),
                "evaluator_manifest_sha256": sha256_file(evaluator_path),
                "test_accessed": False,
            },
            cache_path,
        )
        print("saved", cache_path)

    train_x = splits["train"]["features"].float()
    train_y = splits["train"]["state"].long()
    val_x = splits["val"]["features"].float()
    val_y = splits["val"]["state"].long()
    val_morphology = splits["val"]["morphology"].long()
    if train_x.shape[0] != 5000 or val_x.shape[0] != 500:
        raise ValueError("Stage-31a cache has invalid split counts")
    feature_mean = train_x.mean(dim=0)
    feature_std = train_x.std(dim=0).clamp_min(1e-4)
    train_x = (train_x - feature_mean) / feature_std
    val_x = (val_x - feature_mean) / feature_std

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = FeatureProbe(
        train_x.shape[1], args.hidden_dim, len(STATES), args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    best_score = -1.0
    best_payload: dict[str, Any] | None = None
    patience = args.patience
    history: list[dict[str, Any]] = []
    checkpoint_path = output_dir / "checkpoint_best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for feature_batch, label_batch in loader:
            feature_batch = feature_batch.to(device)
            label_batch = label_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(feature_batch), label_batch)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(label_batch)
            count += len(label_batch)
        model.eval()
        logits = predict(model, val_x, batch_size=args.batch_size, device=device)
        metrics = evaluate_logits(
            logits,
            val_y,
            val_morphology,
            population_size=args.population_size,
            population_reps=args.population_reps,
            seed=args.seed,
        )
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / count,
            "metrics": metrics,
        }
        history.append(record)
        population = metrics["population16"]
        print(
            f"epoch={epoch} loss={record['train_loss']:.6f} "
            f"single={metrics['single_image']['accuracy']:.4f}/"
            f"{metrics['single_image']['macro_f1']:.4f} "
            f"population16={population['accuracy']:.4f}/"
            f"{population['macro_f1']:.4f}",
            flush=True,
        )
        score = float(population["macro_f1"])
        if score > best_score:
            best_score = score
            patience = args.patience
            best_payload = {
                "schema": "exp4.dnn_mask_state_mediation.probe.v1",
                "model_config": model.config,
                "model": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "states": list(STATES),
                "morphologies": list(MORPHOLOGIES),
                "best_epoch": epoch,
                "selection_metric": "validation_population16_macro_f1",
                "metrics": metrics,
                "test_accessed": False,
            }
            torch.save(best_payload, checkpoint_path)
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_payload is None:
        raise RuntimeError("Stage-31a saved no checkpoint")
    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    result = best_payload["metrics"]
    overall = result["population16"]
    round_metrics = result["population16_by_morphology"]["round"]
    oval_metrics = result["population16_by_morphology"]["oval"]
    thresholds = {
        "overall_accuracy_min": 0.80,
        "overall_macro_f1_min": 0.75,
        "within_round_accuracy_min": 0.65,
        "within_round_macro_f1_min": 0.60,
        "within_oval_accuracy_min": 0.65,
        "within_oval_macro_f1_min": 0.60,
    }
    checks = {
        "overall_accuracy": overall["accuracy"]
        >= thresholds["overall_accuracy_min"],
        "overall_macro_f1": overall["macro_f1"]
        >= thresholds["overall_macro_f1_min"],
        "within_round_accuracy": round_metrics["eligible"]
        and round_metrics["population16"]["accuracy"]
        >= thresholds["within_round_accuracy_min"],
        "within_round_macro_f1": round_metrics["eligible"]
        and round_metrics["population16"]["macro_f1"]
        >= thresholds["within_round_macro_f1_min"],
        "within_oval_accuracy": oval_metrics["eligible"]
        and oval_metrics["population16"]["accuracy"]
        >= thresholds["within_oval_accuracy_min"],
        "within_oval_macro_f1": oval_metrics["eligible"]
        and oval_metrics["population16"]["macro_f1"]
        >= thresholds["within_oval_macro_f1_min"],
    }
    passed = all(bool(value) for value in checks.values())
    summary = {
        "schema": "exp4.dnn_mask_state_mediation.stage31a.audit.v1",
        "status": "pass" if passed else "fail",
        "decision": (
            "make_geometry_a_joint_subject_state_mediator"
            if passed
            else "do_not_attribute_dnn_state_failure_to_geometry_alone"
        ),
        "best_epoch": int(best_payload["best_epoch"]),
        "results": result,
        "gate": {
            "thresholds": thresholds,
            "checks": checks,
            "passed": passed,
        },
        "provenance": {
            "input": "binary real train/validation target masks only",
            "rgb_opened": False,
            "prompts_used_as_features": False,
            "background_used": False,
            "train_rows": 5000,
            "validation_rows": 500,
            "checkpoint_selection": "validation population16 macro-F1",
            "hyperparameter_scan": False,
            "test_accessed": False,
        },
        "inputs": {
            "training_manifest_sha256": sha256_file(training_path),
            "evaluator_manifest_sha256": sha256_file(evaluator_path),
            "feature_cache_sha256": sha256_file(cache_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
        "test_accessed": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Stage-31a DNN mask-only state mediation audit",
        "",
        f"- Status: **{summary['status'].upper()}**",
        f"- Decision: `{summary['decision']}`",
        "- Input: real binary target masks only",
        "- RGB/background/prompt features: none",
        "- Test accessed: no",
        "",
        "| Stratum | Accuracy | Macro-F1 |",
        "|---|---:|---:|",
        f"| Single image | {result['single_image']['accuracy']:.4f} | "
        f"{result['single_image']['macro_f1']:.4f} |",
        f"| Population-16 overall | {overall['accuracy']:.4f} | "
        f"{overall['macro_f1']:.4f} |",
    ]
    for morphology in ("round", "oval", "elongated"):
        row = result["population16_by_morphology"][morphology]
        if row["eligible"]:
            metric = row["population16"]
            lines.append(
                f"| Population-16 within {morphology} | "
                f"{metric['accuracy']:.4f} | {metric['macro_f1']:.4f} |"
            )
        else:
            lines.append(f"| Population-16 within {morphology} | N/A | N/A |")
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("saved", summary_path)
    print("saved", report_path)
    print("STAGE-31A GATE:", summary["status"].upper())
    print("TEST DATA ACCESSED: NO")


