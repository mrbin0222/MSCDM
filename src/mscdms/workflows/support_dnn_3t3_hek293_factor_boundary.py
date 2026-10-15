#!/usr/bin/env python3
"""Support dnn 3t3 hek293 factor boundary."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mscdms.workflows.support_decoded_dinov2_surrogate_capacity import (
    DINO_MEAN,
    DINO_STD,
    dino_descriptor,
)
from mscdms.workflows.support_dnn_mask_state_mediation import geometry_features
from mscdms.workflows.support_latent_factor_probe_capacity import train_probe
from mscdms.control_evaluator import read_jsonl, resolve_image_path, sha256_file


SOURCE = "dynamicnuclearnet"
TARGET_STATES = ("3t3", "hek293")
RGB_VIEWS = ("full", "foreground_only", "background_only")


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
        "--output_dir",
        type=Path,
        default=Path("outputs/v4_dnn_3t3_hek293_factor_boundary_stage37a_v1"),
    )
    parser.add_argument("--extract_batch_size", type=int, default=64)
    parser.add_argument("--probe_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--population_size", type=int, default=16)
    parser.add_argument("--population_reps", type=int, default=1000)
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def normalize_state(value: Any) -> str:
    label = str(value or "").strip().lower()
    suffix = " cell-line state"
    return label[: -len(suffix)].strip() if label.endswith(suffix) else label


def fill_value(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    background = image[~mask]
    if not len(background):
        return np.zeros((image.shape[-1],), dtype=np.uint8)
    return np.median(background, axis=0).astype(np.uint8)


def nearest_background_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove foreground using nearest observed background without a flat mask hole."""
    indices = ndimage.distance_transform_edt(
        mask,
        return_distances=False,
        return_indices=True,
    )
    output = image.copy()
    output[mask] = image[indices[0][mask], indices[1][mask]]
    return output


def construct_view(image: Image.Image, mask: Image.Image, view: str) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    binary = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    if binary.shape != rgb.shape[:2]:
        binary = np.asarray(
            mask.convert("L").resize(
                (rgb.shape[1], rgb.shape[0]),
                resample=Image.Resampling.NEAREST,
            ),
            dtype=np.uint8,
        ) > 0
    if not bool(binary.any()) or bool(binary.all()):
        raise ValueError("target mask must be non-empty and non-full")
    if view == "full":
        output = rgb
    elif view == "foreground_only":
        output = np.broadcast_to(fill_value(rgb, binary), rgb.shape).copy()
        output[binary] = rgb[binary]
    elif view == "background_only":
        output = nearest_background_inpaint(rgb, binary)
    else:
        raise ValueError(f"unsupported RGB view: {view}")
    return Image.fromarray(output, mode="RGB")


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
            or state not in TARGET_STATES
        ):
            continue
        sample_id = str(row["sample_id"])
        mask_row = mask_by_id.get(sample_id)
        if mask_row is None:
            raise ValueError(f"missing target mask row: {sample_id}")
        merged = dict(row)
        merged["binary_state_id"] = TARGET_STATES.index(state)
        merged["target_mask_path"] = mask_row.get("target_mask_path")
        selected.append(merged)
    selected.sort(key=lambda row: str(row["sample_id"]))
    expected = {"train": 1667, "val": 167}[split]
    if len(selected) != expected:
        raise ValueError(f"{split}: expected {expected} binary rows, got {len(selected)}")
    counts = Counter(int(row["binary_state_id"]) for row in selected)
    if not all(counts[index] > 0 for index in range(2)):
        raise ValueError(f"{split}: incomplete class coverage: {counts}")
    return selected


class BoundaryDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(self, rows: list[dict[str, Any]], data_dir: Path, view: str) -> None:
        self.rows = rows
        self.data_dir = data_dir
        self.view = view
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    256,
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(DINO_MEAN, DINO_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.rows[index]
        image_path = resolve_image_path(str(row["image_path"]), self.data_dir)
        mask_path = self.data_dir / str(row["target_mask_path"])
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            view = construct_view(image, mask, self.view)
            pixels = self.transform(view)
        return (
            pixels,
            torch.tensor(int(row["binary_state_id"]), dtype=torch.long),
            str(row["sample_id"]),
        )


@torch.inference_mode()
def extract_dino(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
    view: str,
    split: str,
    model_hash: str,
) -> dict[str, Any]:
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    sample_ids: list[str] = []
    amp_enabled = device.type == "cuda" and precision != "fp32"
    for batch_index, (pixels, target, batch_ids) in enumerate(loader, start=1):
        pixels = pixels.to(device, non_blocking=True)
        with torch.amp.autocast(
            "cuda", enabled=amp_enabled, dtype=torch.bfloat16
        ):
            output = model(pixel_values=pixels)
        features.append(dino_descriptor(output.last_hidden_state).cpu().half())
        labels.append(target)
        sample_ids.extend(str(value) for value in batch_ids)
        if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(loader):
            print(
                f"extract {view}/{split}: "
                f"{min(batch_index * loader.batch_size, len(loader.dataset))}/"
                f"{len(loader.dataset)}",
                flush=True,
            )
    return {
        "schema": "exp4.dnn_3t3_hek293.factor_view_cache.stage37a.v1",
        "view": view,
        "split": split,
        "model_sha256": model_hash,
        "features": torch.cat(features),
        "labels": torch.cat(labels),
        "sample_ids": sample_ids,
        "test_accessed": False,
    }


def extract_geometry(rows: list[dict[str, Any]], data_dir: Path, grid_size: int) -> dict[str, Any]:
    features: list[torch.Tensor] = []
    labels: list[int] = []
    ids: list[str] = []
    for index, row in enumerate(rows, start=1):
        mask_path = data_dir / str(row["target_mask_path"])
        with Image.open(mask_path) as mask:
            binary = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
        features.append(geometry_features(binary, grid_size=grid_size))
        labels.append(int(row["binary_state_id"]))
        ids.append(str(row["sample_id"]))
        if index % 500 == 0:
            print(f"extract mask_geometry: {index}/{len(rows)}", flush=True)
    return {
        "schema": "exp4.dnn_3t3_hek293.factor_view_cache.stage37a.v1",
        "view": "mask_geometry",
        "features": torch.stack(features).half(),
        "labels": torch.tensor(labels, dtype=torch.long),
        "sample_ids": ids,
        "test_accessed": False,
    }


def decision(results: dict[str, Any]) -> tuple[str, dict[str, bool]]:
    population = {
        view: float(result["selection"]["macro_f1"])
        for view, result in results.items()
    }
    checks = {
        "full_real_boundary": population["full"] >= 0.90,
        "foreground_boundary": population["foreground_only"] >= 0.80,
        "geometry_boundary": population["mask_geometry"] >= 0.70,
        "background_leakage": population["background_only"] >= 0.80,
    }
    if not checks["full_real_boundary"]:
        label = "stop_3t3_control_real_boundary_not_supported"
    elif checks["foreground_boundary"] and checks["geometry_boundary"]:
        label = "build_sparse_3t3_joint_geometry_appearance_teacher"
    elif checks["foreground_boundary"]:
        label = "build_sparse_3t3_foreground_appearance_teacher"
    elif checks["background_leakage"]:
        label = "stop_background_invariant_3t3_control_state_is_background_confounded"
    else:
        label = "stop_3t3_control_no_stable_observable_factor_view"
    return label, checks


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    model_dir = resolve(args.model_dir)
    data_dir = resolve(args.data_dir)
    evaluator_path = resolve(args.evaluator_manifest)
    mask_path = resolve(args.mask_manifest)
    output_dir = resolve(args.output_dir)
    cache_dir = output_dir / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        if not (model_dir / name).is_file():
            raise FileNotFoundError(model_dir / name)
    evaluator_rows = read_jsonl(evaluator_path)
    mask_rows = read_jsonl(mask_path)
    rows = {
        split: select_rows(evaluator_rows, mask_rows, split)
        for split in ("train", "val")
    }
    model_hash = sha256_file(model_dir / "model.safetensors")

    from transformers import AutoModel

    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for view in RGB_VIEWS:
        for split in ("train", "val"):
            path = cache_dir / f"{view}_{split}.pt"
            if path.is_file():
                cached = torch.load(path, map_location="cpu", weights_only=False)
                if (
                    cached.get("model_sha256") == model_hash
                    and cached.get("sample_ids")
                    == [str(row["sample_id"]) for row in rows[split]]
                ):
                    print("reuse", path, flush=True)
                    continue
            loader = DataLoader(
                BoundaryDataset(rows[split], data_dir, view),
                batch_size=args.extract_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            cache = extract_dino(
                model,
                loader,
                device=device,
                precision=args.precision,
                view=view,
                split=split,
                model_hash=model_hash,
            )
            torch.save(cache, path)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for split in ("train", "val"):
        path = cache_dir / f"mask_geometry_{split}.pt"
        expected_ids = [str(row["sample_id"]) for row in rows[split]]
        if path.is_file():
            cached = torch.load(path, map_location="cpu", weights_only=False)
            if cached.get("sample_ids") == expected_ids:
                print("reuse", path, flush=True)
                continue
        torch.save(extract_geometry(rows[split], data_dir, args.grid_size), path)

    results: dict[str, Any] = {}
    for view in (*RGB_VIEWS, "mask_geometry"):
        train_cache = torch.load(
            cache_dir / f"{view}_train.pt", map_location="cpu", weights_only=False
        )
        val_cache = torch.load(
            cache_dir / f"{view}_val.pt", map_location="cpu", weights_only=False
        )
        results[view] = train_probe(
            task_name=f"3t3_vs_hek293_{view}",
            branch=view,
            train_features=train_cache["features"].float(),
            train_labels=train_cache["labels"].long(),
            val_features=val_cache["features"].float(),
            val_labels=val_cache["labels"].long(),
            num_classes=2,
            args=args,
            device=device,
            output_dir=output_dir,
            population=True,
        )

    action, checks = decision(results)
    summary = {
        "schema": "exp4.dnn_3t3_hek293.factor_boundary.stage37a.v1",
        "status": "pass" if checks["full_real_boundary"] else "fail",
        "decision": action,
        "states": list(TARGET_STATES),
        "results": results,
        "diagnostic_checks": checks,
        "provenance": {
            "source": SOURCE,
            "fit_split": "real train only",
            "selection_split": "real validation only",
            "train_rows": len(rows["train"]),
            "validation_rows": len(rows["val"]),
            "generator_trained": False,
            "test_accessed": False,
            "dino_model_sha256": model_hash,
            "evaluator_manifest_sha256": sha256_file(evaluator_path),
            "mask_manifest_sha256": sha256_file(mask_path),
        },
        "test_accessed": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Stage-37a real 3T3/HEK293 factor-boundary audit",
        "",
        f"- Status: **{summary['status'].upper()}**",
        f"- Decision: `{action}`",
        "- Generator trained: no",
        "- Test accessed: no",
        "",
        "| View | Single Acc | Single F1 | Population-16 Acc | Population-16 F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for view in (*RGB_VIEWS, "mask_geometry"):
        single = results[view]["single_image"]
        population = results[view]["selection"]
        lines.append(
            f"| {view} | {single['accuracy']:.4f} | {single['macro_f1']:.4f} | "
            f"{population['accuracy']:.4f} | {population['macro_f1']:.4f} |"
        )
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("saved", summary_path)
    print("saved", report_path)
    print("STAGE-37A DECISION:", action)
    print("TEST DATA ACCESSED: NO")


