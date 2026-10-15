from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tlc_mplconfig"))

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mscdms.workflows.parent_data_utils import RelationPolicy


SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class TargetSpec:
    name: str
    source_dataset: str
    source_root_default: str
    npz_names: dict[str, str]
    x_channel: int
    x_channels: tuple[int, ...]
    y_channel: int
    source_channel: str
    target_object: str
    category: str
    image_modality: str
    global_template: str


TARGETS: dict[str, TargetSpec] = {
    "dynamic_nuclear": TargetSpec(
        name="dynamicnuclearnet_nuclear_512",
        source_dataset="DynamicNuclearNet-segmentation-v1_0",
        source_root_default="DynamicNuclearNet-segmentation-v1_0",
        npz_names={"train": "train.npz", "val": "val.npz", "test": "test.npz"},
        x_channel=0,
        x_channels=(0,),
        y_channel=0,
        source_channel="nuclear_fluorescence",
        target_object="nucleus",
        category="nucleus",
        image_modality="single-channel nuclear fluorescence stored as grayscale RGB",
        global_template=(
            "single-channel nuclear fluorescence microscopy image from DynamicNuclearNet; "
            "target objects: nuclear instances; specimen: {specimen}; pixel size: {pixel_size}; "
            "time step: {time_step}"
        ),
    ),
    "tissuenet_nuclear": TargetSpec(
        name="tissuenet_nuclear_512",
        source_dataset="tissuenet_v1.1",
        source_root_default="tissuenet_v1.1",
        npz_names={
            "train": "tissuenet_v1.1_train.npz",
            "val": "tissuenet_v1.1_val.npz",
            "test": "tissuenet_v1.1_test.npz",
        },
        x_channel=0,
        x_channels=(0,),
        y_channel=1,
        source_channel="nuclear",
        target_object="nucleus",
        category="nucleus",
        image_modality="single-channel TissueNet nuclear image stored as grayscale RGB",
        global_template=(
            "single-channel TissueNet nuclear microscopy image; target objects: nuclear instances; "
            "specimen: {specimen}; pixel size: {pixel_size}; experiment: {experiment}"
        ),
    ),
    "tissuenet_wholecell": TargetSpec(
        name="tissuenet_wholecell_512",
        source_dataset="tissuenet_v1.1",
        source_root_default="tissuenet_v1.1",
        npz_names={
            "train": "tissuenet_v1.1_train.npz",
            "val": "tissuenet_v1.1_val.npz",
            "test": "tissuenet_v1.1_test.npz",
        },
        x_channel=1,
        x_channels=(0, 1),
        y_channel=0,
        source_channel="nuclear+whole_cell",
        target_object="whole_cell",
        category="whole_cell",
        image_modality="two-channel TissueNet nuclear/whole-cell composite stored as RGB",
        global_template=(
            "two-channel TissueNet nuclear and whole-cell microscopy image; target objects: whole-cell instances; "
            "specimen: {specimen}; pixel size: {pixel_size}; experiment: {experiment}"
        ),
    ),
}


def progress(message: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(f"[prepare_deepcell] {message}", flush=True)


def should_report(count: int, every: int) -> bool:
    return count == 1 or (every > 0 and count % every == 0)


def clean_text(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    try:
        if isinstance(value, float) and not np.isfinite(value):
            return default
    except Exception:
        pass
    text = " ".join(str(value).strip().split())
    if not text or text.lower() == "nan":
        return default
    return text


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    return str(value)


def percentile_uint8(arr: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros(x.shape, dtype=np.uint8)
    vals = x[finite]
    lo, hi = np.percentile(vals, [p_low, p_high])
    if hi <= lo:
        lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        return np.zeros(x.shape, dtype=np.uint8)
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8)


def grayscale_rgb(channel: np.ndarray, image_size: int, p_low: float, p_high: float) -> np.ndarray:
    gray = percentile_uint8(channel, p_low, p_high)
    image = Image.fromarray(gray, mode="L")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    gray512 = np.asarray(image, dtype=np.uint8)
    return np.repeat(gray512[..., None], 3, axis=-1)


def sample_to_rgb(sample: np.ndarray, spec: TargetSpec, image_size: int, p_low: float, p_high: float) -> np.ndarray:
    if len(spec.x_channels) == 1:
        return grayscale_rgb(sample[..., spec.x_channels[0]], image_size, p_low, p_high)
    if spec.source_dataset == "tissuenet_v1.1" and tuple(spec.x_channels) == (0, 1):
        nuclear = Image.fromarray(percentile_uint8(sample[..., 0], p_low, p_high), mode="L")
        whole_cell = Image.fromarray(percentile_uint8(sample[..., 1], p_low, p_high), mode="L")
        if nuclear.size != (image_size, image_size):
            nuclear = nuclear.resize((image_size, image_size), Image.Resampling.BICUBIC)
            whole_cell = whole_cell.resize((image_size, image_size), Image.Resampling.BICUBIC)
        blue = np.asarray(nuclear, dtype=np.uint8)
        green = np.asarray(whole_cell, dtype=np.uint8)
        red = np.zeros_like(green)
        return np.stack([red, green, blue], axis=-1)
    raise ValueError(f"Unsupported channel composition for {spec.name}: {spec.x_channels}")


def resize_label_map(mask: np.ndarray, image_size: int) -> np.ndarray:
    arr = np.asarray(mask)
    image = Image.fromarray(arr.astype(np.int32), mode="I")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.int32)


def relabel_instance_map(mask: np.ndarray, min_area: int, max_area: int) -> tuple[np.ndarray, list[int]]:
    arr = np.asarray(mask, dtype=np.int64)
    if arr.size == 0 or int(arr.max(initial=0)) <= 0:
        return np.zeros(arr.shape, dtype=np.uint16), []
    counts = np.bincount(arr.ravel())
    old_ids = np.flatnonzero(counts)
    old_ids = old_ids[old_ids > 0]
    old_ids = old_ids[(counts[old_ids] >= min_area) & (counts[old_ids] <= max_area)]
    if old_ids.size == 0:
        return np.zeros(arr.shape, dtype=np.uint16), []
    if old_ids.size > np.iinfo(np.uint16).max:
        raise ValueError(f"Too many instances for uint16 map: {old_ids.size}")
    lut = np.zeros(counts.shape[0], dtype=np.uint16)
    lut[old_ids] = np.arange(1, old_ids.size + 1, dtype=np.uint16)
    relabeled = lut[arr]
    return relabeled, [int(x) for x in old_ids.tolist()]


def area_level(area: int, areas: np.ndarray) -> str:
    if areas.size == 0:
        return "medium"
    q1, q2 = np.percentile(areas.astype(np.float32), [33.0, 66.0])
    if area <= q1:
        return "small"
    if area >= q2:
        return "large"
    return "medium"


def intensity_level(value: float, values: np.ndarray) -> str:
    if values.size == 0:
        return "medium"
    q1, q2 = np.percentile(values.astype(np.float32), [33.0, 66.0])
    if value <= q1:
        return "low"
    if value >= q2:
        return "high"
    return "medium"


def morphology_from_bbox(width: int, height: int) -> str:
    short = max(1.0, float(min(width, height)))
    ratio = float(max(width, height)) / short
    if ratio >= 1.75:
        return "elongated"
    if ratio <= 1.2:
        return "round"
    return "oval"


def instance_stats(instance_map: np.ndarray, gray: np.ndarray) -> list[dict[str, Any]]:
    labels = np.asarray(instance_map, dtype=np.int64)
    n = int(labels.max(initial=0))
    if n <= 0:
        return []
    flat_labels = labels.ravel()
    counts = np.bincount(flat_labels, minlength=n + 1).astype(np.int64)
    sums = np.bincount(flat_labels, weights=gray.astype(np.float32).ravel(), minlength=n + 1)
    ids = np.arange(1, n + 1, dtype=np.int64)
    areas = counts[ids]
    means = sums[ids] / np.maximum(areas, 1)

    ys, xs = np.nonzero(labels)
    point_labels = labels[ys, xs]
    xmin = np.full(n + 1, labels.shape[1], dtype=np.int32)
    ymin = np.full(n + 1, labels.shape[0], dtype=np.int32)
    xmax = np.zeros(n + 1, dtype=np.int32)
    ymax = np.zeros(n + 1, dtype=np.int32)
    np.minimum.at(xmin, point_labels, xs)
    np.minimum.at(ymin, point_labels, ys)
    np.maximum.at(xmax, point_labels, xs)
    np.maximum.at(ymax, point_labels, ys)

    out: list[dict[str, Any]] = []
    for iid, area, mean in zip(ids, areas, means):
        x0 = int(xmin[iid])
        y0 = int(ymin[iid])
        w = int(xmax[iid] - xmin[iid] + 1)
        h = int(ymax[iid] - ymin[iid] + 1)
        out.append(
            {
                "instance_id": int(iid),
                "area": int(area),
                "mean_intensity": float(mean),
                "bbox_xywh": [x0, y0, w, h],
                "morphology": morphology_from_bbox(w, h),
                "area_bucket": area_level(int(area), areas),
                "intensity_bucket": intensity_level(float(mean), means),
            }
        )
    return out


def build_instances(instance_map: np.ndarray, rgb: np.ndarray, spec: TargetSpec, state: str) -> list[dict[str, Any]]:
    gray = rgb[..., 0].astype(np.float32) / 255.0
    stats = instance_stats(instance_map, gray)
    instances: list[dict[str, Any]] = []
    object_phrase = "nucleus" if spec.target_object == "nucleus" else "whole-cell object"
    for item in stats:
        morph = str(item["morphology"])
        area_bucket = str(item["area_bucket"])
        intensity_bucket = str(item["intensity_bucket"])
        attrs = {
            "target_object": spec.target_object,
            "source_channel": spec.source_channel,
            "morphology": morph,
            "area": area_bucket,
            "intensity": intensity_bucket,
            "text_source": "deepcell_metadata_template",
        }
        instances.append(
            {
                "instance_id": int(item["instance_id"]),
                "category": spec.category,
                "state": state,
                "instance_text": (
                    f"a {object_phrase} from {state} with {morph} morphology, "
                    f"{area_bucket} area, and {intensity_bucket} fluorescence intensity"
                ),
                "bbox_xywh": [int(x) for x in item["bbox_xywh"]],
                "attributes": attrs,
                "measurements": {
                    "area_px": int(item["area"]),
                    "mean_intensity": float(item["mean_intensity"]),
                },
            }
        )
    return instances


def meta_rows(meta: np.ndarray, n_samples: int) -> tuple[list[str], list[dict[str, Any]]]:
    header = [str(x) for x in np.asarray(meta[0]).tolist()]
    rows: list[dict[str, Any]] = []
    for raw in np.asarray(meta[1:], dtype=object):
        values = raw.tolist()
        if len(values) == len(header) and all(str(v) == str(h) for v, h in zip(values, header)):
            continue
        row = {key: json_safe(value) for key, value in zip(header, values)}
        rows.append(row)
        if len(rows) == n_samples:
            break
    while len(rows) < n_samples:
        rows.append({})
    return header, rows


def relative_to_root(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def summarize(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "min": int(arr.min()),
        "mean": float(arr.mean()),
        "max": int(arr.max()),
    }


def state_from_meta(row: dict[str, Any]) -> str:
    specimen = clean_text(row.get("specimen"), default="")
    if specimen:
        return specimen
    experiment = clean_text(row.get("experiment"), default="")
    if experiment:
        return Path(experiment).name.replace("_", " ")
    return "unknown specimen"


def format_global_text(spec: TargetSpec, row: dict[str, Any]) -> str:
    experiment = clean_text(row.get("experiment"), default="unknown experiment")
    if experiment != "unknown experiment":
        experiment = Path(experiment).name
    values = {
        "specimen": clean_text(row.get("specimen"), default="unknown specimen"),
        "pixel_size": clean_text(row.get("pixel_size"), default="unknown"),
        "time_step": clean_text(row.get("time_step"), default="static"),
        "experiment": experiment,
    }
    return spec.global_template.format(**values)


def ensure_dirs(out_dir: Path) -> None:
    for split in SPLITS:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "instance_maps" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "annotations").mkdir(parents=True, exist_ok=True)


def write_metadata_files(
    out_dir: Path,
    *,
    spec: TargetSpec,
    states: Counter[str],
    category_states: Counter[tuple[str, str]],
    args: argparse.Namespace,
) -> None:
    ann_dir = out_dir / "annotations"
    categories_json = {
        "source_dataset": spec.source_dataset,
        "prepared_dataset": spec.name,
        "image_storage": "RGB composite PNG" if len(spec.x_channels) > 1 else "grayscale RGB PNG",
        "image_size": args.image_size,
        "target_object": spec.target_object,
        "source_channel": spec.source_channel,
        "x_channel": spec.x_channel,
        "x_channels": list(spec.x_channels),
        "y_channel": spec.y_channel,
        "percentile_low": args.percentile_low,
        "percentile_high": args.percentile_high,
        "categories": [
            {
                "id": 1,
                "name": spec.category,
                "supercategory": "cellular_object",
            }
        ],
        "states": [{"id": i + 1, "name": name} for i, name in enumerate(sorted(states.keys()))],
        "category_state_labels": [
            {
                "id": i,
                "category": category,
                "state": state,
                "name": f"{category}::{state}",
            }
            for i, (category, state) in enumerate(sorted(category_states.keys()))
        ],
        "attributes": {
            "target_object": [spec.target_object],
            "source_channel": [spec.source_channel],
            "morphology": ["round", "oval", "elongated"],
            "area": ["small", "medium", "large"],
            "intensity": ["low", "medium", "high"],
            "text_source": ["deepcell_metadata_template"],
        },
    }
    (ann_dir / "categories.json").write_text(json.dumps(categories_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    policy = RelationPolicy(use_delaunay=bool(args.use_delaunay))
    (ann_dir / "relation_policy.json").write_text(json.dumps(policy.__dict__, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_npz_meta(npz_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with np.load(npz_path, allow_pickle=True) as data:
        x_shape = tuple(int(x) for x in data["X"].shape)
        y_shape = tuple(int(x) for x in data["y"].shape)
        _header, rows = meta_rows(data["meta"], x_shape[0])
    return {"X": x_shape, "y": y_shape}, rows


def process_split(
    *,
    npz_path: Path,
    out_dir: Path,
    split: str,
    spec: TargetSpec,
    meta: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[int], int]:
    rows: list[dict[str, Any]] = []
    instance_counts: list[int] = []
    skipped_empty = 0

    with np.load(npz_path, allow_pickle=True) as data:
        X = data["X"]
        n = X.shape[0]
        if args.max_images > 0:
            n = min(n, args.max_images)
        original_h, original_w = int(X.shape[1]), int(X.shape[2])
        storage_label = "RGB composite" if len(spec.x_channels) > 1 else "grayscale RGB"
        progress(f"{spec.name} {split}: writing {n} {storage_label} images from {npz_path.name}", quiet=args.quiet)
        for idx in range(n):
            rgb = sample_to_rgb(X[idx], spec, args.image_size, args.percentile_low, args.percentile_high)
            stem = f"{spec.name}_{split}_{idx:06d}"
            image_out = out_dir / "images" / split / f"{stem}.png"
            Image.fromarray(rgb, mode="RGB").save(image_out)
            if should_report(idx + 1, args.progress_every) or idx + 1 == n:
                progress(f"{spec.name} {split}: images {idx + 1}/{n}", quiet=args.quiet)
    gc.collect()

    with np.load(npz_path, allow_pickle=True) as data:
        y = data["y"]
        n = y.shape[0]
        if args.max_images > 0:
            n = min(n, args.max_images)
        progress(f"{spec.name} {split}: writing {n} instance maps and JSONL records", quiet=args.quiet)
        for idx in range(n):
            stem = f"{spec.name}_{split}_{idx:06d}"
            meta_row = meta[idx] if idx < len(meta) else {}
            state = state_from_meta(meta_row)
            raw_mask = y[idx, ..., spec.y_channel] if y.ndim == 4 else y[idx]
            resized_mask = resize_label_map(raw_mask, args.image_size)
            instance_map, _old_ids = relabel_instance_map(resized_mask, args.min_area, args.max_area)
            map_out = out_dir / "instance_maps" / split / f"{stem}_instance.png"
            image_out = out_dir / "images" / split / f"{stem}.png"
            rgb = np.asarray(Image.open(image_out).convert("RGB"), dtype=np.uint8)
            instances = build_instances(instance_map, rgb, spec, state)
            if not instances:
                skipped_empty += 1
                image_out.unlink(missing_ok=True)
                continue
            Image.fromarray(instance_map.astype(np.uint16), mode="I;16").save(map_out)
            rec = {
                "image_id": stem,
                "image_path": relative_to_root(image_out, out_dir),
                "instance_map_path": relative_to_root(map_out, out_dir),
                "height": args.image_size,
                "width": args.image_size,
                "global_text": format_global_text(spec, meta_row),
                "metadata": {
                    "dataset": spec.source_dataset,
                    "prepared_dataset": spec.name,
                    "source_npz": str(npz_path),
                    "source_split": split,
                    "source_index": idx,
                    "source_channel": spec.source_channel,
                    "target_object": spec.target_object,
                    "image_modality": spec.image_modality,
                    "image_storage": "rgb_composite" if len(spec.x_channels) > 1 else "grayscale_rgb",
                    "x_channel": spec.x_channel,
                    "x_channels": list(spec.x_channels),
                    "y_channel": spec.y_channel,
                    "original_height": int(y.shape[1]),
                    "original_width": int(y.shape[2]),
                    "resized_height": args.image_size,
                    "resized_width": args.image_size,
                    "specimen": clean_text(meta_row.get("specimen"), default="unknown specimen"),
                    "pixel_size": json_safe(meta_row.get("pixel_size")),
                    "screening_passed": json_safe(meta_row.get("screening_passed")),
                    "time_step": json_safe(meta_row.get("time_step")),
                    "experiment": json_safe(meta_row.get("experiment")),
                    "filename": json_safe(meta_row.get("filename")),
                    "text_source": "deepcell_metadata_template",
                    "min_area": args.min_area,
                    "max_area": args.max_area,
                    "original_X_height": original_h,
                    "original_X_width": original_w,
                },
                "instances": instances,
            }
            rows.append(rec)
            instance_counts.append(len(instances))
            if should_report(idx + 1, args.progress_every) or idx + 1 == n:
                progress(
                    f"{spec.name} {split}: records {idx + 1}/{n}; written={len(rows)}; "
                    f"skipped_empty={skipped_empty}; last_instances={len(instances)}",
                    quiet=args.quiet,
                )
    gc.collect()
    return rows, instance_counts, skipped_empty


def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    spec = TARGETS[args.target]
    source_dir = Path(args.source or spec.source_root_default).resolve()
    out_dir = Path(args.out or (Path("exp5/data") / spec.name)).resolve()
    progress(f"start target={args.target} source={source_dir} out={out_dir}", quiet=args.quiet)
    ensure_dirs(out_dir)

    split_counts: dict[str, int] = {}
    instance_counts: dict[str, list[int]] = {}
    skipped_empty_by_split: dict[str, int] = {}
    split_shapes: dict[str, dict[str, Any]] = {}
    states: Counter[str] = Counter()
    category_states: Counter[tuple[str, str]] = Counter()

    for split in SPLITS:
        npz_path = source_dir / spec.npz_names[split]
        if not npz_path.is_file():
            raise FileNotFoundError(f"Missing split file: {npz_path}")
        shapes, meta = load_npz_meta(npz_path)
        split_shapes[split] = shapes
        rows, counts, skipped_empty = process_split(
            npz_path=npz_path,
            out_dir=out_dir,
            split=split,
            spec=spec,
            meta=meta,
            args=args,
        )
        write_jsonl(out_dir / "annotations" / f"{split}.jsonl", rows)
        split_counts[split] = len(rows)
        instance_counts[split] = counts
        skipped_empty_by_split[split] = skipped_empty
        for row in rows:
            for inst in row["instances"]:
                category = str(inst.get("category", spec.category))
                state = str(inst.get("state", "unknown specimen"))
                states[state] += 1
                category_states[(category, state)] += 1
        progress(f"{spec.name} {split}: finished records={len(rows)} skipped_empty={skipped_empty}", quiet=args.quiet)

    write_metadata_files(out_dir, spec=spec, states=states, category_states=category_states, args=args)
    manifest = {
        "source": str(source_dir),
        "output": str(out_dir),
        "prepared_dataset": spec.name,
        "source_dataset": spec.source_dataset,
        "image_size": args.image_size,
        "image_storage": "RGB composite PNG" if len(spec.x_channels) > 1 else "grayscale RGB PNG",
        "resize_policy": "bicubic for image, nearest for instance map",
        "target_object": spec.target_object,
        "source_channel": spec.source_channel,
        "x_channel": spec.x_channel,
        "x_channels": list(spec.x_channels),
        "y_channel": spec.y_channel,
        "percentile_low": args.percentile_low,
        "percentile_high": args.percentile_high,
        "min_area": args.min_area,
        "max_area": args.max_area,
        "splits": split_counts,
        "skipped_empty": skipped_empty_by_split,
        "source_shapes": split_shapes,
        "instance_count_summary": {split: summarize(values) for split, values in instance_counts.items()},
        "source_records": int(sum(split_counts.values()) + sum(skipped_empty_by_split.values())),
        "max_images": args.max_images,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    progress(f"done: wrote {sum(split_counts.values())} records to {out_dir}", quiet=args.quiet)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Prepare DeepCell TissueNet/DynamicNuclearNet NPZ files as TLC-CellDiff datasets.")
    ap.add_argument("--target", choices=tuple(TARGETS.keys()), required=True)
    ap.add_argument("--source", type=str, default="", help="Raw DeepCell dataset root. Defaults depend on --target.")
    ap.add_argument("--out", type=str, default="", help="Output TLC dataset root. Defaults to exp5/data/<prepared_dataset>.")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--percentile_low", type=float, default=1.0)
    ap.add_argument("--percentile_high", type=float, default=99.8)
    ap.add_argument("--min_area", type=int, default=8)
    ap.add_argument("--max_area", type=int, default=250000)
    ap.add_argument("--max_images", type=int, default=0, help="Optional per-split cap for smoke tests; 0 uses all images.")
    ap.add_argument("--progress_every", type=int, default=250, help="Print progress every N images.")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--use_delaunay", action="store_true", help="Enable Delaunay edges in relation_policy.json.")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare_dataset(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
