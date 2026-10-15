from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageSequence
from scipy import ndimage

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tlc_mplconfig"))

try:
    from skimage.feature import peak_local_max
    from skimage.morphology import binary_closing, disk
    from skimage.segmentation import watershed
except Exception:  # pragma: no cover - fallback keeps the converter usable.
    peak_local_max = None
    binary_closing = None
    disk = None
    watershed = None

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mscdms.workflows.parent_data_utils import RelationPolicy
from mscdms.workflows.parent_data_utils import bbox_xywh_from_mask


IMAGE_RE = re.compile(r"field_(?P<field>\d+)_image_(?P<image_id>\d+)\.tiff?$", re.IGNORECASE)
PLATE_SUFFIX_RE = re.compile(r"_(?:illum_corrected|raw|processed)$", re.IGNORECASE)


@dataclass(frozen=True)
class SourceImage:
    path: Path
    plate_name: str
    plate_id: str
    well: str
    field: int
    image_id: str
    cell_outline: Path | None
    nuclei_outline: Path | None


def parse_channel_list(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def normalized_plate_id(plate_name: str) -> str:
    return PLATE_SUFFIX_RE.sub("", plate_name)


def load_annotation_rows(path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            plate = str(row.get("Plate", "")).strip()
            well = str(row.get("Well", "")).strip()
            if plate and well:
                rows[(plate, well)] = row
    return rows


def load_selection_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Load an explicit well-grouped split assignment.

    Required columns are ``plate``, ``well`` and ``split``.  Extra columns are
    preserved in each output record's metadata so expansion roles remain
    auditable downstream.
    """
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for line_no, row in enumerate(csv.DictReader(f, delimiter="\t"), start=2):
            plate = normalized_plate_id(clean_text(row.get("plate")))
            well = clean_text(row.get("well")).upper()
            split = clean_text(row.get("split")).lower()
            if not plate or not well:
                raise ValueError(f"{path}:{line_no}: plate/well 不能为空")
            if split not in {"train", "val", "test"}:
                raise ValueError(
                    f"{path}:{line_no}: split={split!r}，仅支持 train/val/test"
                )
            key = (plate, well)
            if key in rows:
                raise ValueError(f"{path}:{line_no}: 重复 plate/well {key}")
            normalized = {
                clean_text(key_name): clean_text(value)
                for key_name, value in row.items()
                if key_name is not None
            }
            normalized["plate"] = plate
            normalized["well"] = well
            normalized["split"] = split
            rows[key] = normalized
    if not rows:
        raise ValueError(f"显式 split 文件为空：{path}")
    return rows


def discover_source_images(source: Path, mask_kind: str) -> list[SourceImage]:
    image_paths = sorted(source.glob("**/field_*_image_*.tif*"))
    out: list[SourceImage] = []
    for path in image_paths:
        m = IMAGE_RE.match(path.name)
        if m is None:
            continue
        field = int(m.group("field"))
        image_id = m.group("image_id")
        well = path.parent.name
        plate_name = path.parent.parent.name if path.parent.parent != path.parent else ""
        plate_id = normalized_plate_id(plate_name)
        masks_dir = path.parent / f"field_{field}_masks"
        cell_outline = first_existing(masks_dir.glob("*CellOutlines.png"))
        nuclei_outline = first_existing(masks_dir.glob("*NucleiOutlines.png"))
        selected = cell_outline if mask_kind == "cell" else nuclei_outline
        if selected is None:
            continue
        out.append(
            SourceImage(
                path=path,
                plate_name=plate_name,
                plate_id=plate_id,
                well=well,
                field=field,
                image_id=image_id,
                cell_outline=cell_outline,
                nuclei_outline=nuclei_outline,
            )
        )
    return out


def first_existing(paths: Any) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def load_tiff_frames(path: Path) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    with Image.open(path) as im:
        for frame in ImageSequence.Iterator(im):
            frames.append(np.asarray(frame, dtype=np.float32))
    if not frames:
        raise ValueError(f"No frames found in {path}")
    return frames


def normalize_channel(arr: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    lo, hi = np.percentile(arr, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(arr.min())
        hi = float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def merge_channels(frames: list[np.ndarray], channels: tuple[int, ...]) -> np.ndarray:
    valid = [frames[i] for i in channels if 0 <= i < len(frames)]
    if not valid:
        valid = [frames[min(len(frames) - 1, 0)]]
    if len(valid) == 1:
        return valid[0]
    return np.maximum.reduce(valid)


def tiff_to_rgb(
    path: Path,
    *,
    red_channels: tuple[int, ...],
    green_channels: tuple[int, ...],
    blue_channels: tuple[int, ...],
    p_low: float,
    p_high: float,
    image_size: int,
) -> np.ndarray:
    frames = load_tiff_frames(path)
    if len(frames) == 1:
        base = normalize_channel(frames[0], p_low, p_high)
        rgb = np.stack([base, base, base], axis=-1)
    else:
        red = normalize_channel(merge_channels(frames, red_channels), p_low, p_high)
        green = normalize_channel(merge_channels(frames, green_channels), p_low, p_high)
        blue = normalize_channel(merge_channels(frames, blue_channels), p_low, p_high)
        rgb = np.stack([red, green, blue], axis=-1)
    im = Image.fromarray((rgb * 255.0).round().astype(np.uint8), mode="RGB")
    if im.size != (image_size, image_size):
        im = im.resize((image_size, image_size), Image.Resampling.BICUBIC)
    return np.asarray(im, dtype=np.uint8)


def load_outline(path: Path, threshold: int = 128) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"))
    return arr > threshold


def outlines_to_instance_map(
    outline_path: Path,
    *,
    image_size: int,
    min_area: int,
    max_area: int,
    peak_min_distance: int,
) -> np.ndarray:
    outlines = load_outline(outline_path)
    if binary_closing is not None and disk is not None:
        outlines = binary_closing(outlines, disk(1))
    filled = ndimage.binary_fill_holes(outlines)
    foreground = filled & ~outlines
    if not np.any(foreground):
        labels, _ = ndimage.label(~outlines)
    else:
        labels = watershed_instances(foreground, peak_min_distance)
    labels = filter_and_relabel_connected(labels, min_area=min_area, max_area=max_area)
    im = Image.fromarray(labels.astype(np.uint16), mode="I;16")
    if im.size != (image_size, image_size):
        im = im.resize((image_size, image_size), Image.Resampling.NEAREST)
    resized = np.asarray(im).astype(np.int64, copy=False)
    resized = filter_and_relabel(resized, min_area=max(1, int(min_area * (image_size / max(outlines.shape)) ** 2)), max_area=max_area)
    return resized.astype(np.uint16)


def watershed_instances(foreground: np.ndarray, peak_min_distance: int) -> np.ndarray:
    distance = ndimage.distance_transform_edt(foreground)
    if peak_local_max is None or watershed is None:
        labels, _ = ndimage.label(foreground)
        return labels
    coords = peak_local_max(distance, min_distance=peak_min_distance, exclude_border=False)
    if coords.size == 0:
        markers, _ = ndimage.label(foreground)
    else:
        local_max = np.zeros(distance.shape, dtype=bool)
        local_max[tuple(coords.T)] = True
        markers, _ = ndimage.label(local_max)
    return watershed(-distance, markers, mask=foreground)


def filter_and_relabel(labels: np.ndarray, *, min_area: int, max_area: int) -> np.ndarray:
    out = np.zeros(labels.shape, dtype=np.uint16)
    next_id = 1
    for label_id in sorted(int(x) for x in np.unique(labels) if int(x) != 0):
        mask = labels == label_id
        area = int(mask.sum())
        if min_area <= area <= max_area:
            out[mask] = next_id
            next_id += 1
    return out


def filter_and_relabel_connected(labels: np.ndarray, *, min_area: int, max_area: int) -> np.ndarray:
    binary = np.zeros(labels.shape, dtype=bool)
    for label_id in sorted(int(x) for x in np.unique(labels) if int(x) != 0):
        mask = labels == label_id
        area = int(mask.sum())
        if min_area <= area <= max_area:
            binary |= mask
    connected, _ = ndimage.label(binary)
    return filter_and_relabel(connected, min_area=min_area, max_area=max_area)


def phenotype_values(row: dict[str, str]) -> list[str]:
    values: list[str] = []
    for key, value in row.items():
        if not key.startswith("Phenotype "):
            continue
        if "Term Accession" in key or "Term Name" in key:
            continue
        value = str(value or "").strip()
        if value:
            values.append(value)
    return values


def phenotype_terms(row: dict[str, str]) -> list[str]:
    values: list[str] = []
    for key, value in row.items():
        if "Term Name" not in key:
            continue
        value = str(value or "").strip()
        if value:
            values.append(value)
    return values


def category_from_row(row: dict[str, str]) -> str:
    return clean_text(row.get("Characteristics [Cell Line]")) or "U2OS"


def state_from_row(row: dict[str, str]) -> str:
    control_type = clean_text(row.get("Control Type"))
    control_comments = clean_text(row.get("Control Comments"))
    gene = clean_text(row.get("Gene Symbol"))
    allele = clean_text(row.get("Gene Allele Name"))
    phenos = phenotype_terms(row)
    if control_type:
        suffix = f": {control_comments}" if control_comments else ""
        return f"{control_type}{suffix}"
    if phenos:
        return "; ".join(phenos[:2])
    if gene:
        if allele and allele != gene:
            return f"{gene} ORF perturbation ({allele})"
        return f"{gene} ORF perturbation"
    return "unknown perturbation"


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def global_text(row: dict[str, str], source: SourceImage, category: str, state: str) -> str:
    channels = clean_text(row.get("Channels"))
    bits = [
        f"fluorescence microscopy image of {category} cells from IDR0033",
        state,
        "five-channel Cell Painting composite",
    ]
    if channels:
        bits.append("channels: " + channels)
    bits.append(f"plate {source.plate_id} well {source.well} field {source.field}")
    return "; ".join(bits)


def area_level(area: int, areas: list[int]) -> str:
    if not areas:
        return "medium"
    q1, q2 = np.percentile(np.asarray(areas, dtype=np.float32), [33.0, 66.0])
    if area <= q1:
        return "small"
    if area >= q2:
        return "large"
    return "medium"


def intensity_level(value: float, values: list[float]) -> str:
    if not values:
        return "medium"
    q1, q2 = np.percentile(np.asarray(values, dtype=np.float32), [33.0, 66.0])
    if value <= q1:
        return "low"
    if value >= q2:
        return "high"
    return "medium"


def morphology_from_bbox(bbox: list[int]) -> str:
    _, _, w, h = bbox
    short = max(1.0, float(min(w, h)))
    ratio = float(max(w, h)) / short
    if ratio >= 1.75:
        return "elongated"
    if ratio <= 1.2:
        return "round"
    return "oval"


def build_instances(instance_map: np.ndarray, rgb: np.ndarray, category: str, state: str) -> list[dict[str, Any]]:
    instance_ids = [int(x) for x in np.unique(instance_map) if int(x) != 0]
    areas = [int((instance_map == iid).sum()) for iid in instance_ids]
    gray = rgb.astype(np.float32).mean(axis=-1)
    means = [float(gray[instance_map == iid].mean()) if areas[idx] else 0.0 for idx, iid in enumerate(instance_ids)]
    instances: list[dict[str, Any]] = []
    for iid, area, mean_intensity in zip(instance_ids, areas, means):
        mask = instance_map == iid
        bbox = [int(x) for x in bbox_xywh_from_mask(mask)]
        morph = morphology_from_bbox(bbox)
        area_bucket = area_level(area, areas)
        intensity_bucket = intensity_level(mean_intensity, means)
        attrs = {
            "morphology": morph,
            "area": area_bucket,
            "intensity": intensity_bucket,
            "text_source": "template",
        }
        instances.append(
            {
                "instance_id": iid,
                "category": category,
                "state": state,
                "instance_text": f"a {category} cell under {state} with {morph} morphology, {area_bucket} area, and {intensity_bucket} intensity",
                "bbox_xywh": bbox,
                "attributes": attrs,
            }
        )
    return instances


def split_sources(
    sources: list[SourceImage],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[SourceImage]]:
    rng = random.Random(seed)
    shuffled = list(sources)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(round(n * test_fraction)) if n >= 3 else 0
    n_val = int(round(n * val_fraction)) if n >= 2 else 0
    if n >= 3 and n_val == 0 and val_fraction > 0:
        n_val = 1
    if n >= 4 and n_test == 0 and test_fraction > 0:
        n_test = 1
    if n_val + n_test >= n:
        n_val = min(n_val, max(0, n - 1))
        n_test = min(n_test, max(0, n - 1 - n_val))
    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return {
        "train": sorted(train, key=source_sort_key),
        "val": sorted(val, key=source_sort_key),
        "test": sorted(test, key=source_sort_key),
    }


def source_sort_key(source: SourceImage) -> tuple[str, str, int, str]:
    return source.plate_id, source.well, source.field, source.image_id


def relative_to_root(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_metadata_files(
    out_dir: Path,
    *,
    categories: Counter[str],
    states: Counter[str],
    category_states: Counter[tuple[str, str]],
    args: argparse.Namespace,
) -> None:
    ann_dir = out_dir / "annotations"
    cat_names = sorted(categories.keys())
    state_names = sorted(states.keys())
    category_rows = [{"id": i + 1, "name": name, "supercategory": "cell"} for i, name in enumerate(cat_names)]
    state_rows = [{"id": i + 1, "name": name} for i, name in enumerate(state_names)]
    label_rows = []
    for i, (category, state) in enumerate(sorted(category_states.keys())):
        label_rows.append(
            {
                "id": i,
                "category": category,
                "state": state,
                "name": f"{category}::{state}",
            }
        )
    categories_json = {
        "source_dataset": "idr0033-rohban-pathways/screenA",
        "mask_kind": args.mask_kind,
        "image_composite": {
            "red_channels": list(args.red_channels),
            "green_channels": list(args.green_channels),
            "blue_channels": list(args.blue_channels),
            "percentile_low": args.percentile_low,
            "percentile_high": args.percentile_high,
        },
        "categories": category_rows,
        "states": state_rows,
        "category_state_labels": label_rows,
        "attributes": {
            "morphology": ["round", "oval", "elongated"],
            "area": ["small", "medium", "large"],
            "intensity": ["low", "medium", "high"],
            "text_source": ["template"],
        },
    }
    (ann_dir / "categories.json").write_text(json.dumps(categories_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    policy = RelationPolicy(use_delaunay=bool(args.use_delaunay))
    (ann_dir / "relation_policy.json").write_text(json.dumps(policy.__dict__, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = Path(args.source).resolve()
    out_dir = Path(args.out).resolve()
    annotation_path = Path(args.annotation_csv).resolve() if args.annotation_csv else source_dir / "screenA" / "idr0033-screenA-annotation.csv"
    annotation_rows = load_annotation_rows(annotation_path)
    sources = discover_source_images(source_dir, args.mask_kind)
    if not sources:
        raise SystemExit(f"No source TIFF images with {args.mask_kind} outlines found under {source_dir}")

    selection_path = (
        Path(args.selection_tsv).resolve() if args.selection_tsv else None
    )
    selection_rows: dict[tuple[str, str], dict[str, str]] = {}
    if selection_path is not None:
        selection_rows = load_selection_rows(selection_path)
        discovered_keys = {(source.plate_id, source.well) for source in sources}
        missing_wells = sorted(set(selection_rows) - discovered_keys)
        if missing_wells:
            raise ValueError(
                "显式 split 中有未下载或缺 mask 的孔："
                + ", ".join(f"{plate}/{well}" for plate, well in missing_wells)
            )
        sources = [
            source
            for source in sources
            if (source.plate_id, source.well) in selection_rows
        ]
        split_map = {
            split: sorted(
                [
                    source
                    for source in sources
                    if selection_rows[(source.plate_id, source.well)]["split"]
                    == split
                ],
                key=source_sort_key,
            )
            for split in ("train", "val", "test")
        }
    else:
        split_map = split_sources(
            sources,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )

    if args.max_images > 0:
        remaining = args.max_images
        limited: dict[str, list[SourceImage]] = {}
        for split in ("train", "val", "test"):
            limited[split] = split_map[split][:remaining]
            remaining -= len(limited[split])
            if remaining <= 0:
                remaining = 0
        split_map = limited

    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "instance_maps" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "annotations").mkdir(parents=True, exist_ok=True)

    categories: Counter[str] = Counter()
    states: Counter[str] = Counter()
    category_states: Counter[tuple[str, str]] = Counter()
    split_counts: dict[str, int] = {}
    instance_counts: dict[str, list[int]] = {}

    for split, split_sources_list in split_map.items():
        rows: list[dict[str, Any]] = []
        instance_counts[split] = []
        for source in split_sources_list:
            row = annotation_rows.get((source.plate_id, source.well), {})
            selection = selection_rows.get((source.plate_id, source.well), {})
            category = category_from_row(row)
            state = state_from_row(row)
            categories[category] += 1
            states[state] += 1
            category_states[(category, state)] += 1
            selected_outline = source.cell_outline if args.mask_kind == "cell" else source.nuclei_outline
            if selected_outline is None:
                continue

            stem = f"idr0033_{source.plate_id}_{source.well}_s{source.field}_img{source.image_id}"
            image_out = out_dir / "images" / split / f"{stem}.png"
            map_out = out_dir / "instance_maps" / split / f"{stem}_instance.png"
            rgb = tiff_to_rgb(
                source.path,
                red_channels=args.red_channels,
                green_channels=args.green_channels,
                blue_channels=args.blue_channels,
                p_low=args.percentile_low,
                p_high=args.percentile_high,
                image_size=args.image_size,
            )
            instance_map = outlines_to_instance_map(
                selected_outline,
                image_size=args.image_size,
                min_area=args.min_area,
                max_area=args.max_area,
                peak_min_distance=args.peak_min_distance,
            )
            instances = build_instances(instance_map, rgb, category, state)
            if len(instances) == 0:
                continue

            Image.fromarray(rgb, mode="RGB").save(image_out)
            Image.fromarray(instance_map.astype(np.uint16), mode="I;16").save(map_out)
            phenotypes = phenotype_values(row)
            terms = phenotype_terms(row)
            rec = {
                "image_id": stem,
                "image_path": relative_to_root(image_out, out_dir),
                "instance_map_path": relative_to_root(map_out, out_dir),
                "height": args.image_size,
                "width": args.image_size,
                "global_text": global_text(row, source, category, state),
                "metadata": {
                    "dataset": "idr0033",
                    "screen": "screenA",
                    "source_image_id": source.image_id,
                    "source_image_path": str(source.path),
                    "source_outline_path": str(selected_outline),
                    "plate": source.plate_id,
                    "plate_name": source.plate_name,
                    "well": source.well,
                    "field": source.field,
                    "organism": clean_text(row.get("Characteristics [Organism]")) or "Homo sapiens",
                    "cell_line": category,
                    "gene_symbol": clean_text(row.get("Gene Symbol")),
                    "gene_identifier": clean_text(row.get("Gene Identifier")),
                    "gene_allele_name": clean_text(row.get("Gene Allele Name")),
                    "control_type": clean_text(row.get("Control Type")),
                    "control_comments": clean_text(row.get("Control Comments")),
                    "morphological_cluster_id": clean_text(row.get("Morphological Cluster ID")),
                    "has_phenotype": clean_text(row.get("Has Phenotype")),
                    "phenotypes": phenotypes,
                    "phenotype_terms": terms,
                    "channels": clean_text(row.get("Channels")),
                    "mask_kind": args.mask_kind,
                    "text_source": "metadata_template",
                    "data_role": clean_text(selection.get("data_role")),
                    "selection_reason": clean_text(selection.get("reason")),
                    "selection_split": clean_text(selection.get("split")),
                },
                "instances": instances,
            }
            rows.append(rec)
            instance_counts[split].append(len(instances))
        write_jsonl(out_dir / "annotations" / f"{split}.jsonl", rows)
        split_counts[split] = len(rows)

    write_metadata_files(out_dir, categories=categories, states=states, category_states=category_states, args=args)
    manifest = {
        "source": str(source_dir),
        "annotation_csv": str(annotation_path),
        "selection_tsv": str(selection_path) if selection_path else None,
        "split_unit": "well_explicit" if selection_path else "field_random",
        "output": str(out_dir),
        "image_size": args.image_size,
        "mask_kind": args.mask_kind,
        "splits": split_counts,
        "instance_count_summary": {
            split: summarize(values) for split, values in instance_counts.items()
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Prepare downloaded IDR0033 wells as TLC-CellDiff JSONL datasets.")
    ap.add_argument("--source", type=str, default="idr0033-rohban-pathways", help="IDR0033 download root or a subdirectory containing field_* TIFF files.")
    ap.add_argument("--out", type=str, required=True, help="Output TLC dataset root.")
    ap.add_argument("--annotation_csv", type=str, default="", help="Path to idr0033-screenA-annotation.csv. Defaults to SOURCE/screenA/...")
    ap.add_argument(
        "--selection_tsv",
        type=str,
        default="",
        help=(
            "Optional TSV with plate, well, split columns. When supplied, "
            "only listed wells are prepared and fields from a well cannot "
            "leak across splits."
        ),
    )
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--mask_kind", choices=("cell", "nuclei"), default="cell")
    ap.add_argument("--red_channels", type=parse_channel_list, default=parse_channel_list("3,4"), help="0-based TIFF frames merged into RGB red.")
    ap.add_argument("--green_channels", type=parse_channel_list, default=parse_channel_list("1,2"), help="0-based TIFF frames merged into RGB green.")
    ap.add_argument("--blue_channels", type=parse_channel_list, default=parse_channel_list("0"), help="0-based TIFF frames merged into RGB blue.")
    ap.add_argument("--percentile_low", type=float, default=1.0)
    ap.add_argument("--percentile_high", type=float, default=99.8)
    ap.add_argument("--min_area", type=int, default=100, help="Minimum source-resolution instance area.")
    ap.add_argument("--max_area", type=int, default=50000, help="Maximum source-resolution instance area.")
    ap.add_argument("--peak_min_distance", type=int, default=5)
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--test_fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_images", type=int, default=0, help="Optional cap for quick tests; 0 uses all discovered images.")
    ap.add_argument("--use_delaunay", action="store_true", help="Enable Delaunay edges in relation_policy.json.")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare_dataset(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
