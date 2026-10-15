from __future__ import annotations

import argparse
import csv
import json
import os
import random
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
from mscdms.workflows.parent_data_utils import bbox_xywh_from_mask


MeasurementKey = tuple[str, str, str, int, int]
MeasurementRow = dict[str, str]

MASK_FIELDS = {
    "cell": "FileName_CellsImage",
    "cytoplasm": "FileName_CytoplasmImage",
    "nuclei": "FileName_NucleiImage",
}

MEASUREMENT_FILES = {
    "cell": "bbbc021_Cells.csv",
    "cytoplasm": "bbbc021_Cytoplasm.csv",
    "nuclei": "bbbc021_Nuclei.csv",
}

MEASUREMENT_VALUE_COLUMNS = {
    "AreaShape_Area",
    "AreaShape_Eccentricity",
    "AreaShape_FormFactor",
    "AreaShape_Compactness",
    "Intensity_MeanIntensity_CorrDAPI",
    "Intensity_MeanIntensity_CorrTub",
    "Intensity_MeanIntensity_CorrActin",
    "Neighbors_NumberOfNeighbors_10",
    "Neighbors_FirstClosestDistance_10",
}


def progress(message: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(f"[prepare_bbbc021] {message}", flush=True)


def should_report(count: int, every: int) -> bool:
    return count == 1 or (every > 0 and count % every == 0)


def display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class SourceRecord:
    image_csv: Path
    week: str
    plate: str
    well: str
    site: int
    image_number: int
    row: dict[str, str]
    dapi_path: Path
    tubulin_path: Path
    actin_path: Path
    cell_mask_path: Path
    cytoplasm_mask_path: Path
    nuclei_mask_path: Path
    overlay_path: Path | None

    def mask_path(self, mask_kind: str) -> Path:
        if mask_kind == "cell":
            return self.cell_mask_path
        if mask_kind == "cytoplasm":
            return self.cytoplasm_mask_path
        if mask_kind == "nuclei":
            return self.nuclei_mask_path
        raise ValueError(f"Unknown mask kind: {mask_kind}")


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except Exception:
        return default


def to_float(value: Any) -> float | None:
    try:
        out = float(str(value))
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def iter_csv_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def find_week_dir(image_csv: Path) -> Path:
    for parent in image_csv.parents:
        if parent.name == "measurements":
            return parent.parent
    raise ValueError(f"Cannot infer BBBC021 week directory from {image_csv}")


def local_file(directory: Path, filename: str) -> Path | None:
    name = clean_text(filename)
    if not name:
        return None
    path = directory / name
    if path.is_file():
        return path
    alt = path.with_suffix(".tiff" if path.suffix.lower() in {".tif", ""} else ".tif")
    if alt.is_file():
        return alt
    return None


def discover_source_records(
    source_dir: Path,
    mask_kind: str,
    *,
    progress_every: int,
    quiet: bool,
) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    image_csvs = sorted(source_dir.glob("**/bbbc021_Image.csv"))
    progress(f"found {len(image_csvs)} image metadata CSV files under {source_dir}", quiet=quiet)
    for csv_idx, image_csv in enumerate(image_csvs, start=1):
        before = len(records)
        week_dir = find_week_dir(image_csv)
        week = week_dir.name
        for row in iter_csv_rows(image_csv):
            plate = clean_text(row.get("Metadata_Plate"))
            well = clean_text(row.get("Metadata_Well"))
            site = to_int(row.get("Metadata_Site"))
            image_number = to_int(row.get("ImageNumber"))
            if not plate or not well or site <= 0 or image_number <= 0:
                continue

            illum_dir = week_dir / "illum_corrected" / plate
            label_dir = week_dir / "labels" / plate
            overlay_dir = week_dir / "overlay" / plate
            dapi = local_file(illum_dir, row.get("FileName_CorrDAPI", ""))
            tubulin = local_file(illum_dir, row.get("FileName_CorrTub", ""))
            actin = local_file(illum_dir, row.get("FileName_CorrActin", ""))
            cell_mask = local_file(label_dir, row.get("FileName_CellsImage", ""))
            cytoplasm_mask = local_file(label_dir, row.get("FileName_CytoplasmImage", ""))
            nuclei_mask = local_file(label_dir, row.get("FileName_NucleiImage", ""))
            selected_mask = {
                "cell": cell_mask,
                "cytoplasm": cytoplasm_mask,
                "nuclei": nuclei_mask,
            }[mask_kind]
            if not all([dapi, tubulin, actin, cell_mask, cytoplasm_mask, nuclei_mask, selected_mask]):
                continue
            overlay_name = clean_text(row.get("FileName_OrigOverlay"))
            overlay = local_file(overlay_dir, overlay_name) if overlay_name else None
            records.append(
                SourceRecord(
                    image_csv=image_csv,
                    week=week,
                    plate=plate,
                    well=well,
                    site=site,
                    image_number=image_number,
                    row=row,
                    dapi_path=dapi,
                    tubulin_path=tubulin,
                    actin_path=actin,
                    cell_mask_path=cell_mask,
                    cytoplasm_mask_path=cytoplasm_mask,
                    nuclei_mask_path=nuclei_mask,
                    overlay_path=overlay,
                )
            )
        added = len(records) - before
        if should_report(csv_idx, max(1, progress_every)) or csv_idx == len(image_csvs):
            progress(
                f"scanned metadata {csv_idx}/{len(image_csvs)}: "
                f"{display_path(image_csv, source_dir)}; usable records +{added}, total {len(records)}",
                quiet=quiet,
            )
    return sorted(records, key=source_sort_key)


def source_sort_key(source: SourceRecord) -> tuple[str, str, int, int]:
    return source.plate, source.well, source.site, source.image_number


def measurement_week(path: Path) -> str:
    for parent in path.parents:
        if parent.name == "measurements":
            return parent.parent.name
    return ""


def measurement_key(row: dict[str, str], week: str) -> MeasurementKey | None:
    plate = clean_text(row.get("Metadata_Plate"))
    well = clean_text(row.get("Metadata_Well"))
    image_number = to_int(row.get("ImageNumber"))
    object_number = to_int(row.get("ObjectNumber"))
    if not week or not plate or not well or image_number <= 0 or object_number <= 0:
        return None
    return week, plate, well, image_number, object_number


def compact_measurement_row(row: dict[str, str]) -> MeasurementRow:
    out = {
        "ImageNumber": clean_text(row.get("ImageNumber")),
        "ObjectNumber": clean_text(row.get("ObjectNumber")),
        "Metadata_Plate": clean_text(row.get("Metadata_Plate")),
        "Metadata_Well": clean_text(row.get("Metadata_Well")),
    }
    for key in MEASUREMENT_VALUE_COLUMNS:
        value = row.get(key)
        if value is not None and value != "":
            out[key] = value
    return out


def load_group_measurements(sources: list[SourceRecord]) -> dict[str, dict[MeasurementKey, MeasurementRow]]:
    out: dict[str, dict[MeasurementKey, MeasurementRow]] = {kind: {} for kind in MEASUREMENT_FILES}
    if not sources:
        return out
    image_numbers = {source.image_number for source in sources}
    measurement_dir = sources[0].image_csv.parent
    for kind, filename in MEASUREMENT_FILES.items():
        path = measurement_dir / filename
        if not path.is_file():
            continue
        week = measurement_week(path)
        for row in iter_csv_rows(path):
            image_number = to_int(row.get("ImageNumber"))
            if image_number not in image_numbers:
                continue
            key = measurement_key(row, week)
            if key is not None:
                out[kind][key] = compact_measurement_row(row)
    return out


def group_sources_by_measurement_dir(sources: list[SourceRecord]) -> list[list[SourceRecord]]:
    grouped: dict[Path, list[SourceRecord]] = {}
    for source in sources:
        grouped.setdefault(source.image_csv.parent, []).append(source)
    return [sorted(rows, key=source_sort_key) for _, rows in sorted(grouped.items(), key=lambda x: x[0].as_posix())]


def normalize_channel(arr: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    lo, hi = np.percentile(arr, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(arr.min())
        hi = float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def read_grayscale(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def bbbc021_to_rgb(
    source: SourceRecord,
    *,
    p_low: float,
    p_high: float,
    image_size: int,
) -> np.ndarray:
    dapi = normalize_channel(read_grayscale(source.dapi_path), p_low, p_high)
    tubulin = normalize_channel(read_grayscale(source.tubulin_path), p_low, p_high)
    actin = normalize_channel(read_grayscale(source.actin_path), p_low, p_high)
    rgb = np.stack([actin, tubulin, dapi], axis=-1)
    im = Image.fromarray((rgb * 255.0).round().astype(np.uint8), mode="RGB")
    if im.size != (image_size, image_size):
        im = im.resize((image_size, image_size), Image.Resampling.BICUBIC)
    return np.asarray(im, dtype=np.uint8)


def load_instance_map(path: Path, *, image_size: int, min_area: int, max_area: int) -> np.ndarray:
    arr = read_grayscale(path)
    im = Image.fromarray(arr.astype(np.uint16), mode="I;16")
    if im.size != (image_size, image_size):
        im = im.resize((image_size, image_size), Image.Resampling.NEAREST)
    resized = np.asarray(im).astype(np.int64, copy=False)
    out = np.zeros(resized.shape, dtype=np.uint16)
    for label_id in sorted(int(x) for x in np.unique(resized) if int(x) != 0):
        mask = resized == label_id
        area = int(mask.sum())
        if min_area <= area <= max_area:
            out[mask] = label_id
    return out


def area_level(area: int, areas: list[int]) -> str:
    if not areas:
        return "medium"
    arr = np.asarray(areas, dtype=np.float32)
    q1, q2 = np.percentile(arr, [33.0, 66.0])
    if area <= q1:
        return "small"
    if area >= q2:
        return "large"
    return "medium"


def bucket_value(value: float | None, values: list[float]) -> str:
    if value is None or not values:
        return "medium"
    arr = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float32)
    if arr.size == 0:
        return "medium"
    q1, q2 = np.percentile(arr, [33.0, 66.0])
    if value <= q1:
        return "low"
    if value >= q2:
        return "high"
    return "medium"


def intensity_level(value: float | None, values: list[float]) -> str:
    return bucket_value(value, values)


def morphology_from_bbox(bbox: list[int]) -> str:
    _, _, w, h = bbox
    short = max(1.0, float(min(w, h)))
    ratio = float(max(w, h)) / short
    if ratio >= 1.75:
        return "elongated"
    if ratio <= 1.2:
        return "round"
    return "oval"


def source_key(source: SourceRecord, instance_id: int) -> MeasurementKey:
    return source.week, source.plate, source.well, source.image_number, int(instance_id)


def measurement_float(row: dict[str, str] | None, key: str) -> float | None:
    if row is None:
        return None
    return to_float(row.get(key))


def compact_measurements(
    cell_row: MeasurementRow | None,
    nuclei_row: MeasurementRow | None,
    cytoplasm_row: MeasurementRow | None,
) -> dict[str, float]:
    specs = [
        ("cell_area", cell_row, "AreaShape_Area"),
        ("cell_eccentricity", cell_row, "AreaShape_Eccentricity"),
        ("cell_form_factor", cell_row, "AreaShape_FormFactor"),
        ("cell_compactness", cell_row, "AreaShape_Compactness"),
        ("nucleus_area", nuclei_row, "AreaShape_Area"),
        ("cytoplasm_area", cytoplasm_row, "AreaShape_Area"),
        ("mean_intensity_dapi", cell_row, "Intensity_MeanIntensity_CorrDAPI"),
        ("mean_intensity_tubulin", cell_row, "Intensity_MeanIntensity_CorrTub"),
        ("mean_intensity_actin", cell_row, "Intensity_MeanIntensity_CorrActin"),
        ("neighbors_10", cell_row, "Neighbors_NumberOfNeighbors_10"),
        ("nearest_neighbor_distance", cell_row, "Neighbors_FirstClosestDistance_10"),
    ]
    out: dict[str, float] = {}
    for out_key, row, source_key_name in specs:
        value = measurement_float(row, source_key_name)
        if value is not None:
            out[out_key] = float(value)
    return out


def build_instances(
    source: SourceRecord,
    instance_map: np.ndarray,
    rgb: np.ndarray,
    category: str,
    state: str,
    measurements: dict[str, dict[MeasurementKey, MeasurementRow]],
) -> list[dict[str, Any]]:
    instance_ids = [int(x) for x in np.unique(instance_map) if int(x) != 0]
    areas = [int((instance_map == iid).sum()) for iid in instance_ids]
    gray = rgb.astype(np.float32).mean(axis=-1) / 255.0
    composite_means = [
        float(gray[instance_map == iid].mean()) if areas[idx] else 0.0
        for idx, iid in enumerate(instance_ids)
    ]

    cell_rows = {iid: measurements["cell"].get(source_key(source, iid)) for iid in instance_ids}
    dapi_values = [measurement_float(cell_rows[iid], "Intensity_MeanIntensity_CorrDAPI") for iid in instance_ids]
    tubulin_values = [measurement_float(cell_rows[iid], "Intensity_MeanIntensity_CorrTub") for iid in instance_ids]
    actin_values = [measurement_float(cell_rows[iid], "Intensity_MeanIntensity_CorrActin") for iid in instance_ids]
    neighbor_values = [measurement_float(cell_rows[iid], "Neighbors_NumberOfNeighbors_10") for iid in instance_ids]
    dapi_pool = [x for x in dapi_values if x is not None]
    tubulin_pool = [x for x in tubulin_values if x is not None]
    actin_pool = [x for x in actin_values if x is not None]
    neighbor_pool = [x for x in neighbor_values if x is not None]

    instances: list[dict[str, Any]] = []
    for idx, iid in enumerate(instance_ids):
        mask = instance_map == iid
        bbox = [int(x) for x in bbox_xywh_from_mask(mask)]
        morph = morphology_from_bbox(bbox)
        area_bucket = area_level(areas[idx], areas)
        intensity_bucket = intensity_level(composite_means[idx], composite_means)
        dapi_bucket = intensity_level(dapi_values[idx], dapi_pool)
        tubulin_bucket = intensity_level(tubulin_values[idx], tubulin_pool)
        actin_bucket = intensity_level(actin_values[idx], actin_pool)
        neighbor_bucket = bucket_value(neighbor_values[idx], neighbor_pool)
        key = source_key(source, iid)
        measurements_out = compact_measurements(
            measurements["cell"].get(key),
            measurements["nuclei"].get(key),
            measurements["cytoplasm"].get(key),
        )
        attrs = {
            "morphology": morph,
            "area": area_bucket,
            "intensity": intensity_bucket,
            "dapi_intensity": dapi_bucket,
            "tubulin_intensity": tubulin_bucket,
            "actin_intensity": actin_bucket,
            "neighbor_density": neighbor_bucket,
            "text_source": "template_with_cellprofiler_measurements" if measurements_out else "template",
        }
        instances.append(
            {
                "instance_id": iid,
                "category": category,
                "state": state,
                "instance_text": (
                    f"an {category} cell under {state} with {morph} morphology, "
                    f"{area_bucket} area, {dapi_bucket} DAPI intensity, "
                    f"{tubulin_bucket} tubulin intensity, and {actin_bucket} actin intensity"
                ),
                "bbox_xywh": bbox,
                "attributes": attrs,
                "measurements": measurements_out,
            }
        )
    return instances


def split_sources(
    sources: list[SourceRecord],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[SourceRecord]]:
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


def category_from_row(row: dict[str, str], fallback: str) -> str:
    return clean_text(row.get("Metadata_CellLine")) or fallback


def state_from_row(row: dict[str, str]) -> str:
    compound = clean_text(row.get("Metadata_Image_Metadata_Compound")) or clean_text(row.get("Metadata_Compound"))
    concentration = clean_text(row.get("Metadata_Image_Metadata_Concentration")) or clean_text(row.get("Metadata_Concentration"))
    if compound:
        conc_value = to_float(concentration)
        if compound.upper() == "DMSO" and (conc_value is None or abs(conc_value) < 1e-12):
            return "DMSO vehicle control"
        if concentration:
            return f"{compound} treatment at concentration {concentration}"
        return f"{compound} treatment"
    return "unknown compound treatment"


def global_text(row: dict[str, str], source: SourceRecord, category: str, state: str) -> str:
    return "; ".join(
        [
            f"fluorescence microscopy image of {category} cells from BBBC021",
            state,
            "three-channel Cell Painting composite",
            "channels: DAPI nucleus, beta-tubulin microtubules, F-actin cytoskeleton",
            f"plate {source.plate} well {source.well} site {source.site}",
        ]
    )


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
    label_rows = [
        {
            "id": i,
            "category": category,
            "state": state,
            "name": f"{category}::{state}",
        }
        for i, (category, state) in enumerate(sorted(category_states.keys()))
    ]
    categories_json = {
        "source_dataset": "BBBC021",
        "mask_kind": args.mask_kind,
        "image_composite": {
            "red_channel": "CorrActin",
            "green_channel": "CorrTub",
            "blue_channel": "CorrDAPI",
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
            "dapi_intensity": ["low", "medium", "high"],
            "tubulin_intensity": ["low", "medium", "high"],
            "actin_intensity": ["low", "medium", "high"],
            "neighbor_density": ["low", "medium", "high"],
            "text_source": ["template", "template_with_cellprofiler_measurements"],
        },
    }
    (ann_dir / "categories.json").write_text(json.dumps(categories_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    policy = RelationPolicy(use_delaunay=bool(args.use_delaunay))
    (ann_dir / "relation_policy.json").write_text(json.dumps(policy.__dict__, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = Path(args.source).resolve()
    out_dir = Path(args.out).resolve()
    progress(f"start source={source_dir} out={out_dir}", quiet=args.quiet)
    sources = discover_source_records(
        source_dir,
        args.mask_kind,
        progress_every=args.progress_every,
        quiet=args.quiet,
    )
    if args.max_images > 0:
        progress(f"applying --max_images {args.max_images}: {len(sources)} -> {min(len(sources), args.max_images)} records", quiet=args.quiet)
        sources = sources[: args.max_images]
    if not sources:
        raise SystemExit(f"No complete BBBC021 image/mask records found under {source_dir}")
    progress(f"usable source records: {len(sources)}", quiet=args.quiet)

    split_map = split_sources(
        sources,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    progress(
        "split sizes: "
        + ", ".join(f"{split}={len(rows)}" for split, rows in split_map.items()),
        quiet=args.quiet,
    )

    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "instance_maps" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "annotations").mkdir(parents=True, exist_ok=True)

    categories: Counter[str] = Counter()
    states: Counter[str] = Counter()
    category_states: Counter[tuple[str, str]] = Counter()
    split_counts: dict[str, int] = {}
    instance_counts: dict[str, list[int]] = {}
    measurement_rows_loaded: Counter[str] = Counter()
    processed_images = 0
    written_images = 0
    skipped_empty = 0
    total_images = sum(len(rows) for rows in split_map.values())

    for split, split_sources_list in split_map.items():
        rows: list[dict[str, Any]] = []
        instance_counts[split] = []
        groups = group_sources_by_measurement_dir(split_sources_list)
        progress(f"processing split {split}: {len(split_sources_list)} images in {len(groups)} measurement groups", quiet=args.quiet)
        for group_idx, group_sources in enumerate(groups, start=1):
            group_dir = group_sources[0].image_csv.parent if group_sources else source_dir
            progress(
                f"{split} group {group_idx}/{len(groups)}: loading measurements from "
                f"{display_path(group_dir, source_dir)} for {len(group_sources)} images",
                quiet=args.quiet,
            )
            measurements = load_group_measurements(group_sources)
            for kind, kind_rows in measurements.items():
                measurement_rows_loaded[kind] += len(kind_rows)
            progress(
                f"{split} group {group_idx}/{len(groups)}: loaded measurement rows "
                + ", ".join(f"{kind}={len(measurements[kind])}" for kind in sorted(MEASUREMENT_FILES)),
                quiet=args.quiet,
            )
            for source in group_sources:
                processed_images += 1
                category = category_from_row(source.row, args.cell_line)
                state = state_from_row(source.row)
                rgb = bbbc021_to_rgb(
                    source,
                    p_low=args.percentile_low,
                    p_high=args.percentile_high,
                    image_size=args.image_size,
                )
                instance_map = load_instance_map(
                    source.mask_path(args.mask_kind),
                    image_size=args.image_size,
                    min_area=args.min_area,
                    max_area=args.max_area,
                )
                instances = build_instances(source, instance_map, rgb, category, state, measurements)
                if not instances:
                    skipped_empty += 1
                    if should_report(processed_images, args.progress_every) or processed_images == total_images:
                        progress(
                            f"processed {processed_images}/{total_images}; written={written_images}; "
                            f"skipped_empty={skipped_empty}; last={source.week}/{source.plate}/{source.well}/s{source.site}",
                            quiet=args.quiet,
                        )
                    continue

                stem = f"bbbc021_{source.week}_{source.plate}_{source.well}_s{source.site}"
                image_out = out_dir / "images" / split / f"{stem}.png"
                map_out = out_dir / "instance_maps" / split / f"{stem}_instance.png"
                Image.fromarray(rgb, mode="RGB").save(image_out)
                Image.fromarray(instance_map.astype(np.uint16), mode="I;16").save(map_out)

                categories[category] += 1
                states[state] += 1
                category_states[(category, state)] += 1
                rec = {
                    "image_id": stem,
                    "image_path": relative_to_root(image_out, out_dir),
                    "instance_map_path": relative_to_root(map_out, out_dir),
                    "height": args.image_size,
                    "width": args.image_size,
                    "global_text": global_text(source.row, source, category, state),
                    "metadata": {
                        "dataset": "bbbc021",
                        "week": source.week,
                        "plate": source.plate,
                        "well": source.well,
                        "site": source.site,
                        "image_number": source.image_number,
                        "cell_line": category,
                        "compound": clean_text(source.row.get("Metadata_Image_Metadata_Compound")),
                        "concentration": clean_text(source.row.get("Metadata_Image_Metadata_Concentration")),
                        "source_image_paths": {
                            "dapi": str(source.dapi_path),
                            "tubulin": str(source.tubulin_path),
                            "actin": str(source.actin_path),
                        },
                        "source_mask_path": str(source.mask_path(args.mask_kind)),
                        "source_cell_mask_path": str(source.cell_mask_path),
                        "source_nuclei_mask_path": str(source.nuclei_mask_path),
                        "source_cytoplasm_mask_path": str(source.cytoplasm_mask_path),
                        "source_overlay_path": str(source.overlay_path) if source.overlay_path is not None else "",
                        "source_image_csv": str(source.image_csv),
                        "original_height": to_int(source.row.get("Height_DAPI")),
                        "original_width": to_int(source.row.get("Width_DAPI")),
                        "count_cells": to_float(source.row.get("Count_Cells")),
                        "count_nuclei": to_float(source.row.get("Count_Nuclei")),
                        "mask_kind": args.mask_kind,
                        "text_source": "metadata_template_cellprofiler_measurements",
                    },
                    "instances": instances,
                }
                rows.append(rec)
                instance_counts[split].append(len(instances))
                written_images += 1
                if should_report(processed_images, args.progress_every) or processed_images == total_images:
                    progress(
                        f"processed {processed_images}/{total_images}; written={written_images}; "
                        f"split={split}; last={stem}; instances={len(instances)}",
                        quiet=args.quiet,
                    )
        write_jsonl(out_dir / "annotations" / f"{split}.jsonl", rows)
        split_counts[split] = len(rows)
        progress(f"finished split {split}: records={split_counts[split]}", quiet=args.quiet)

    write_metadata_files(out_dir, categories=categories, states=states, category_states=category_states, args=args)
    manifest = {
        "source": str(source_dir),
        "output": str(out_dir),
        "image_size": args.image_size,
        "mask_kind": args.mask_kind,
        "cell_line_fallback": args.cell_line,
        "splits": split_counts,
        "instance_count_summary": {
            split: summarize(values) for split, values in instance_counts.items()
        },
        "source_records": len(sources),
        "measurement_rows_loaded": {kind: int(measurement_rows_loaded[kind]) for kind in sorted(MEASUREMENT_FILES)},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    progress(f"done: wrote {written_images} records to {out_dir}", quiet=args.quiet)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Prepare BBBC021 image/mask examples as TLC-CellDiff JSONL datasets.")
    ap.add_argument("--source", type=str, default="data/bbbc021", help="BBBC021 raw/example root containing week*/illum_corrected, labels, and measurements.")
    ap.add_argument("--out", type=str, required=True, help="Output TLC dataset root.")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--mask_kind", choices=("cell", "cytoplasm", "nuclei"), default="cell")
    ap.add_argument("--cell_line", type=str, default="MCF7", help="Fallback cell line when local metadata has no explicit cell-line field.")
    ap.add_argument("--percentile_low", type=float, default=1.0)
    ap.add_argument("--percentile_high", type=float, default=99.8)
    ap.add_argument("--min_area", type=int, default=20, help="Minimum resized instance-map area.")
    ap.add_argument("--max_area", type=int, default=50000, help="Maximum resized instance-map area.")
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--test_fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_images", type=int, default=0, help="Optional cap for quick tests; 0 uses all discovered images.")
    ap.add_argument("--progress_every", type=int, default=25, help="Print progress every N processed images; use 0 to report only group/split boundaries.")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress messages and only print the final manifest.")
    ap.add_argument("--use_delaunay", action="store_true", help="Enable Delaunay edges in relation_policy.json.")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare_dataset(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
