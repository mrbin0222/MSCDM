from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from mscdms.workflows.ctc_archive_utils import CTC_DATASETS, save_label_tiff, write_jsonl


GT_3D_RE = re.compile(r"man_seg_(\d+)_(\d+)\.tif$", re.IGNORECASE)
GT_2D_RE = re.compile(r"man_seg(\d+)\.tif$", re.IGNORECASE)
ST_RE = re.compile(r"mask(\d+)\.tif$", re.IGNORECASE)
RAW_RE = re.compile(r"t(\d+)\.tif$", re.IGNORECASE)


def parse_datasets(value: str) -> list[str]:
    datasets = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in datasets if item not in CTC_DATASETS]
    if unknown:
        raise ValueError(f"Unknown CTC datasets: {unknown}")
    return datasets


def read_tiff_from_zip(archive: zipfile.ZipFile, name: str) -> np.ndarray:
    return np.asarray(tifffile.imread(io.BytesIO(archive.read(name))))


def annotation_entries(
    archive: zipfile.ZipFile,
    dataset: str,
    sequence: str,
    source: str,
) -> tuple[str, list[str]]:
    names = archive.namelist()
    sources = ("st", "gt") if source == "auto" else (source,)
    # Fluo-N2DH-SIM+ is computer-generated; CTC provides exact GT rather than ST.
    if source == "st" and dataset == "Fluo-N2DH-SIM+":
        sources = ("st", "gt")
    for candidate in sources:
        marker = f"/{sequence}_{candidate.upper()}/SEG/"
        entries = [name for name in names if marker in name and name.lower().endswith(('.tif', '.tiff'))]
        if entries:
            return candidate, sorted(entries)
    raise FileNotFoundError(
        f"{dataset} sequence {sequence} has no {source.upper()} SEG masks in the local archive. "
        "The current experiment contract requires official ST masks; download the CTC training archive that includes *_ST/SEG."
    )


def raw_lookup(archive: zipfile.ZipFile, dataset: str, sequence: str) -> dict[int, str]:
    prefix = f"{dataset}/{sequence}/"
    lookup: dict[int, str] = {}
    for name in archive.namelist():
        if not name.startswith(prefix):
            continue
        match = RAW_RE.search(Path(name).name)
        if match:
            lookup[int(match.group(1))] = name
    return lookup


def parse_annotation_index(name: str, source: str) -> tuple[int, int | None]:
    basename = Path(name).name
    if source == "st":
        match = ST_RE.search(basename)
        if match:
            return int(match.group(1)), None
        # Older CTC ST archives use the same man_seg naming as GT.
        match3d = GT_3D_RE.search(basename)
        if match3d:
            return int(match3d.group(1)), int(match3d.group(2))
        match2d = GT_2D_RE.search(basename)
        if match2d:
            return int(match2d.group(1)), None
        raise ValueError(f"Unrecognized ST mask name: {name}")
    match3d = GT_3D_RE.search(basename)
    if match3d:
        return int(match3d.group(1)), int(match3d.group(2))
    match2d = GT_2D_RE.search(basename)
    if match2d:
        return int(match2d.group(1)), None
    raise ValueError(f"Unrecognized GT mask name: {name}")


def split_annotation(
    raw: np.ndarray,
    mask: np.ndarray,
    requested_z: int | None,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    if raw.ndim == 2 and mask.ndim == 2:
        return [(-1, raw, mask)]
    if raw.ndim == 3 and mask.ndim == 3:
        if raw.shape != mask.shape:
            raise ValueError(f"3D raw/mask shape mismatch: {raw.shape} vs {mask.shape}")
        return [(z, raw[z], mask[z]) for z in range(raw.shape[0])]
    if raw.ndim == 3 and mask.ndim == 2 and requested_z is not None:
        if requested_z >= raw.shape[0]:
            raise IndexError(f"Requested z={requested_z}, raw stack shape={raw.shape}")
        return [(requested_z, raw[requested_z], mask)]
    if raw.ndim == 3 and mask.ndim == 2 and raw.shape[-2:] == mask.shape:
        raise ValueError("A 2D mask for a 3D raw frame requires a z index in the CTC annotation filename.")
    raise ValueError(f"Unsupported raw/mask ranks or shapes: {raw.shape} vs {mask.shape}")


def prepare_dataset(
    archive_path: Path,
    out_root: Path,
    dataset: str,
    annotation_source: str,
    limit_per_sequence: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    with zipfile.ZipFile(archive_path) as archive:
        for sequence in ("01", "02"):
            source_used, entries = annotation_entries(archive, dataset, sequence, annotation_source)
            if limit_per_sequence > 0:
                entries = entries[:limit_per_sequence]
            raw_by_time = raw_lookup(archive, dataset, sequence)
            source_counts[f"{sequence}_{source_used}"] += len(entries)
            for annotation_name in entries:
                time_index, requested_z = parse_annotation_index(annotation_name, source_used)
                if time_index not in raw_by_time:
                    raise FileNotFoundError(f"No raw frame t{time_index:03d}.tif for {annotation_name}")
                raw = read_tiff_from_zip(archive, raw_by_time[time_index])
                mask = read_tiff_from_zip(archive, annotation_name)
                for z_index, image_plane, mask_plane in split_annotation(raw, mask, requested_z):
                    z_suffix = f"_z{z_index:03d}" if z_index >= 0 else ""
                    sample_id = f"{dataset}_{sequence}_t{time_index:04d}{z_suffix}"
                    frame_id = f"{dataset}/{sequence}/t{time_index:04d}"
                    image_rel = Path("images") / dataset / sequence / f"{sample_id}.tif"
                    mask_rel = Path("masks") / dataset / sequence / f"{sample_id}.tif"
                    image_path = out_root / image_rel
                    mask_path = out_root / mask_rel
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(image_path, np.asarray(image_plane), compression="zlib")
                    save_label_tiff(mask_path, np.asarray(mask_plane))
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "frame_id": frame_id,
                            "dataset": dataset,
                            "sequence": sequence,
                            "time_index": time_index,
                            "z_index": z_index,
                            "annotation_source": source_used,
                            "image_path": str(image_rel),
                            "mask_path": str(mask_rel),
                            "height": int(image_plane.shape[-2]),
                            "width": int(image_plane.shape[-1]),
                            "num_instances_on_plane": int(np.count_nonzero(np.unique(mask_plane) > 0)),
                            "source_archive": str(archive_path),
                            "source_annotation": annotation_name,
                        }
                    )
    summary = {
        "dataset": dataset,
        "archive": str(archive_path),
        "num_slice_records": len(rows),
        "num_frames": len({row["frame_id"] for row in rows}),
        "sequence_slice_records": dict(Counter(row["sequence"] for row in rows)),
        "annotation_entries": dict(source_counts),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official CTC ST masks for 01<->02 model comparison.")
    parser.add_argument("--raw_dir", type=Path, default=Path("exp5/rawdata/CTC"))
    parser.add_argument("--out_dir", type=Path, default=Path("exp5/data/ctc_segmentation_st_v1"))
    parser.add_argument("--datasets", default=",".join(CTC_DATASETS))
    parser.add_argument("--annotation_source", choices=("st", "gt", "auto"), default="st")
    parser.add_argument("--limit_per_sequence", type=int, default=0, help="Smoke-only annotation-file limit.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_dir = args.raw_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for dataset in parse_datasets(args.datasets):
        archive_path = raw_dir / f"{dataset}.zip"
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        rows, summary = prepare_dataset(
            archive_path,
            out_dir,
            dataset,
            args.annotation_source,
            args.limit_per_sequence,
        )
        all_rows.extend(rows)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))
    write_jsonl(out_dir / "manifest.jsonl", all_rows)
    payload = {
        "protocol": "CTC official ST sequence-swap comparison",
        "annotation_source_requested": args.annotation_source,
        "datasets": summaries,
        "num_slice_records": len(all_rows),
        "num_frames": len({row["frame_id"] for row in all_rows}),
    }
    (out_dir / "dataset_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
