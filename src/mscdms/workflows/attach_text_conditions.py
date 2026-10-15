#!/usr/bin/env python3
"""Attach text conditions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


SOURCES = (
    "bbbc021",
    "idr0033",
    "tissuenet_wholecell",
    "tissuenet_nuclear",
    "dynamicnuclearnet",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isolated_data_dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--condition_manifest",
        type=Path,
        default=Path(
            "data/exp4_cells_v2_core_balanced/manifest_hybrid.jsonl"
        ),
    )
    parser.add_argument("--output_name", default="manifest_hybrid.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"JSONL 解析失败：{path}:{line_number}") from error
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    temporary.replace(path)


def write_derived_manifests(output: Path, records: list[dict]) -> None:
    test_records = [record for record in records if record["split"] == "test"]
    manifests = output / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    write_jsonl(manifests / "test_balanced10k.jsonl", test_records)
    for source in SOURCES:
        write_jsonl(
            manifests / f"test_{source}.jsonl",
            [
                record
                for record in test_records
                if record["source_name"] == source
            ],
        )
    write_jsonl(
        manifests / "test_idr0033_crossplate2k.jsonl",
        [
            record
            for record in test_records
            if record["source_name"] == "idr0033"
            and str((record.get("metadata") or {}).get("plate") or "")
            == "41754"
        ],
    )


def main() -> None:
    args = parse_args()
    root = args.isolated_data_dir.resolve()
    isolated_manifest = root / "manifest.jsonl"
    condition_manifest = args.condition_manifest.resolve()
    output = root / args.output_name
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"输出已存在；请检查后显式使用 --overwrite：{output}"
        )

    isolated = load_jsonl(isolated_manifest)
    conditions = load_jsonl(condition_manifest)
    isolated_by_id = {str(row["sample_id"]): row for row in isolated}
    condition_ids = [str(row["sample_id"]) for row in conditions]
    if len(condition_ids) != len(set(condition_ids)):
        raise RuntimeError("条件清单存在重复 sample_id")
    isolated_ids = set(isolated_by_id)
    if set(condition_ids) != isolated_ids:
        missing = sorted(isolated_ids - set(condition_ids))[:10]
        extra = sorted(set(condition_ids) - isolated_ids)[:10]
        raise RuntimeError(
            "条件清单尚未完整对应 isolated 数据，已拒绝合并。"
            f" isolated={len(isolated_ids)} condition={len(condition_ids)}"
            f" missing={missing} extra={extra}"
        )

    merged = []
    changed_pixels = []
    for condition in conditions:
        sample_id = str(condition["sample_id"])
        pixel_record = isolated_by_id[sample_id]
        output_record = dict(condition)
        output_record["image_path"] = pixel_record["image_path"]
        output_record["target_mask_path"] = pixel_record["target_mask_path"]
        output_record["pixel_preprocessing"] = pixel_record[
            "pixel_preprocessing"
        ]
        if (
            output_record["image_path"] != pixel_record["image_path"]
            or output_record["target_mask_path"]
            != pixel_record["target_mask_path"]
        ):
            changed_pixels.append(sample_id)
        merged.append(output_record)
    if changed_pixels:
        raise RuntimeError(f"像素路径合并异常：{changed_pixels[:10]}")

    write_jsonl(output, merged)
    write_derived_manifests(root, merged)

    summary_path = root / "dataset_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["condition_manifest"] = {
        "status": "synced",
        "source": str(condition_manifest),
        "output": str(output),
        "records": len(merged),
        "providers": dict(
            sorted(
                Counter(
                    str(row.get("condition_source") or "unknown")
                    for row in merged
                ).items()
            )
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved {output}")
    print(f"records={len(merged)}")


if __name__ == "__main__":
    main()
