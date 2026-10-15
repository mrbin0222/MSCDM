#!/usr/bin/env python3
"""Isolate instances."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


EXP4_ROOT = Path(__file__).resolve().parents[3]
if str(EXP4_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP4_ROOT))

from mscdms.instance_isolation import (  # noqa: E402
    IsolationConfig,
    isolate_instance,
    stable_seed,
)


SOURCES = (
    "bbbc021",
    "idr0033",
    "tissuenet_wholecell",
    "tissuenet_nuclear",
    "dynamicnuclearnet",
)


def parse_args() -> argparse.Namespace:
    exp_root = EXP4_ROOT.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_data_dir",
        type=Path,
        default=Path("data/exp4_cells_v2_core_balanced"),
    )
    parser.add_argument("--manifest_name", default="manifest_hybrid.jsonl")
    parser.add_argument(
        "--exp5_data",
        type=Path,
        default=exp_root / "exp5" / "data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--padding_ratio", type=float, default=0.30)
    parser.add_argument(
        "--background_mode",
        choices=("matched", "median", "zero"),
        default="matched",
    )
    parser.add_argument("--background_exclusion_radius", type=int, default=2)
    parser.add_argument("--background_blur_radius", type=float, default=0.65)
    parser.add_argument(
        "--background_intensity_quantile",
        type=float,
        default=0.50,
        help="仅用无实例像素中较暗的这一分位构建背景。",
    )
    parser.add_argument("--background_noise_scale", type=float, default=0.15)
    parser.add_argument("--inner_feather_radius", type=float, default=0.0)
    parser.add_argument("--min_safe_background_pixels", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    parser.add_argument("--progress_every", type=int, default=500)
    parser.add_argument(
        "--limit_per_source",
        type=int,
        default=0,
        help="仅用于冒烟；0 表示处理完整清单。",
    )
    parser.add_argument("--preview_per_source", type=int, default=4)
    parser.add_argument(
        "--allow_incomplete_manifest",
        action="store_true",
        help="仅用于诊断；正式构建不得启用。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖同名生成文件；不会删除输出目录内的其他文件。",
    )
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    temporary.replace(path)


def validate_input_manifest(
    input_root: Path,
    manifest_path: Path,
    records: list[dict],
    *,
    allow_incomplete: bool,
) -> None:
    sample_ids = [str(record.get("sample_id") or "") for record in records]
    if not sample_ids or any(not sample_id for sample_id in sample_ids):
        raise RuntimeError("输入清单为空或存在缺失 sample_id")
    duplicates = len(sample_ids) - len(set(sample_ids))
    if duplicates:
        raise RuntimeError(f"输入清单存在 {duplicates} 个重复 sample_id")

    base_path = input_root / "manifest.jsonl"
    if manifest_path.resolve() == base_path.resolve() or not base_path.is_file():
        return
    base_records = load_jsonl(base_path)
    base_ids = {str(record["sample_id"]) for record in base_records}
    input_ids = set(sample_ids)
    if input_ids != base_ids and not allow_incomplete:
        missing = sorted(base_ids - input_ids)[:5]
        extra = sorted(input_ids - base_ids)[:5]
        raise RuntimeError(
            "条件清单尚未覆盖完整基础清单；正式像素重建已拒绝启动。"
            f" base={len(base_ids)} input={len(input_ids)}"
            f" missing_examples={missing} extra_examples={extra}"
        )


def select_limit(records: list[dict], limit_per_source: int) -> list[dict]:
    if limit_per_source <= 0:
        return records
    counts: Counter = Counter()
    selected = []
    for record in records:
        source = str(record["source_name"])
        if counts[source] >= limit_per_source:
            continue
        selected.append(record)
        counts[source] += 1
    return selected


def parent_key(record: dict) -> tuple[str, str, str, str]:
    return (
        str(record["source_dataset_dir"]),
        str(record["parent_image_path"]),
        str(record["parent_instance_map_path"]),
        str(record["parent_image_id"]),
    )


def build_tasks(
    records: list[dict],
    *,
    input_root: Path,
    exp5_data: Path,
    output: Path,
    config: IsolationConfig,
    seed: int,
    overwrite: bool,
) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for index, record in enumerate(records):
        item = dict(record)
        item["_input_index"] = index
        grouped[parent_key(item)].append(item)

    return [
        {
            "records": items,
            "input_root": str(input_root),
            "exp5_data": str(exp5_data),
            "output": str(output),
            "config": asdict(config),
            "seed": seed,
            "overwrite": overwrite,
        }
        for items in grouped.values()
    ]


def process_parent(task: dict) -> list[dict]:
    records = task["records"]
    first = records[0]
    exp5_data = Path(task["exp5_data"])
    source_root = exp5_data / first["source_dataset_dir"]
    image_path = source_root / first["parent_image_path"]
    map_path = source_root / first["parent_instance_map_path"]
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    instance_map = np.asarray(Image.open(map_path))
    if instance_map.ndim == 3:
        instance_map = instance_map[..., 0]

    output_root = Path(task["output"])
    config = IsolationConfig(**task["config"])
    results = []
    for record in records:
        sample_id = str(record["sample_id"])
        split = str(record["split"])
        relative_image = Path("images") / split / f"{sample_id}.png"
        relative_mask = Path("target_masks") / split / f"{sample_id}.png"
        output_image = output_root / relative_image
        output_mask = output_root / relative_mask
        if (
            not task["overwrite"]
            and (output_image.exists() or output_mask.exists())
        ):
            raise FileExistsError(
                f"输出已存在；请换目录或显式使用 --overwrite：{output_image}"
            )

        item_seed = stable_seed(task["seed"], sample_id)
        isolated, target_mask, diagnostics = isolate_instance(
            image,
            instance_map,
            int(record["instance_id"]),
            record["bbox_xywh"],
            config=config,
            seed=item_seed,
        )
        output_image.parent.mkdir(parents=True, exist_ok=True)
        output_mask.parent.mkdir(parents=True, exist_ok=True)
        isolated.save(output_image, compress_level=1)
        target_mask.save(output_mask, compress_level=6)

        output_record = dict(record)
        output_record["image_path"] = relative_image.as_posix()
        output_record["target_mask_path"] = relative_mask.as_posix()
        output_record["pixel_preprocessing"] = {
            "protocol": "isolated_instance_matched_background_v1",
            "source_centered_dataset": Path(task["input_root"]).name,
            "source_centered_image_path": str(record["image_path"]),
            "source_parent_image_path": str(image_path),
            "source_parent_instance_map_path": str(map_path),
            "target_instance_id": int(record["instance_id"]),
            "background_mode": config.background_mode,
            "background_pixel_policy": (
                "same-parent pixels outside every labelled instance"
            ),
            "background_exclusion_radius": config.background_exclusion_radius,
            "background_blur_radius": config.background_blur_radius,
            "background_intensity_quantile": (
                config.background_intensity_quantile
            ),
            "background_noise_scale": config.background_noise_scale,
            "inner_feather_radius": config.inner_feather_radius,
            "seed": item_seed,
            **asdict(diagnostics),
        }
        results.append(output_record)
    return results


def make_preview(
    input_root: Path,
    output_root: Path,
    records: list[dict],
    per_source: int,
) -> None:
    if per_source <= 0:
        return
    chosen: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        source = str(record["source_name"])
        if len(chosen[source]) < per_source:
            chosen[source].append(record)

    preview_root = output_root / "previews"
    preview_root.mkdir(parents=True, exist_ok=True)
    font_fill = (255, 255, 255)
    for source, source_records in chosen.items():
        rows = []
        for record in source_records:
            centered_path = (
                input_root
                / record["pixel_preprocessing"]["source_centered_image_path"]
            )
            centered = Image.open(centered_path).convert("RGB")
            isolated = Image.open(output_root / record["image_path"]).convert("RGB")
            mask = Image.open(output_root / record["target_mask_path"]).convert("L")
            mask_rgb = Image.merge("RGB", (mask, mask, mask))
            row = Image.new("RGB", (512 * 3, 512 + 28), color=(12, 12, 12))
            row.paste(centered, (0, 28))
            row.paste(mask_rgb, (512, 28))
            row.paste(isolated, (1024, 28))
            draw = ImageDraw.Draw(row)
            draw.text((8, 7), "centered crop", fill=font_fill)
            draw.text((520, 7), "target mask", fill=font_fill)
            draw.text((1032, 7), "isolated composite", fill=font_fill)
            rows.append(row)
        sheet = Image.new(
            "RGB",
            (512 * 3, (512 + 28) * len(rows)),
            color=(12, 12, 12),
        )
        for index, row in enumerate(rows):
            sheet.paste(row, (0, index * (512 + 28)))
        sheet.save(preview_root / f"{source}.jpg", quality=94)


def write_derived_manifests(output: Path, records: list[dict]) -> None:
    test_records = [record for record in records if record["split"] == "test"]
    write_jsonl(output / "manifests" / "test_balanced10k.jsonl", test_records)
    for source in SOURCES:
        source_test = [
            record
            for record in test_records
            if record["source_name"] == source
        ]
        write_jsonl(
            output / "manifests" / f"test_{source}.jsonl",
            source_test,
        )
    idr_crossplate = [
        record
        for record in test_records
        if record["source_name"] == "idr0033"
        and str((record.get("metadata") or {}).get("plate") or "") == "41754"
    ]
    write_jsonl(
        output / "manifests" / "test_idr0033_crossplate2k.jsonl",
        idr_crossplate,
    )


def summarize(
    records: list[dict],
    *,
    args: argparse.Namespace,
    input_manifest: Path,
    config: IsolationConfig,
) -> dict:
    per_source_split: Counter = Counter()
    diagnostics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        source = str(record["source_name"])
        per_source_split[f"{source}/{record['split']}"] += 1
        pixel = record["pixel_preprocessing"]
        diagnostics[source]["other_count"].append(
            float(pixel["original_other_instance_count"])
        )
        diagnostics[source]["other_fraction"].append(
            float(pixel["original_other_labeled_fraction"])
        )
        diagnostics[source]["target_fraction"].append(
            float(pixel["target_fraction_native"])
        )
        diagnostics[source]["background_fraction"].append(
            float(pixel["background_pool_fraction"])
        )
        diagnostics[source]["fallback"].append(
            float(bool(pixel["background_pool_fallback"]))
        )

    source_diagnostics = {}
    for source, values in sorted(diagnostics.items()):
        other_count = values["other_count"]
        source_diagnostics[source] = {
            "records": len(other_count),
            "source_patches_with_other_instances": sum(
                value > 0 for value in other_count
            ),
            "source_patches_with_other_instances_fraction": sum(
                value > 0 for value in other_count
            )
            / len(other_count),
            "source_other_instance_count_median": median(other_count),
            "source_other_labeled_fraction_median": median(
                values["other_fraction"]
            ),
            "isolated_target_fraction_median": median(
                values["target_fraction"]
            ),
            "background_pool_fraction_median": median(
                values["background_fraction"]
            ),
            "background_pool_fallbacks": int(sum(values["fallback"])),
            "retained_other_instance_pixels": 0,
        }

    return {
        "format": "exp4_isolated_instance_dataset_v1",
        "status": "built",
        "input_data_dir": str(args.input_data_dir.resolve()),
        "input_manifest": str(input_manifest),
        "output": str(args.output.resolve()),
        "records": len(records),
        "limit_per_source": args.limit_per_source,
        "seed": args.seed,
        "isolation_config": asdict(config),
        "pixel_contract": {
            "target": "only the requested labelled instance",
            "background": (
                "deterministic texture synthesized from same-parent pixels "
                "outside all labelled instances"
            ),
            "foreign_instance_pixels_retained": 0,
            "target_mask_exported": True,
            "split_assignment_changed": False,
            "text_conditions_changed": False,
        },
        "per_source_split": dict(sorted(per_source_split.items())),
        "source_diagnostics": source_diagnostics,
    }


def copy_provenance(input_root: Path, output: Path) -> None:
    provenance = output / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    for name in (
        "dataset_summary.json",
        "audit_balanced_core_v2.json",
        "condition_repair_report.json",
        "selection_plan.jsonl",
    ):
        source = input_root / name
        if source.is_file():
            shutil.copy2(source, provenance / f"source_{name}")


def main() -> None:
    args = parse_args()
    input_root = args.input_data_dir.resolve()
    exp5_data = args.exp5_data.resolve()
    output = args.output.resolve()
    manifest_path = input_root / args.manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not exp5_data.is_dir():
        raise FileNotFoundError(exp5_data)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"输出目录非空；请换目录或显式使用 --overwrite：{output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(manifest_path)
    validate_input_manifest(
        input_root,
        manifest_path,
        records,
        allow_incomplete=args.allow_incomplete_manifest,
    )
    records = select_limit(records, args.limit_per_source)
    config = IsolationConfig(
        crop_size=args.crop_size,
        padding_ratio=args.padding_ratio,
        background_mode=args.background_mode,
        background_exclusion_radius=args.background_exclusion_radius,
        background_blur_radius=args.background_blur_radius,
        background_intensity_quantile=args.background_intensity_quantile,
        background_noise_scale=args.background_noise_scale,
        inner_feather_radius=args.inner_feather_radius,
        min_safe_background_pixels=args.min_safe_background_pixels,
    )
    tasks = build_tasks(
        records,
        input_root=input_root,
        exp5_data=exp5_data,
        output=output,
        config=config,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(
        f"records={len(records)} parent_tasks={len(tasks)} "
        f"workers={args.workers} background={args.background_mode}"
    )

    completed_records = []
    written = 0
    if args.workers <= 1:
        for task in tasks:
            result = process_parent(task)
            completed_records.extend(result)
            written += len(result)
            if args.progress_every > 0 and written % args.progress_every == 0:
                print(f"isolated {written}/{len(records)}")
    else:
        try:
            executor = ProcessPoolExecutor(max_workers=args.workers)
        except (OSError, PermissionError) as error:
            print(
                "process pool unavailable; falling back to thread workers: "
                f"{error}"
            )
            with ThreadPoolExecutor(max_workers=args.workers) as thread_executor:
                futures = [
                    thread_executor.submit(process_parent, task)
                    for task in tasks
                ]
                for future in as_completed(futures):
                    result = future.result()
                    completed_records.extend(result)
                    written += len(result)
                    if (
                        args.progress_every > 0
                        and written // args.progress_every
                        != (written - len(result)) // args.progress_every
                    ):
                        print(f"isolated {written}/{len(records)}")
        else:
            with executor:
                futures = [
                    executor.submit(process_parent, task) for task in tasks
                ]
                for future in as_completed(futures):
                    result = future.result()
                    completed_records.extend(result)
                    written += len(result)
                    if (
                        args.progress_every > 0
                        and written // args.progress_every
                        != (written - len(result)) // args.progress_every
                    ):
                        print(f"isolated {written}/{len(records)}")

    completed_records.sort(key=lambda record: int(record["_input_index"]))
    for record in completed_records:
        record.pop("_input_index", None)

    write_jsonl(output / args.manifest_name, completed_records)
    if args.manifest_name != "manifest.jsonl":
        write_jsonl(output / "manifest.jsonl", completed_records)
    write_derived_manifests(output, completed_records)
    copy_provenance(input_root, output)
    summary = summarize(
        completed_records,
        args=args,
        input_manifest=manifest_path,
        config=config,
    )
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    make_preview(
        input_root,
        output,
        completed_records,
        args.preview_per_source,
    )
    print(f"saved {output / args.manifest_name}")
    print(f"saved {output / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()
