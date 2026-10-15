#!/usr/bin/env python3
"""Crop instances."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_SOURCES = {
    "bbbc021": ("bbbc021_all_weeks_512_v3", "whole_cell"),
    "idr0033": ("idr0033_multiwell_512", "whole_cell"),
    "tissuenet_wholecell": ("tissuenet_wholecell_512", "whole_cell"),
    "tissuenet_nuclear": ("tissuenet_nuclear_512", "nucleus"),
    "dynamicnuclearnet": ("dynamicnuclearnet_nuclear_512", "nucleus"),
}


def parse_args() -> argparse.Namespace:
    exp_root = Path(__file__).resolve().parents[4]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp5_data", type=Path, default=exp_root / "exp5" / "data")
    ap.add_argument("--output", type=Path, default=Path("data/exp4_cells_v1"))
    ap.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help=f"逗号分隔；可选: {','.join(DEFAULT_SOURCES)}",
    )
    ap.add_argument(
        "--source_spec",
        action="append",
        default=[],
        metavar="NAME=DATASET_DIR=OBJECT_TYPE",
        help=(
            "Add or override a TLC source definition; repeatable. "
            "OBJECT_TYPE must be whole_cell or nucleus."
        ),
    )
    ap.add_argument("--crop_size", type=int, default=512)
    ap.add_argument("--padding_ratio", type=float, default=0.30)
    ap.add_argument("--min_area", type=int, default=32)
    ap.add_argument("--min_side", type=int, default=6)
    ap.add_argument("--max_instances_per_image", type=int, default=16)
    ap.add_argument(
        "--max_samples_per_source_split",
        type=int,
        default=-1,
        help="若 >=0，则覆盖下面三个 split 上限；0 表示不限制",
    )
    ap.add_argument("--max_train_samples_per_source", type=int, default=5000)
    ap.add_argument("--max_val_samples_per_source", type=int, default=500)
    ap.add_argument("--max_test_samples_per_source", type=int, default=500)
    ap.add_argument("--max_images_per_source_split", type=int, default=0, help="调试用；0 不限制")
    ap.add_argument("--include_boundary_instances", action="store_true")
    ap.add_argument("--mask_background", action="store_true", help="裁剪后把目标实例之外像素置零")
    ap.add_argument("--seed", type=int, default=20260723)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def iter_records(source_root: Path, split: str):
    path = source_root / "annotations" / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def square_bounds(
    bbox_xywh: list[int],
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x, y, w, h = (int(v) for v in bbox_xywh)
    side = max(w, h) * (1.0 + 2.0 * padding_ratio)
    side = max(1, int(round(side)))
    cx = x + w / 2.0
    cy = y + h / 2.0
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x1 = x0 + side
    y1 = y0 + side
    return max(0, x0), max(0, y0), min(image_width, x1), min(image_height, y1)


def crop_instance(
    image: np.ndarray,
    instance_map: np.ndarray,
    instance_id: int,
    bbox_xywh: list[int],
    *,
    crop_size: int,
    padding_ratio: float,
    mask_background: bool,
) -> Image.Image:
    h, w = instance_map.shape
    x0, y0, x1, y1 = square_bounds(bbox_xywh, w, h, padding_ratio)
    patch = image[y0:y1, x0:x1].copy()
    if mask_background:
        mask = instance_map[y0:y1, x0:x1] == instance_id
        patch[~mask] = 0
    ph, pw = patch.shape[:2]
    side = max(ph, pw)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    oy = (side - ph) // 2
    ox = (side - pw) // 2
    canvas[oy : oy + ph, ox : ox + pw] = patch
    return Image.fromarray(canvas, mode="RGB").resize(
        (crop_size, crop_size),
        Image.Resampling.BICUBIC,
    )


def is_boundary_instance(bbox_xywh: list[int], width: int, height: int) -> bool:
    x, y, w, h = (int(v) for v in bbox_xywh)
    return x <= 0 or y <= 0 or x + w >= width or y + h >= height


def template_conditions(record: dict, instance: dict, object_type: str) -> tuple[str, str, str]:
    attrs = instance.get("attributes") or {}
    category = str(instance.get("category") or object_type).replace("_", " ")
    morphology = str(attrs.get("morphology") or "").strip()
    area = str(attrs.get("area") or "").strip()
    subject_bits = [category]
    if object_type == "nucleus" and "nucleus" not in category.lower():
        subject_bits.append("nucleus")
    elif object_type == "whole_cell" and "cell" not in category.lower():
        subject_bits.append("cell")
    if morphology:
        subject_bits.append(f"with {morphology} morphology")
    if area:
        subject_bits.append(f"and {area} area")
    subject = " ".join(subject_bits)

    state = str(instance.get("state") or "").strip()
    if not state:
        meta = record.get("metadata") or {}
        state = str(meta.get("fine_state") or meta.get("coarse_state") or "unspecified cellular state")

    global_text = str(record.get("global_text") or "").strip()
    background = global_text if global_text else "microscopy image"
    return subject, state, background


def safe_rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    source_defs = dict(DEFAULT_SOURCES)
    for raw_spec in args.source_spec:
        parts = raw_spec.split("=")
        if len(parts) != 3 or not all(part.strip() for part in parts):
            raise ValueError(
                "--source_spec 格式必须是 NAME=DATASET_DIR=OBJECT_TYPE"
            )
        name, dataset_dir, object_type = (part.strip() for part in parts)
        if object_type not in {"whole_cell", "nucleus"}:
            raise ValueError(
                f"--source_spec {name}: object type 必须是 whole_cell/nucleus"
            )
        source_defs[name] = (dataset_dir, object_type)
    selected_sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = sorted(set(selected_sources) - set(source_defs))
    if unknown:
        raise ValueError(f"未知来源: {unknown}")

    out = args.output.resolve()
    manifest_path = out / "manifest.jsonl"
    summary_path = out / "dataset_summary.json"
    if (manifest_path.exists() or summary_path.exists()) and not args.overwrite:
        raise FileExistsError(f"{out} 已含输出；如需重建请显式加 --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter()
    per_group: dict[str, int] = defaultdict(int)
    records_out: list[dict] = []
    split_caps = {
        "train": args.max_train_samples_per_source,
        "val": args.max_val_samples_per_source,
        "test": args.max_test_samples_per_source,
    }

    for source_name in selected_sources:
        source_dir_name, object_type = source_defs[source_name]
        source_root = args.exp5_data.resolve() / source_dir_name
        for split in ("train", "val", "test"):
            group_key = f"{source_name}/{split}"
            group_cap = (
                args.max_samples_per_source_split
                if args.max_samples_per_source_split >= 0
                else split_caps[split]
            )
            for parent_index, parent in enumerate(iter_records(source_root, split)):
                if (
                    args.max_images_per_source_split > 0
                    and parent_index >= args.max_images_per_source_split
                ):
                    break
                if (
                    group_cap > 0
                    and per_group[group_key] >= group_cap
                ):
                    break
                image_path = source_root / parent["image_path"]
                map_path = source_root / parent["instance_map_path"]
                image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
                instance_map = np.asarray(Image.open(map_path))
                if instance_map.ndim == 3:
                    instance_map = instance_map[..., 0]
                height, width = instance_map.shape
                instances = list(parent.get("instances") or [])
                rng.shuffle(instances)
                kept_for_image = 0
                for inst in instances:
                    if kept_for_image >= args.max_instances_per_image:
                        break
                    if (
                        group_cap > 0
                        and per_group[group_key] >= group_cap
                    ):
                        break
                    bbox = inst.get("bbox_xywh")
                    if not bbox or len(bbox) != 4:
                        stats["skip_missing_bbox"] += 1
                        continue
                    _, _, bw, bh = (int(v) for v in bbox)
                    if bw * bh < args.min_area or min(bw, bh) < args.min_side:
                        stats["skip_too_small"] += 1
                        continue
                    if not args.include_boundary_instances and is_boundary_instance(bbox, width, height):
                        stats["skip_boundary"] += 1
                        continue
                    instance_id = int(inst["instance_id"])
                    if not np.any(instance_map == instance_id):
                        stats["skip_missing_label"] += 1
                        continue

                    parent_id = str(parent["image_id"])
                    sample_id = f"{source_name}__{parent_id}__i{instance_id:05d}"
                    out_image = out / "images" / split / f"{sample_id}.png"
                    crop = crop_instance(
                        image,
                        instance_map,
                        instance_id,
                        bbox,
                        crop_size=args.crop_size,
                        padding_ratio=args.padding_ratio,
                        mask_background=args.mask_background,
                    )
                    # 训练数据优先构建吞吐；PNG 无损，低压缩级别只影响文件大小。
                    crop.save(out_image, compress_level=1)
                    subject, state, background = template_conditions(parent, inst, object_type)
                    records_out.append(
                        {
                            "sample_id": sample_id,
                            "split": split,
                            "image_path": safe_rel(out_image, out),
                            "subject": subject,
                            "state": state,
                            "background": background,
                            "raw_instance_text": str(inst.get("instance_text") or ""),
                            "raw_global_text": str(parent.get("global_text") or ""),
                            "source_name": source_name,
                            "source_dataset_dir": source_dir_name,
                            "object_type": object_type,
                            "parent_image_id": parent_id,
                            "parent_image_path": str(parent["image_path"]),
                            "parent_instance_map_path": str(parent["instance_map_path"]),
                            "instance_id": instance_id,
                            "bbox_xywh": [int(v) for v in bbox],
                            "attributes": inst.get("attributes") or {},
                            "metadata": parent.get("metadata") or {},
                            "condition_source": "deterministic_template",
                        }
                    )
                    kept_for_image += 1
                    per_group[group_key] += 1
                    stats["written"] += 1

    records_out.sort(key=lambda rec: (rec["split"], rec["source_name"], rec["sample_id"]))
    with manifest_path.open("w", encoding="utf-8") as fh:
        for rec in records_out:
            fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")

    split_counts = Counter(rec["split"] for rec in records_out)
    source_counts = Counter(rec["source_name"] for rec in records_out)
    object_counts = Counter(rec["object_type"] for rec in records_out)
    summary = {
        "format": "exp4_instance_crops_v1",
        "output": str(out),
        "seed": args.seed,
        "crop_size": args.crop_size,
        "padding_ratio": args.padding_ratio,
        "mask_background": args.mask_background,
        "sources": selected_sources,
        "split_caps_per_source": split_caps,
        "split_counts": dict(sorted(split_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "object_type_counts": dict(sorted(object_counts.items())),
        "skip_counts": dict(sorted(stats.items())),
        "per_source_split": dict(sorted(per_group.items())),
        "manifest": str(manifest_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
