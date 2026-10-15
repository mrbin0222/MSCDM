#!/usr/bin/env python3
"""Generate text dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist

from mscdms import MSCDM
from mscdms.checkpoint import config_from_checkpoint, load_checkpoint
from mscdms.distributed import cleanup_distributed, setup_distributed
from mscdms.text_encoder import MultiScaleTextEncoder

NUCLEUS_GATED_MODES = {
    "subject_state_low_nucleus",
    "subject_state_low_nucleus_adapter",
    "subject_state_low_nucleus_multiscale_adapter",
    "subject_state_low_nucleus_prompt_adapters",
    "hierarchical_nucleus_distill_prompt_adapters",
    "hierarchical_nucleus_distill_output_adapters",
    "subject_state_low_nucleus_temporal_multiscale_adapter",
}
FACTOR_DELTA_ADAPTER_MODE = "hierarchical_factor_delta_adapters"
FACTOR_ATTENTION_DELTA_ADAPTER_MODE = (
    "hierarchical_factor_attention_delta_adapters"
)
FACTOR_EDIT_MODES = {
    FACTOR_DELTA_ADAPTER_MODE,
    FACTOR_ATTENTION_DELTA_ADAPTER_MODE,
}
SPATIAL_SUBJECT_INPUT_ADAPTER_MODE = (
    "hierarchical_spatial_subject_input_adapter"
)
MORPHOLOGY_LATENT_BANK_MODE = "hierarchical_morphology_latent_bank"
MORPHOLOGY_FEATURE_AFFINE_MODE = (
    "hierarchical_morphology_feature_affine"
)
MORPHOLOGY_SPATIAL_AFFINE_MODE = (
    "hierarchical_morphology_spatial_affine"
)
STRUCTURED_SUBJECT_MODES = {
    SPATIAL_SUBJECT_INPUT_ADAPTER_MODE,
    MORPHOLOGY_LATENT_BANK_MODE,
    MORPHOLOGY_FEATURE_AFFINE_MODE,
    MORPHOLOGY_SPATIAL_AFFINE_MODE,
}
BACKGROUND_PROMPTS = {
    "bbbc021": (
        "three-channel Cell Painting fluorescence background with DAPI, "
        "beta-tubulin, and F-actin channels"
    ),
    "idr0033": (
        "five-channel Cell Painting fluorescence background with nuclear, "
        "organelle, membrane, actin, and mitochondrial channels"
    ),
    "tissuenet_wholecell": (
        "two-channel TissueNet tissue background with nuclear and whole-cell "
        "channels"
    ),
    "tissuenet_nuclear": (
        "single-channel TissueNet nuclear microscopy background"
    ),
    "dynamicnuclearnet": (
        "single-channel DynamicNuclearNet nuclear fluorescence background"
    ),
}
AREA_SUFFIX_RE = re.compile(
    r"\s+and\s+(?:large|medium|small)\s+area$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--manifest_name", default="manifest_hybrid.jsonl")
    ap.add_argument("--split", choices=("train", "val", "test"), default="test")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--sample_ids_file",
        type=Path,
        default=None,
        help="可选：每行一个 sample_id，仅生成这些样本",
    )
    ap.add_argument(
        "--limit_per_source",
        type=int,
        default=0,
        help="每个 source_name 确定性随机抽取 N 条；与 --limit 互斥",
    )
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260723)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg_scale", type=float, default=None)
    ap.add_argument(
        "--cfg_scale_subject",
        type=float,
        default=None,
        help="Low/subject branch CFG; defaults to --cfg_scale.",
    )
    ap.add_argument(
        "--cfg_scale_state",
        type=float,
        default=None,
        help="Mid/state branch CFG; defaults to --cfg_scale.",
    )
    ap.add_argument(
        "--cfg_scale_background",
        type=float,
        default=None,
        help="High/background branch CFG; defaults to --cfg_scale.",
    )
    ap.add_argument(
        "--target_factor_cfg_scale",
        type=float,
        default=None,
        help=(
            "Per-record intervention for control-edit manifests: apply this "
            "scale only to the record's target_attribute branch "
            "(identity->subject, state->state, background->background), "
            "while other branches retain their configured scales."
        ),
    )
    ap.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument(
        "--distributed_backend",
        choices=("nccl", "gloo"),
        default="nccl",
        help=(
            "torchrun 多进程控制面；各 rank 独立生成且只需 barrier 时可用 gloo，"
            "避免不必要的 NCCL communicator。"
        ),
    )
    ap.add_argument("--clip_model_dir", default="models/clip-vit-base-patch32")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use_personalization", action="store_true")
    ap.add_argument(
        "--concept_token",
        default=None,
        help="覆盖 checkpoint 中的 concept_token；仅 --use_personalization 时生效",
    )
    ap.add_argument(
        "--no_checkpoint_concept_token",
        action="store_true",
        help="启用个体化向量，但不向主体文本自动添加 checkpoint concept_token",
    )
    return ap.parse_args()


def load_source_manifest(root: Path, name: str, split: str) -> list[dict]:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    records = [record for record in records if record.get("split") == split]
    records.sort(key=lambda record: record["sample_id"])
    return records


def append_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        fh.flush()


def select_per_source(records: list[dict], limit: int, seed: int) -> list[dict]:
    """按 source_name 做可复现的均衡抽样，避免排序截断偏向单一来源。"""
    if limit <= 0:
        return records
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record.get("source_name") or "unknown"), []).append(record)
    selected = []
    for source_name in sorted(grouped):
        candidates = sorted(grouped[source_name], key=lambda record: record["sample_id"])
        rng = random.Random(f"{seed}:{source_name}")
        count = min(limit, len(candidates))
        selected.extend(rng.sample(candidates, count))
    selected.sort(key=lambda record: record["sample_id"])
    return selected


def select_sample_ids(records: list[dict], path: Path | None) -> list[dict]:
    if path is None:
        return records
    requested = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(set(requested)) != len(requested):
        raise ValueError(f"{path} 含重复 sample_id")
    by_id = {str(record["sample_id"]): record for record in records}
    missing = [sample_id for sample_id in requested if sample_id not in by_id]
    if missing:
        raise ValueError(
            f"{path} 中有 {len(missing)} 个 sample_id 不在所选 split: "
            + ", ".join(missing[:5])
        )
    return [by_id[sample_id] for sample_id in requested]


def prepend_token(subjects: list[str], token: str) -> list[str]:
    token = token.strip()
    if not token:
        return subjects
    return [subject if token in subject else f"{token} {subject}" for subject in subjects]


def anchor_prompts_from_record(record: dict) -> tuple[str, str, str]:
    labels = record.get("original_labels")
    if not isinstance(labels, dict):
        raise ValueError(
            f"{record.get('sample_id', '<unknown>')}: factor delta edit "
            "requires original_labels"
        )
    subject = AREA_SUFFIX_RE.sub("", str(labels["identity"]).strip()).lower()
    state_group = str(labels["state_group"]).strip().lower()
    state_label = str(labels["state"]).strip()
    if state_group == "bbbc021":
        state = f"{state_label} phenotype"
    elif state_group == "dynamicnuclearnet":
        state = f"{state_label} cell-line state"
    else:
        raise ValueError(
            f"{record.get('sample_id', '<unknown>')}: unsupported formal "
            f"state group {state_group!r}"
        )
    background_label = str(labels["background"]).strip().lower()
    try:
        background = BACKGROUND_PROMPTS[background_label]
    except KeyError as exc:
        raise ValueError(
            f"{record.get('sample_id', '<unknown>')}: unsupported anchor "
            f"background {background_label!r}"
        ) from exc
    return subject, state, background


def resolve_reference_image(record: dict, data_root: Path) -> Path:
    """Resolve paired-data and control-edit reference fields uniformly."""
    raw_path = record.get("image_path") or record.get("reference_image_path")
    if not raw_path:
        raise KeyError(
            f"{record.get('sample_id', '<unknown>')}: expected image_path "
            "or reference_image_path"
        )
    reference = Path(str(raw_path))
    if not reference.is_absolute():
        reference = data_root / reference
    return reference.resolve()


def spatial_subject_masks_from_records(
    records: list[dict],
    data_root: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str | None]]:
    """Load optional train-only spatial templates for edit-time routing."""
    masks = []
    routes = []
    resolved_paths: list[str | None] = []
    for record in records:
        raw_path = str(record.get("spatial_subject_mask_path") or "").strip()
        if not raw_path:
            masks.append(torch.zeros(1, 128, 128, dtype=torch.float32))
            routes.append(False)
            resolved_paths.append(None)
            continue
        mask_path = Path(raw_path)
        if not mask_path.is_absolute():
            mask_path = data_root / mask_path
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        with Image.open(mask_path) as mask_image:
            mask_array = np.asarray(
                mask_image.convert("L").resize(
                    (128, 128),
                    Image.Resampling.NEAREST,
                ),
                dtype=np.uint8,
            )
        mask = torch.from_numpy((mask_array > 0).astype(np.float32)).unsqueeze(0)
        if not bool(mask.any()):
            raise ValueError(f"empty spatial subject mask: {mask_path}")
        masks.append(mask)
        routes.append(True)
        resolved_paths.append(str(mask_path.resolve()))
    return (
        torch.stack(masks).to(device=device, non_blocking=True),
        torch.tensor(routes, device=device, dtype=torch.bool),
        resolved_paths,
    )


def morphology_class_ids_from_records(
    records: list[dict],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve the source-neutral Stage-20 morphology coordinate."""
    class_to_id = {"round": 0, "oval": 1, "elongated": 2}
    class_ids: list[int] = []
    routes: list[bool] = []
    for record in records:
        raw_class = record.get("morphology_class")
        if raw_class is None or not str(raw_class).strip():
            # Dummy ID is never applied because the route is disabled.
            class_ids.append(0)
            routes.append(False)
            continue
        morphology = str(raw_class).strip().lower()
        if morphology not in class_to_id:
            raise ValueError(
                f"{record.get('sample_id', '<unknown>')}: invalid "
                f"morphology_class {morphology!r}"
            )
        class_ids.append(class_to_id[morphology])
        routes.append(True)
    return (
        torch.tensor(class_ids, device=device, dtype=torch.long),
        torch.tensor(routes, device=device, dtype=torch.bool),
    )


def target_factor_cfg_scales(
    records: list[dict],
    *,
    device: torch.device,
    default_scale: float,
    subject_scale: float | None,
    state_scale: float | None,
    background_scale: float | None,
    target_scale: float | None,
) -> tuple[float | torch.Tensor, float | torch.Tensor, float | torch.Tensor]:
    base = {
        "identity": (
            default_scale if subject_scale is None else subject_scale
        ),
        "state": default_scale if state_scale is None else state_scale,
        "background": (
            default_scale if background_scale is None else background_scale
        ),
    }
    if target_scale is None:
        return base["identity"], base["state"], base["background"]
    attributes = [
        str(record.get("target_attribute") or "").strip()
        for record in records
    ]
    invalid = sorted(set(attributes) - {"identity", "state", "background"})
    if invalid:
        raise ValueError(
            "--target_factor_cfg_scale requires identity/state/background "
            f"target_attribute values; found {invalid}"
        )
    per_branch = {}
    for branch in ("identity", "state", "background"):
        per_branch[branch] = torch.tensor(
            [
                target_scale if attribute == branch else base[branch]
                for attribute in attributes
            ],
            device=device,
            dtype=torch.float32,
        )
    return (
        per_branch["identity"],
        per_branch["state"],
        per_branch["background"],
    )


def cfg_value_at(
    value: float | torch.Tensor,
    index: int,
) -> float:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.item())
        return float(value[index].item())
    return float(value)


def load_completed(manifests: list[Path]) -> set[str]:
    completed: set[str] = set()
    for manifest in manifests:
        if not manifest.is_file():
            continue
        with manifest.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    completed.add(json.loads(line)["generated_id"])
    return completed


def merge_rank_manifests(
    output_root: Path,
    output_manifest: Path,
    expected_ids: set[str],
) -> None:
    """合并各 rank 的独立 manifest，按 generated_id 去重并确定性排序。"""
    records: dict[str, dict] = {}
    candidates = [output_manifest, *sorted(output_root.glob("generated_manifest.rank*.jsonl"))]
    for manifest in candidates:
        if not manifest.is_file():
            continue
        with manifest.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    record = json.loads(line)
                    if record["generated_id"] in expected_ids:
                        records[record["generated_id"]] = record
    temp = output_manifest.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8") as fh:
        for generated_id in sorted(records):
            fh.write(json.dumps(records[generated_id], ensure_ascii=False, separators=(",", ":")) + "\n")
    temp.replace(output_manifest)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    array = (tensor.detach().float().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1)
    Image.fromarray((array * 255.0).astype(np.uint8)).save(path, optimize=True)


def main() -> None:
    args = parse_args()
    if args.limit > 0 and args.limit_per_source > 0:
        raise ValueError("--limit 与 --limit_per_source 不能同时使用")

    use_ddp = "RANK" in os.environ
    if use_ddp:
        rank, local_rank, world_size = setup_distributed(
            backend=args.distributed_backend
        )
        device = torch.device("cuda", local_rank)
    else:
        rank, world_size = 0, 1
        device = torch.device(args.device)

    data_root = Path(args.data_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    images_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)
    output_manifest = output_root / "generated_manifest.jsonl"

    source = load_source_manifest(data_root, args.manifest_name, args.split)
    source = select_sample_ids(source, args.sample_ids_file)
    source = select_per_source(source, args.limit_per_source, args.seed)
    if args.limit > 0:
        source = source[: args.limit]
    jobs = [
        (record, repeat)
        for record in source
        for repeat in range(max(1, args.num_repeats))
    ]
    expected_ids = {
        f"{record['sample_id']}__r{repeat:02d}"
        for record, repeat in jobs
    }
    rank_manifest = (
        output_root / f"generated_manifest.rank{rank:03d}.jsonl"
        if use_ddp
        else output_manifest
    )
    completed = load_completed(
        [output_manifest, *sorted(output_root.glob("generated_manifest.rank*.jsonl"))]
    )
    jobs = [
        (record, repeat)
        for record, repeat in jobs
        if f"{record['sample_id']}__r{repeat:02d}" not in completed
    ]
    rank_jobs = jobs[rank::world_size]
    print(
        f"rank={rank}/{world_size} source={len(source)} "
        f"pending_global={len(jobs)} pending_rank={len(rank_jobs)} completed={len(completed)}"
    )

    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    cfg = config_from_checkpoint(checkpoint)
    cfg.text_pretrained_name = args.clip_model_dir
    cfg.text_local_files_only = True
    cfg.freeze_clip_backbone = True
    model = MSCDM(cfg).to(device)
    model.load_state_dict(checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint)
    model.eval()
    concept_token = ""
    if args.use_personalization:
        if args.no_checkpoint_concept_token:
            concept_token = ""
        elif args.concept_token is not None:
            concept_token = args.concept_token.strip()
        elif isinstance(checkpoint, dict):
            concept_token = str(checkpoint.get("concept_token") or "").strip()
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    amp_enabled = device.type == "cuda" and args.precision != "fp32"

    for start in range(0, len(rank_jobs), args.batch_size):
        batch_jobs = rank_jobs[start : start + args.batch_size]
        records = [job[0] for job in batch_jobs]
        reference_paths = [
            resolve_reference_image(record, data_root)
            for record in records
        ]
        batch_seed = args.seed + rank * 1_000_000 + start
        generator = torch.Generator(device=device).manual_seed(batch_seed)
        conditioning_subjects = prepend_token(
            [record["subject"] for record in records],
            concept_token,
        )
        subject_ids, state_ids, bg_ids = MultiScaleTextEncoder.texts_to_batch(
            conditioning_subjects,
            [record["state"] for record in records],
            [record["background"] for record in records],
            device,
            max_len=cfg.max_text_len,
            pretrained_name=cfg.text_pretrained_name,
            local_files_only=cfg.text_local_files_only,
        )
        anchor_subject_ids = None
        anchor_state_ids = None
        anchor_bg_ids = None
        intervention_factor = None
        anchor_prompts: list[tuple[str, str, str] | None] = [
            None for _ in records
        ]
        if cfg.conditioning_mode in FACTOR_EDIT_MODES:
            attributes = [
                str(record.get("target_attribute") or "").strip()
                for record in records
            ]
            if any(attributes):
                if not all(attributes):
                    raise ValueError(
                        "factor delta batch cannot mix direct and edit records"
                    )
                factor_lookup = {
                    "identity": 0,
                    "state": 1,
                    "background": 2,
                }
                invalid = sorted(set(attributes) - set(factor_lookup))
                if invalid:
                    raise ValueError(
                        f"unsupported factor delta target attributes: {invalid}"
                    )
                anchor_prompts = [
                    anchor_prompts_from_record(record) for record in records
                ]
                anchor_subject_ids, anchor_state_ids, anchor_bg_ids = (
                    MultiScaleTextEncoder.texts_to_batch(
                        [prompt[0] for prompt in anchor_prompts],
                        [prompt[1] for prompt in anchor_prompts],
                        [prompt[2] for prompt in anchor_prompts],
                        device,
                        max_len=cfg.max_text_len,
                        pretrained_name=cfg.text_pretrained_name,
                        local_files_only=cfg.text_local_files_only,
                    )
                )
                intervention_factor = torch.tensor(
                    [factor_lookup[attribute] for attribute in attributes],
                    device=device,
                    dtype=torch.long,
                )
        subject_state_low_mask = None
        if cfg.conditioning_mode in NUCLEUS_GATED_MODES:
            object_types = [
                str(record.get("object_type") or "").strip()
                for record in records
            ]
            invalid = sorted(set(object_types) - {"nucleus", "whole_cell"})
            if invalid:
                raise ValueError(
                    "subject_state_low_nucleus requires valid object_type values; "
                    f"found {invalid}"
                )
            subject_state_low_mask = torch.tensor(
                [value == "nucleus" for value in object_types],
                device=device,
                dtype=torch.bool,
            )
        spatial_subject_mask = None
        spatial_subject_route_mask = None
        morphology_class_ids = None
        spatial_subject_mask_paths: list[str | None] = [None for _ in records]
        if cfg.conditioning_mode in {
            SPATIAL_SUBJECT_INPUT_ADAPTER_MODE,
            MORPHOLOGY_SPATIAL_AFFINE_MODE,
        }:
            (
                spatial_subject_mask,
                spatial_subject_route_mask,
                spatial_subject_mask_paths,
            ) = spatial_subject_masks_from_records(records, data_root, device)
        elif cfg.conditioning_mode in {
            MORPHOLOGY_LATENT_BANK_MODE,
            MORPHOLOGY_FEATURE_AFFINE_MODE,
        }:
            (
                morphology_class_ids,
                spatial_subject_route_mask,
            ) = morphology_class_ids_from_records(records, device)
        default_cfg_scale = (
            float(args.cfg_scale)
            if args.cfg_scale is not None
            else float(cfg.cfg_scale)
        )
        (
            batch_cfg_subject,
            batch_cfg_state,
            batch_cfg_background,
        ) = target_factor_cfg_scales(
            records,
            device=device,
            default_scale=default_cfg_scale,
            subject_scale=args.cfg_scale_subject,
            state_scale=args.cfg_scale_state,
            background_scale=args.cfg_scale_background,
            target_scale=args.target_factor_cfg_scale,
        )
        with torch.inference_mode(), torch.amp.autocast(
            "cuda",
            enabled=amp_enabled,
            dtype=amp_dtype,
        ):
            generated = model.sample(
                subject_ids,
                state_ids,
                bg_ids,
                cfg_scale=args.cfg_scale,
                cfg_scale_subject=batch_cfg_subject,
                cfg_scale_state=batch_cfg_state,
                cfg_scale_background=batch_cfg_background,
                num_steps=args.steps,
                use_personalization=args.use_personalization,
                generator=generator,
                subject_state_low_mask=subject_state_low_mask,
                spatial_subject_mask=spatial_subject_mask,
                spatial_subject_route_mask=spatial_subject_route_mask,
                morphology_class_ids=morphology_class_ids,
                anchor_subject_ids=anchor_subject_ids,
                anchor_state_ids=anchor_state_ids,
                anchor_bg_ids=anchor_bg_ids,
                intervention_factor=intervention_factor,
            )
        output_records = []
        for index, ((record, repeat), image, reference) in enumerate(
            zip(batch_jobs, generated, reference_paths)
        ):
            generated_id = f"{record['sample_id']}__r{repeat:02d}"
            out_path = images_root / f"{generated_id}.png"
            save_image(image, out_path)
            output_records.append(
                {
                    "generated_id": generated_id,
                    "sample_id": record["sample_id"],
                    "repeat": repeat,
                    "batch_seed": batch_seed,
                    "batch_offset": index,
                    "batch_members": [
                        f"{member_record['sample_id']}__r{member_repeat:02d}"
                        for member_record, member_repeat in batch_jobs
                    ],
                    "rank": rank,
                    "world_size": world_size,
                    "generated_image": str(out_path),
                    "reference_image": str(reference.resolve()),
                    "subject": record.get("evaluation_subject", record["subject"]),
                    "conditioning_subject": conditioning_subjects[index],
                    "state": record["state"],
                    "background": record["background"],
                    "target_attribute": record.get("target_attribute"),
                    "anchor_prompt": (
                        {
                            "subject": anchor_prompts[index][0],
                            "state": anchor_prompts[index][1],
                            "background": anchor_prompts[index][2],
                        }
                        if anchor_prompts[index] is not None
                        else None
                    ),
                    "source_name": record.get("source_name") or record.get("source_dataset"),
                    "object_type": record.get("object_type") or record.get("object_scope"),
                    "morphology_class": record.get("morphology_class"),
                    "spatial_subject_mask_path": spatial_subject_mask_paths[index],
                    "spatial_subject_routed": bool(
                        spatial_subject_route_mask[index].item()
                    ) if spatial_subject_route_mask is not None else False,
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "use_personalization": args.use_personalization,
                    "concept_token": concept_token,
                    "steps": args.steps or cfg.num_inference_steps,
                    "cfg_scale": args.cfg_scale if args.cfg_scale is not None else cfg.cfg_scale,
                    "cfg_scale_subject": cfg_value_at(
                        batch_cfg_subject,
                        index,
                    ),
                    "cfg_scale_state": cfg_value_at(
                        batch_cfg_state,
                        index,
                    ),
                    "cfg_scale_background": cfg_value_at(
                        batch_cfg_background,
                        index,
                    ),
                }
            )
        append_jsonl(rank_manifest, output_records)
        print(f"rank={rank} written {start + len(batch_jobs)}/{len(rank_jobs)}")

    if use_ddp:
        dist.barrier()
        if rank == 0:
            merge_rank_manifests(output_root, output_manifest, expected_ids)
            print("saved", output_manifest)
        dist.barrier()
        cleanup_distributed()
    else:
        merge_rank_manifests(output_root, output_manifest, expected_ids)
        print("saved", output_manifest)


if __name__ == "__main__":
    main()
