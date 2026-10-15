#!/usr/bin/env python3
"""Evaluate images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import transformers
from transformers import CLIPModel, CLIPProcessor

from mscdms.text_encoder import resolve_local_pretrained_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument(
        "--evaluation_size",
        type=int,
        default=0,
        help=(
            "可选：把生成图和参考图统一缩放到固定方形尺寸后计算；"
            "混合原生尺寸数据集必须显式设置。0表示沿用每张参考图尺寸。"
        ),
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--expected_count",
        type=int,
        default=0,
        help="可选：正式评价前要求 manifest 恰好包含该数量的生成记录",
    )
    ap.add_argument("--clip_model_dir", default="models/clip-vit-base-patch32")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--reference_normalize",
        choices=("percentile", "minmax", "none"),
        default="percentile",
        help="参考图按训练输入口径归一化；正式训练默认 percentile",
    )
    ap.add_argument("--p_lo", type=float, default=1.0)
    ap.add_argument("--p_hi", type=float, default=99.0)
    ap.add_argument(
        "--metrics",
        nargs="+",
        choices=("fid", "clipsim", "ssim", "psnr", "lpips"),
        default=("fid", "clipsim", "ssim", "psnr", "lpips"),
        help="默认计算论文中的五项通用指标；可只指定其中一部分",
    )
    ap.add_argument(
        "--fid_weights",
        type=Path,
        default=None,
        help="可选：torch-fidelity Inception 权重文件；服务器离线时建议显式指定",
    )
    ap.add_argument(
        "--lpips_backbone",
        choices=("alex", "vgg", "squeeze"),
        default="alex",
        help="LPIPS 特征主干；论文和结果表中必须记录该选择",
    )
    return ap.parse_args()


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def image_tensor(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    array = np.asarray(image.resize(size, Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def normalize_reference(
    image: Image.Image,
    mode: str,
    p_lo: float,
    p_hi: float,
) -> Image.Image:
    """复现 CellMicroscopyDataset 的逐通道训练输入归一化。"""
    if mode == "none":
        return image
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    flat = array.reshape(-1, array.shape[-1])
    if mode == "percentile":
        lo = np.percentile(flat, p_lo, axis=0)
        hi = np.percentile(flat, p_hi, axis=0)
    elif mode == "minmax":
        lo = flat.min(axis=0)
        hi = flat.max(axis=0)
    else:
        raise ValueError(mode)
    normalized = np.clip((array - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
    return Image.fromarray((normalized * 255.0).round().astype(np.uint8))


def gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    one_d = torch.exp(-(coords**2) / (2 * sigma**2))
    one_d /= one_d.sum()
    kernel = one_d[:, None] @ one_d[None, :]
    return kernel[None, None]


def ssim_per_image(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    kernel = gaussian_kernel().to(device=x.device, dtype=x.dtype).expand(3, 1, -1, -1)
    # 原始 SSIM 使用 11x11、sigma=1.5 的高斯窗，并只汇总有效区域；
    # 不做零填充，避免图像边缘被人为拉低。
    mu_x = F.conv2d(x, kernel, groups=3)
    mu_y = F.conv2d(y, kernel, groups=3)
    sigma_x = F.conv2d(x * x, kernel, groups=3) - mu_x.square()
    sigma_y = F.conv2d(y * y, kernel, groups=3) - mu_y.square()
    sigma_xy = F.conv2d(x * y, kernel, groups=3) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return score.mean(dim=(1, 2, 3))


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()) if array.size else float("nan"),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)) if array.size else float("nan"),
    }


def pooled_feature_tensor(output) -> torch.Tensor:
    """兼容 Transformers 4.x 的 Tensor 与 5.x 的结构化特征输出。"""
    if torch.is_tensor(output):
        return output
    pooled = getattr(output, "pooler_output", None)
    if torch.is_tensor(pooled):
        return pooled
    raise TypeError(
        "CLIP get_*_features 返回了不支持的类型："
        f"{type(output)!r}；既不是 Tensor，也没有 Tensor 类型的 pooler_output"
    )


def build_fid_metric(args: argparse.Namespace, device: torch.device):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "计算 FID 需要 torchmetrics 和 torch-fidelity；"
            "请先安装 requirements-eval.txt"
        ) from exc

    kwargs: dict[str, Any] = {
        "feature": 2048,
        "normalize": True,
        # 与原始 FID 的 resize 口径一致，不额外启用抗锯齿。
        "antialias": False,
    }
    if args.fid_weights is not None:
        weights = args.fid_weights.resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"FID 权重不存在: {weights}")
        kwargs["feature_extractor_weights_path"] = str(weights)
    metric = FrechetInceptionDistance(**kwargs).to(device)
    # 官方实现建议用 float64 累积均值/协方差，提高矩阵平方根稳定性。
    return metric.set_dtype(torch.float64)


def build_lpips_metric(args: argparse.Namespace, device: torch.device):
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "计算 LPIPS 需要 torchmetrics 和 torchvision；"
            "请先安装 requirements-eval.txt"
        ) from exc
    return (
        LearnedPerceptualImagePatchSimilarity(
            net_type=args.lpips_backbone,
            reduction="none",
            normalize=True,
        )
        .to(device)
        .eval()
    )


def summarize_records(records: list[dict], metrics: tuple[str, ...]) -> dict:
    return {
        "n": len(records),
        "metrics": {
            metric: summarize([record[metric] for record in records])
            for metric in metrics
        },
    }


def summarize_by(records: list[dict], metrics: tuple[str, ...], keys: tuple[str, ...]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        label = "/".join(str(record.get(key) or "unknown") for key in keys)
        grouped.setdefault(label, []).append(record)
    return {
        label: summarize_records(group, metrics)
        for label, group in sorted(grouped.items())
    }


def main() -> None:
    args = parse_args()
    requested_metrics = set(args.metrics)
    manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else manifest.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest.open("r", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise ValueError("生成 manifest 为空")
    if args.expected_count > 0 and len(records) != args.expected_count:
        raise ValueError(
            f"生成 manifest 数量不符: expected={args.expected_count}, found={len(records)}"
        )
    generated_ids = [str(record["generated_id"]) for record in records]
    if len(set(generated_ids)) != len(generated_ids):
        raise ValueError("生成 manifest 含重复 generated_id")

    device = torch.device(args.device)
    clip_path = None
    clip_model = None
    processor = None
    if "clipsim" in requested_metrics:
        clip_path = resolve_local_pretrained_path(args.clip_model_dir)
        clip_model = CLIPModel.from_pretrained(clip_path, local_files_only=True).to(device).eval()
        processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)
    fid_metric = build_fid_metric(args, device) if "fid" in requested_metrics else None
    lpips_metric = build_lpips_metric(args, device) if "lpips" in requested_metrics else None
    per_sample = []

    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        generated_pil = [load_rgb(record["generated_image"]) for record in batch]
        reference_pil = [
            normalize_reference(
                load_rgb(record["reference_image"]),
                args.reference_normalize,
                args.p_lo,
                args.p_hi,
            )
            for record in batch
        ]
        target_sizes = (
            [(args.evaluation_size, args.evaluation_size) for _ in reference_pil]
            if args.evaluation_size > 0
            else [image.size for image in reference_pil]
        )
        if len(set(target_sizes)) != 1:
            raise ValueError(
                "同一评价批次含不同参考尺寸；请设置 --evaluation_size 固定空间口径"
            )
        generated_tensor = torch.stack(
            [image_tensor(image, size) for image, size in zip(generated_pil, target_sizes)]
        ).to(device)
        reference_tensor = torch.stack(
            [image_tensor(image, size) for image, size in zip(reference_pil, target_sizes)]
        ).to(device)
        psnr = None
        if "psnr" in requested_metrics:
            mse = (
                (generated_tensor - reference_tensor)
                .square()
                .mean(dim=(1, 2, 3))
                .clamp(min=1e-12)
            )
            # 输入已缩放到 [0,1]，故 MAX_I=1。
            psnr = 10.0 * torch.log10(1.0 / mse)
        ssim = (
            ssim_per_image(generated_tensor, reference_tensor)
            if "ssim" in requested_metrics
            else None
        )

        lpips = None
        with torch.inference_mode():
            if fid_metric is not None:
                fid_metric.update(reference_tensor, real=True)
                fid_metric.update(generated_tensor, real=False)
            if lpips_metric is not None:
                lpips = lpips_metric(generated_tensor, reference_tensor).reshape(-1)

        clip_text = None
        clip_sim = None
        clip_image = None
        if clip_model is not None and processor is not None:
            prompts = [
                f"{record['subject']}; {record['state']}; {record['background']}"
                for record in batch
            ]
            generated_inputs = processor(
                text=prompts,
                images=generated_pil,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            reference_inputs = processor(images=reference_pil, return_tensors="pt").to(device)
            with torch.inference_mode():
                generated_image_features = F.normalize(
                    pooled_feature_tensor(
                        clip_model.get_image_features(
                            pixel_values=generated_inputs["pixel_values"]
                        )
                    ),
                    dim=-1,
                )
                text_features = F.normalize(
                    pooled_feature_tensor(
                        clip_model.get_text_features(
                            input_ids=generated_inputs["input_ids"],
                            attention_mask=generated_inputs.get("attention_mask"),
                        )
                    ),
                    dim=-1,
                )
                reference_features = F.normalize(
                    pooled_feature_tensor(
                        clip_model.get_image_features(
                            pixel_values=reference_inputs["pixel_values"]
                        )
                    ),
                    dim=-1,
                )
            clip_text = (generated_image_features * text_features).sum(dim=-1)
            # 与论文式 (2-22) 对齐：CLIPSim=max(cosine, 0)。
            clip_sim = clip_text.clamp(min=0.0, max=1.0)
            # 非论文五项指标，仅保留作生成图—参考图的语义诊断。
            clip_image = (generated_image_features * reference_features).sum(dim=-1)

        for index, record in enumerate(batch):
            sample = {
                "generated_id": record["generated_id"],
                "sample_id": record["sample_id"],
                "source_name": record.get("source_name"),
                "object_type": record.get("object_type"),
            }
            if psnr is not None:
                sample["psnr"] = float(psnr[index].cpu())
            if ssim is not None:
                sample["ssim"] = float(ssim[index].cpu())
            if lpips is not None:
                sample["lpips"] = float(lpips[index].cpu())
            if clip_sim is not None and clip_text is not None and clip_image is not None:
                sample["clipsim"] = float(clip_sim[index].cpu())
                sample["clip_text_cosine_raw"] = float(clip_text[index].cpu())
                sample["clip_image_cosine_diagnostic"] = float(clip_image[index].cpu())
            per_sample.append(sample)
        print(f"evaluated {start + len(batch)}/{len(records)}")

    per_sample_path = output_dir / "per_sample.jsonl"
    with per_sample_path.open("w", encoding="utf-8") as fh:
        for record in per_sample:
            fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    paired_metrics = tuple(
        metric
        for metric in ("clipsim", "ssim", "psnr", "lpips")
        if metric in requested_metrics
    )
    metric_summary = {
        metric: summarize([record[metric] for record in per_sample])
        for metric in paired_metrics
    }
    if fid_metric is not None:
        metric_summary["fid"] = {
            "value": float(fid_metric.compute().cpu()),
            "n_real": len(per_sample),
            "n_generated": len(per_sample),
            "feature": 2048,
        }
    summary = {
        "n": len(per_sample),
        "manifest": str(manifest),
        "requested_metrics": list(args.metrics),
        "clip_model": clip_path,
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(device),
        },
        "reference_preprocessing": {
            "normalize": args.reference_normalize,
            "p_lo": args.p_lo,
            "p_hi": args.p_hi,
            "evaluation_size": args.evaluation_size or "per-reference native size",
        },
        "metric_protocol": {
            "fid": {
                "scope": "overall generated/reference distributions",
                "implementation": "torchmetrics/torch-fidelity InceptionV3 pool3",
                "feature": 2048,
                "float64_statistics": True,
                "antialias": False,
                "weights": str(args.fid_weights.resolve()) if args.fid_weights else "default",
            }
            if fid_metric is not None
            else None,
            "clipsim": {
                "definition": "max(cos(CLIP_image, CLIP_text), 0)",
                "prompt": "subject; state; background",
            }
            if "clipsim" in requested_metrics
            else None,
            "ssim": {
                "window": 11,
                "sigma": 1.5,
                "data_range": 1.0,
                "valid_region": True,
            }
            if "ssim" in requested_metrics
            else None,
            "psnr": {"data_range": 1.0} if "psnr" in requested_metrics else None,
            "lpips": {
                "backbone": args.lpips_backbone,
                "input_range": [0.0, 1.0],
            }
            if "lpips" in requested_metrics
            else None,
        },
        "metrics": metric_summary,
        "by_source": summarize_by(per_sample, paired_metrics, ("source_name",)),
        "by_object_type": summarize_by(per_sample, paired_metrics, ("object_type",)),
        "by_source_object_type": summarize_by(
            per_sample,
            paired_metrics,
            ("source_name", "object_type"),
        ),
        "interpretation": {
            "fid": "lower is better; compare only under the same sample count and preprocessing",
            "clipsim": "higher is better; prompt-image semantic alignment",
            "psnr_ssim_lpips": (
                "paired diagnostics only; conditional generation is one-to-many "
                "(PSNR/SSIM higher is better, LPIPS lower is better)"
            ),
            "grouped_fid": (
                "not reported: per-source sample counts are too small and unequal for a stable "
                "2048-dimensional FID estimate"
            ),
            "clip_image_cosine_diagnostic": (
                "stored only in per_sample.jsonl; not one of the five formal metrics"
            ),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("saved", summary_path)


if __name__ == "__main__":
    main()
