"""Infer text."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mscdms import MSCDM
from mscdms.checkpoint import config_from_checkpoint, load_checkpoint
from mscdms.text_encoder import MultiScaleTextEncoder

NUCLEUS_GATED_MODES = {
    "subject_state_low_nucleus",
    "subject_state_low_nucleus_adapter",
    "subject_state_low_nucleus_multiscale_adapter",
    "subject_state_low_nucleus_prompt_adapters",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/mscdm.pt")
    ap.add_argument("--out", type=str, default="sample.png")
    ap.add_argument("--subject", type=str, default="MCF7 cell")
    ap.add_argument("--state", type=str, default="normal proliferating")
    ap.add_argument("--background", type=str, default="brightfield microscope")
    ap.add_argument("--cfg_scale", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument(
        "--object_type",
        choices=("nucleus", "whole_cell"),
        default=None,
        help="subject_state_low_nucleus checkpoint 必须显式指定对象类型",
    )
    ap.add_argument("--clip_model_dir", default="models/clip-vit-base-patch32")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--use_personalization",
        action="store_true",
        help="使用 personalize.py 保存的权重时开启，以启用概念向量（低分辨率主体分支）",
    )
    ap.add_argument(
        "--concept_token",
        default=None,
        help="覆盖 checkpoint 中保存的 concept_token",
    )
    ap.add_argument(
        "--no_checkpoint_concept_token",
        action="store_true",
        help="启用个体化向量，但不自动向主体文本添加 checkpoint concept_token",
    )
    args = ap.parse_args()

    ckpt = load_checkpoint(args.checkpoint, map_location=args.device)
    cfg = config_from_checkpoint(ckpt)
    cfg.text_pretrained_name = args.clip_model_dir
    cfg.text_local_files_only = True
    cfg.freeze_clip_backbone = True
    model = MSCDM(cfg).to(args.device)
    checkpoint_token = ""
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        checkpoint_token = str(ckpt.get("concept_token") or "").strip()
        if args.use_personalization and checkpoint_token:
            print("checkpoint concept_token:", checkpoint_token)
    else:
        model.load_state_dict(ckpt)

    model.eval()
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device=dev).manual_seed(args.seed)
    n = max(1, args.num_samples)
    concept_token = ""
    if args.use_personalization:
        if args.no_checkpoint_concept_token:
            concept_token = ""
        elif args.concept_token is not None:
            concept_token = args.concept_token.strip()
        else:
            concept_token = checkpoint_token
    subject = args.subject
    if concept_token and concept_token not in subject:
        subject = f"{concept_token} {subject}"
    sid, stid, bgid = MultiScaleTextEncoder.texts_to_batch(
        [subject] * n,
        [args.state] * n,
        [args.background] * n,
        dev,
        max_len=cfg.max_text_len,
        pretrained_name=cfg.text_pretrained_name,
        local_files_only=cfg.text_local_files_only,
    )
    subject_state_low_mask = None
    if cfg.conditioning_mode in NUCLEUS_GATED_MODES:
        if args.object_type is None:
            raise ValueError(
                "subject_state_low_nucleus checkpoint requires --object_type"
            )
        subject_state_low_mask = torch.full(
            (n,),
            args.object_type == "nucleus",
            device=dev,
            dtype=torch.bool,
        )

    with torch.no_grad():
        rgb = model.sample(
            sid,
            stid,
            bgid,
            cfg_scale=args.cfg_scale,
            num_steps=args.steps,
            use_personalization=args.use_personalization,
            generator=generator,
            subject_state_low_mask=subject_state_low_mask,
        )
    # [-1,1] -> [0,255] uint8
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for index in range(n):
        t = (rgb[index].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1)
        arr = (t * 255.0).astype(np.uint8)
        path = out if n == 1 else out.with_name(f"{out.stem}_{index:03d}{out.suffix or '.png'}")
        Image.fromarray(arr).save(path)
        print("saved", path)


if __name__ == "__main__":
    main()
