#!/usr/bin/env python3
"""Cache foreground latents."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mscdms.latent_foreground_diffusion import (
    load_original_v4_high_vae,
    normalize_high_latent,
)
from mscdms.workflows.train_foreground_renderer import (
    ForegroundRendererDataset,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest_audit", type=Path, required=True)
    parser.add_argument("--vae_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    audit = json.loads(args.manifest_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or audit.get("test_accessed") is not False:
        raise ValueError("Stage-26 manifest audit did not pass")
    rows = read_jsonl(args.manifest)
    if len(rows) != 11000:
        raise ValueError(f"expected 11,000 train/val records, got {len(rows)}")
    if any(row.get("split") == "test" for row in rows):
        raise ValueError("test records entered Stage-27a latent cache")
    dataset = ForegroundRendererDataset(
        rows, data_dir=args.data_dir, image_size=512, augment=False
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    device = torch.device(args.device)
    vae, calibration = load_original_v4_high_vae(
        args.vae_checkpoint, device=device
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_path = args.output_dir / "target_latents.npy"
    background_path = args.output_dir / "background_latents.npy"
    mask_path = args.output_dir / "masks.npy"
    target_memmap = np.lib.format.open_memmap(
        target_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), int(calibration["latent_channels"]), 64, 64),
    )
    background_memmap = np.lib.format.open_memmap(
        background_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), int(calibration["latent_channels"]), 64, 64),
    )
    mask_memmap = np.lib.format.open_memmap(
        mask_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(rows), 1, 64, 64),
    )
    row_by_id = {str(row["sample_id"]): row for row in rows}
    index_rows: list[dict[str, Any]] = []
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            target = batch["target"].to(device, non_blocking=True)
            background = batch["background"].to(device, non_blocking=True)
            combined = torch.cat((target, background), dim=0)
            with torch.amp.autocast(
                "cuda", enabled=device.type == "cuda", dtype=torch.bfloat16
            ):
                means, _ = vae.encode(combined)
                means = normalize_high_latent(means, calibration)
            target_latent, background_latent = means.chunk(2, dim=0)
            latent_mask = F.interpolate(
                batch["mask"], size=(64, 64), mode="nearest"
            )
            count = len(target)
            target_memmap[cursor : cursor + count] = (
                target_latent.float().cpu().numpy().astype(np.float16)
            )
            background_memmap[cursor : cursor + count] = (
                background_latent.float().cpu().numpy().astype(np.float16)
            )
            mask_memmap[cursor : cursor + count] = (
                (latent_mask.numpy() >= 0.5).astype(np.uint8)
            )
            for offset, sample_id in enumerate(batch["sample_id"]):
                row = row_by_id[str(sample_id)]
                payload = row["foreground_renderer"]
                index_rows.append(
                    {
                        "cache_index": cursor + offset,
                        "sample_id": str(sample_id),
                        "split": str(row["split"]),
                        "source_name": str(row["source_name"]),
                        "source_id": int(payload["source_id"]),
                        "state_id": int(payload["state_id"]),
                        "state_key": str(payload["state_key"]),
                        "morphology": str(payload["morphology"]),
                        "image_path": str(row["image_path"]),
                        "target_mask_path": str(row["target_mask_path"]),
                    }
                )
            cursor += count
            print(f"cached high-VAE latents {cursor}/{len(rows)}", flush=True)
    target_memmap.flush()
    background_memmap.flush()
    mask_memmap.flush()
    if cursor != len(rows):
        raise ValueError(f"cache incomplete: {cursor}/{len(rows)}")
    index_path = args.output_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {
        "schema": "exp4.latent_foreground_diffusion.stage27a.cache.v1",
        "status": "pass",
        "records": len(index_rows),
        "split_counts": {
            "train": sum(row["split"] == "train" for row in index_rows),
            "val": sum(row["split"] == "val" for row in index_rows),
        },
        "shape": [len(rows), int(calibration["latent_channels"]), 64, 64],
        "latent_dtype": "float16",
        "mask_dtype": "uint8",
        "vae_checkpoint": str(args.vae_checkpoint.resolve()),
        "vae_checkpoint_sha256": sha256_file(args.vae_checkpoint),
        "latent_calibration": calibration,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (target_path, background_path, mask_path, index_path)
        },
        "paired_counterfactual_targets_used": False,
        "test_accessed": False,
    }
    (args.output_dir / "cache_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("saved", args.output_dir / "cache_audit.json")


if __name__ == "__main__":
    main()
