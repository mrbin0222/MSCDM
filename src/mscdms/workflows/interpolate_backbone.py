#!/usr/bin/env python3
"""Interpolate backbone."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_checkpoint", type=Path, required=True)
    parser.add_argument("--state_checkpoint", type=Path, required=True)
    parser.add_argument("--output_checkpoint", type=Path, required=True)
    parser.add_argument("--audit_json", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interpolate_state_dicts(
    base: Mapping[str, torch.Tensor],
    state: Mapping[str, torch.Tensor],
    *,
    alpha: float,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if set(base) != set(state):
        missing = sorted(set(base) - set(state))
        extra = sorted(set(state) - set(base))
        raise ValueError(
            f"checkpoint model keys differ: missing={missing[:8]} extra={extra[:8]}"
        )
    output: dict[str, torch.Tensor] = {}
    for name in sorted(base):
        left = base[name].detach().cpu()
        right = state[name].detach().cpu()
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(
                f"tensor mismatch for {name}: "
                f"{tuple(left.shape)}/{left.dtype} vs "
                f"{tuple(right.shape)}/{right.dtype}"
            )
        if left.is_floating_point() or left.is_complex():
            work_dtype = torch.float64 if left.dtype == torch.float64 else torch.float32
            mixed = torch.lerp(
                left.to(dtype=work_dtype),
                right.to(dtype=work_dtype),
                float(alpha),
            ).to(dtype=left.dtype)
            output[name] = mixed.contiguous()
        else:
            if not torch.equal(left, right):
                raise ValueError(f"non-floating tensor differs: {name}")
            output[name] = left.clone().contiguous()
    return output


def require_equal_metadata(
    base: Mapping[str, Any],
    state: Mapping[str, Any],
    keys: tuple[str, ...],
) -> None:
    for key in keys:
        if base.get(key) != state.get(key):
            raise ValueError(f"checkpoint metadata differs: {key}")


def main() -> None:
    args = parse_args()
    if abs(float(args.alpha) - 0.5) > 1e-12:
        raise ValueError("Stage-35a is preregistered for alpha=0.5 only")
    base = torch.load(
        args.base_checkpoint, map_location="cpu", weights_only=False
    )
    state = torch.load(
        args.state_checkpoint, map_location="cpu", weights_only=False
    )
    base.pop("optimizer", None)
    state.pop("optimizer", None)
    if base.get("schema") != "exp4.latent_foreground_diffusion.stage27a.checkpoint.v1":
        raise ValueError("base checkpoint is not Stage-27a")
    if state.get("schema") != "exp4.latent_foreground_diffusion.stage28a.checkpoint.v1":
        raise ValueError("state checkpoint is not Stage-28a")
    if state.get("initial_checkpoint_sha256") != sha256_file(args.base_checkpoint):
        raise ValueError("Stage-28a was not initialized from the supplied Stage-27a")
    require_equal_metadata(
        base,
        state,
        (
            "model_config",
            "source_to_id",
            "state_to_id",
            "latent_calibration",
            "vae_checkpoint_sha256",
        ),
    )
    mixed_model = interpolate_state_dicts(
        base["model"], state["model"], alpha=float(args.alpha)
    )
    payload = {
        key: copy.deepcopy(value)
        for key, value in state.items()
        if key not in {"model", "optimizer"}
    }
    payload["model"] = mixed_model
    payload["training_role"] = "stage35a_fixed_weight_midpoint"
    payload["stage35a_weight_interpolation"] = {
        "schema": "exp4.latent_foreground.weight_midpoint.stage35a.v1",
        "alpha_state_checkpoint": float(args.alpha),
        "alpha_base_checkpoint": 1.0 - float(args.alpha),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "state_checkpoint": str(args.state_checkpoint.resolve()),
        "state_checkpoint_sha256": sha256_file(args.state_checkpoint),
        "hyperparameter_scan": False,
        "new_training": False,
        "test_accessed": False,
    }
    payload["test_accessed"] = False
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_checkpoint)
    audit = {
        "schema": "exp4.latent_foreground.weight_midpoint.stage35a.audit.v1",
        "status": "pass",
        "alpha": float(args.alpha),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "state_checkpoint": str(args.state_checkpoint.resolve()),
        "state_checkpoint_sha256": sha256_file(args.state_checkpoint),
        "output_checkpoint": str(args.output_checkpoint.resolve()),
        "output_checkpoint_sha256": sha256_file(args.output_checkpoint),
        "model_tensor_count": len(mixed_model),
        "model_keys_identical": True,
        "metadata_identical": True,
        "hyperparameter_scan": False,
        "new_training": False,
        "test_accessed": False,
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("saved", args.output_checkpoint)
    print("saved", args.audit_json)
    print("STAGE-35A FIXED ALPHA: 0.5")
    print("NEW TRAINING: NO")
    print("TEST DATA ACCESSED: NO")


if __name__ == "__main__":
    main()
