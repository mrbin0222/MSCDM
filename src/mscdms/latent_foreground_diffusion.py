from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .checkpoint import config_from_checkpoint, load_checkpoint
from .diffusion import GaussianDiffusion
from .latent_vae import LatentAutoEncoder
from .unet import ScaleUNet


class FactorTokenConditioner(nn.Module):
    """Keep acquisition/source and biological state as separate tokens."""

    def __init__(self, num_sources: int, num_states: int, dim: int = 256) -> None:
        super().__init__()
        self.num_sources = int(num_sources)
        self.num_states = int(num_states)
        self.dim = int(dim)
        self.source = nn.Embedding(self.num_sources, self.dim)
        self.state = nn.Embedding(self.num_states, self.dim)
        self.source_projection = nn.Sequential(
            nn.Linear(self.dim, self.dim), nn.SiLU(), nn.Linear(self.dim, self.dim)
        )
        self.state_projection = nn.Sequential(
            nn.Linear(self.dim, self.dim), nn.SiLU(), nn.Linear(self.dim, self.dim)
        )

    def forward(
        self, source_ids: torch.Tensor, state_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.source_projection(self.source(source_ids))
        state = self.state_projection(self.state(state_ids))
        sequence = torch.stack((source, state), dim=1)
        return sequence, sequence.mean(dim=1)


class MaskConditionedLatentForegroundDiffusion(nn.Module):
    """Stochastic foreground model with explicit geometry and exact RGB bypass.

    The diffusion state is the frozen original-v4 high-resolution VAE latent.
    The denoiser additionally observes the masked background latent and a binary
    support mask.  Source and state are separate categorical condition tokens.
    """

    def __init__(
        self,
        *,
        num_sources: int,
        num_states: int,
        latent_channels: int = 4,
        condition_dim: int = 256,
        base_channels: int = 64,
        input_resolution: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 1,
        num_heads: int = 4,
        num_train_timesteps: int = 1000,
        prediction_type: str = "v",
        state_attention_delta_rank: int | None = None,
        state_feature_delta_classes: int | None = None,
        target_state_response_class: int | None = None,
    ) -> None:
        super().__init__()
        if prediction_type not in {"epsilon", "v"}:
            raise ValueError(f"unsupported prediction type: {prediction_type}")
        if state_attention_delta_rank is not None and state_attention_delta_rank <= 0:
            raise ValueError("state_attention_delta_rank must be positive when enabled")
        if state_feature_delta_classes is not None and state_feature_delta_classes <= 1:
            raise ValueError("state_feature_delta_classes must exceed one when enabled")
        if target_state_response_class is not None and target_state_response_class < 0:
            raise ValueError("target_state_response_class must be non-negative")
        if (
            state_attention_delta_rank is not None
            and state_feature_delta_classes is not None
        ):
            raise ValueError("attention and feature state deltas are mutually exclusive")
        self.latent_channels = int(latent_channels)
        self.input_resolution = int(input_resolution)
        self.prediction_type = prediction_type
        self.state_attention_delta_rank = state_attention_delta_rank
        self.state_feature_delta_classes = state_feature_delta_classes
        self.target_state_response_class = target_state_response_class
        self.conditioner = FactorTokenConditioner(
            num_sources=num_sources, num_states=num_states, dim=condition_dim
        )
        attention_resolutions = tuple(
            value
            for value in (
                input_resolution // 2,
                input_resolution // 4,
                input_resolution // 8,
            )
            if value >= 1
        )
        self.denoiser = ScaleUNet(
            in_channels=self.latent_channels * 2 + 1,
            out_channels=self.latent_channels,
            base_ch=base_channels,
            cond_dim=condition_dim,
            tdim=256,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            num_heads=num_heads,
            context_dim=condition_dim,
            input_resolution=input_resolution,
            use_reversible=False,
            factor_delta_attention_rank=state_attention_delta_rank,
            state_delta_feature_classes=state_feature_delta_classes,
            target_state_response_class=target_state_response_class,
        )
        self.diffusion = GaussianDiffusion(
            num_train_timesteps, "linear", 1e-4, 2e-2
        )

    @property
    def model_config(self) -> dict[str, Any]:
        denoiser = self.denoiser
        channel_mult = tuple(
            int(block[0].conv1.out_channels // denoiser.down_blocks[0][0].conv1.out_channels)
            for block in denoiser.down_blocks
        )
        return {
            "num_sources": self.conditioner.num_sources,
            "num_states": self.conditioner.num_states,
            "latent_channels": self.latent_channels,
            "condition_dim": self.conditioner.dim,
            "base_channels": denoiser.down_blocks[0][0].conv1.out_channels,
            "input_resolution": self.input_resolution,
            "channel_mult": channel_mult,
            "num_res_blocks": denoiser.num_res_blocks,
            "num_heads": denoiser.mid_attn.cross_attn.num_heads,
            "num_train_timesteps": int(self.diffusion.betas.shape[0]),
            "prediction_type": self.prediction_type,
            "state_attention_delta_rank": self.state_attention_delta_rank,
            "state_feature_delta_classes": self.state_feature_delta_classes,
            "target_state_response_class": self.target_state_response_class,
        }

    @property
    def has_state_attention_delta(self) -> bool:
        return self.state_attention_delta_rank is not None

    def state_attention_delta_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for name, parameter in self.named_parameters()
            if ".factor_delta_" in name
        ]

    @property
    def has_state_feature_delta(self) -> bool:
        return self.state_feature_delta_classes is not None

    def state_feature_delta_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for name, parameter in self.named_parameters()
            if "state_delta_affines" in name
        ]

    @property
    def has_target_state_response(self) -> bool:
        return self.target_state_response_class is not None

    def target_state_response_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for name, parameter in self.named_parameters()
            if "target_state_response_affines" in name
        ]

    def _state_attention_delta(
        self,
        source_ids: torch.Tensor,
        target_state_ids: torch.Tensor,
        anchor_state_ids: torch.Tensor | None,
        target_condition: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Build an edit-only target-minus-anchor state token delta.

        Factual generation deliberately omits ``anchor_state_ids`` and thereby
        bypasses every new attention parameter.  This is stronger than a
        floating-point gate set to zero: the base attention path is called
        without a delta branch at all.
        """
        if not self.has_state_attention_delta or anchor_state_ids is None:
            return None, None
        if anchor_state_ids.shape != target_state_ids.shape:
            raise ValueError("anchor_state_ids must match target_state_ids")
        anchor_sequence, _ = self.conditioner(source_ids, anchor_state_ids)
        target_sequence = target_condition[0]
        delta = torch.zeros_like(target_sequence)
        delta[:, 1] = target_sequence[:, 1] - anchor_sequence[:, 1]
        route_mask = target_state_ids.ne(anchor_state_ids)
        return delta, route_mask

    def _predict(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        background_latent: torch.Tensor,
        mask: torch.Tensor,
        condition: tuple[torch.Tensor, torch.Tensor],
        factor_delta_context: torch.Tensor | None = None,
        factor_delta_mask: torch.Tensor | None = None,
        state_target_class_ids: torch.Tensor | None = None,
        state_anchor_class_ids: torch.Tensor | None = None,
        state_delta_route_mask: torch.Tensor | None = None,
        target_state_response_route_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs = torch.cat((noisy, background_latent, mask), dim=1)
        return self.denoiser(
            inputs,
            timesteps,
            condition[0],
            condition[1],
            factor_delta_context=factor_delta_context,
            factor_delta_mask=factor_delta_mask,
            state_target_class_ids=state_target_class_ids,
            state_anchor_class_ids=state_anchor_class_ids,
            state_delta_route_mask=state_delta_route_mask,
            target_state_response_route_mask=target_state_response_route_mask,
        )

    def loss(
        self,
        clean: torch.Tensor,
        background_latent: torch.Tensor,
        mask: torch.Tensor,
        source_ids: torch.Tensor,
        state_ids: torch.Tensor,
        *,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if timesteps is None:
            timesteps = torch.randint(
                0, len(self.diffusion.betas), (len(clean),), device=clean.device
            )
        if noise is None:
            noise = torch.randn_like(clean)
        if mask.shape[-2:] != clean.shape[-2:]:
            mask = F.interpolate(mask, size=clean.shape[-2:], mode="nearest")
        mask = (mask >= 0.5).to(dtype=clean.dtype)
        condition = self.conditioner(source_ids, state_ids)
        noisy = self.diffusion.q_sample(clean, timesteps, noise)
        prediction = self._predict(
            noisy, timesteps, background_latent, mask, condition
        )
        target = self.diffusion.training_target(
            clean, timesteps, noise, self.prediction_type
        )
        per_pixel = (prediction - target).square().mean(dim=1, keepdim=True)
        # The full latent remains supervised because decoder receptive fields
        # cross the mask boundary; a dilated support receives the dominant weight.
        support = F.max_pool2d(mask, kernel_size=7, stride=1, padding=3)
        weights = 0.25 + 0.75 * support
        weighted = (per_pixel * weights).sum() / weights.sum().clamp_min(1.0)
        inside = (per_pixel * support).sum() / support.sum().clamp_min(1.0)
        outside = (per_pixel * (1.0 - support)).sum() / (
            (1.0 - support).sum().clamp_min(1.0)
        )
        return weighted, {
            "weighted_diffusion_mse": weighted.detach(),
            "inside_diffusion_mse": inside.detach(),
            "outside_diffusion_mse": outside.detach(),
        }

    def predict_x0_from_clean(
        self,
        clean: torch.Tensor,
        background_latent: torch.Tensor,
        mask: torch.Tensor,
        source_ids: torch.Tensor,
        state_ids: torch.Tensor,
        *,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        anchor_state_ids: torch.Tensor | None = None,
        target_state_class_ids: torch.Tensor | None = None,
        anchor_state_class_ids: torch.Tensor | None = None,
        state_feature_timestep_max: int | None = None,
        state_feature_target_class_gate: int | None = None,
    ) -> torch.Tensor:
        """Predict x0 for a supplied intervention while retaining gradients.

        This path is deliberately separate from the factual diffusion loss. It
        lets a frozen image-space judge supervise counterfactual factor tokens
        without treating an unrelated factual image as the pixel target of the
        intervention.
        """
        if timesteps.shape != (len(clean),):
            raise ValueError("timesteps must have shape [batch]")
        if noise.shape != clean.shape:
            raise ValueError("noise must match the clean latent shape")
        if mask.shape[-2:] != clean.shape[-2:]:
            mask = F.interpolate(mask, size=clean.shape[-2:], mode="nearest")
        mask = (mask >= 0.5).to(dtype=clean.dtype)
        condition = self.conditioner(source_ids, state_ids)
        factor_delta_context, factor_delta_mask = self._state_attention_delta(
            source_ids,
            state_ids,
            anchor_state_ids,
            condition,
        )
        if self.has_state_feature_delta:
            class_values = (target_state_class_ids, anchor_state_class_ids)
            if any(value is not None for value in class_values) and not all(
                value is not None for value in class_values
            ):
                raise ValueError(
                    "target and anchor state class IDs must be supplied together"
                )
            state_delta_route_mask = (
                target_state_class_ids.ne(anchor_state_class_ids)
                if target_state_class_ids is not None
                and anchor_state_class_ids is not None
                else None
            )
            if (
                state_delta_route_mask is not None
                and state_feature_target_class_gate is not None
            ):
                state_delta_route_mask = state_delta_route_mask & target_state_class_ids.eq(
                    int(state_feature_target_class_gate)
                )
        else:
            state_delta_route_mask = None
        target_state_response_route_mask = (
            target_state_class_ids.ne(anchor_state_class_ids)
            if self.has_target_state_response
            and target_state_class_ids is not None
            and anchor_state_class_ids is not None
            else None
        )
        if state_feature_timestep_max is not None:
            if not 0 <= state_feature_timestep_max < len(self.diffusion.betas):
                raise ValueError("state feature timestep maximum is invalid")
            if state_delta_route_mask is not None:
                state_delta_route_mask = state_delta_route_mask & timesteps.le(
                    state_feature_timestep_max
                )
        noisy = self.diffusion.q_sample(clean, timesteps, noise)
        prediction = self._predict(
            noisy,
            timesteps,
            background_latent,
            mask,
            condition,
            factor_delta_context,
            factor_delta_mask,
            target_state_class_ids,
            anchor_state_class_ids,
            state_delta_route_mask,
            target_state_response_route_mask,
        )
        return self.diffusion.model_output_to_x0_eps(
            noisy,
            timesteps,
            prediction,
            self.prediction_type,
        )[0]

    def forward(
        self,
        clean: torch.Tensor,
        background_latent: torch.Tensor,
        mask: torch.Tensor,
        source_ids: torch.Tensor,
        state_ids: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        counterfactual_state_ids: torch.Tensor | None = None,
        counterfactual_anchor_state_ids: torch.Tensor | None = None,
        counterfactual_target_state_class_ids: torch.Tensor | None = None,
        counterfactual_anchor_state_class_ids: torch.Tensor | None = None,
        counterfactual_timesteps: torch.Tensor | None = None,
        counterfactual_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, diagnostics = self.loss(
            clean,
            background_latent,
            mask,
            source_ids,
            state_ids,
            timesteps=timesteps,
            noise=noise,
        )
        supplied = (
            counterfactual_state_ids,
            counterfactual_timesteps,
            counterfactual_noise,
        )
        if any(value is not None for value in supplied):
            if not all(value is not None for value in supplied):
                raise ValueError(
                    "counterfactual state, timesteps and noise must be supplied together"
                )
            assert counterfactual_state_ids is not None
            assert counterfactual_timesteps is not None
            assert counterfactual_noise is not None
            if (
                self.has_state_attention_delta
                and counterfactual_anchor_state_ids is None
            ):
                raise ValueError(
                    "state-attention counterfactuals require anchor state IDs"
                )
            if self.has_state_feature_delta and (
                counterfactual_target_state_class_ids is None
                or counterfactual_anchor_state_class_ids is None
            ):
                raise ValueError(
                    "state-feature counterfactuals require target and anchor class IDs"
                )
            diagnostics["counterfactual_x0"] = self.predict_x0_from_clean(
                clean,
                background_latent,
                mask,
                source_ids,
                counterfactual_state_ids,
                timesteps=counterfactual_timesteps,
                noise=counterfactual_noise,
                anchor_state_ids=counterfactual_anchor_state_ids,
                target_state_class_ids=counterfactual_target_state_class_ids,
                anchor_state_class_ids=counterfactual_anchor_state_class_ids,
            )
        return loss, diagnostics

    @torch.no_grad()
    def sample(
        self,
        background_latent: torch.Tensor,
        mask: torch.Tensor,
        source_ids: torch.Tensor,
        state_ids: torch.Tensor,
        *,
        steps: int,
        generator: torch.Generator,
        anchor_state_ids: torch.Tensor | None = None,
        target_state_class_ids: torch.Tensor | None = None,
        anchor_state_class_ids: torch.Tensor | None = None,
        state_feature_timestep_max: int | None = None,
        state_feature_target_class_gate: int | None = None,
    ) -> torch.Tensor:
        if mask.shape[-2:] != background_latent.shape[-2:]:
            mask = F.interpolate(
                mask, size=background_latent.shape[-2:], mode="nearest"
            )
        mask = (mask >= 0.5).to(dtype=background_latent.dtype)
        condition = self.conditioner(source_ids, state_ids)
        factor_delta_context, factor_delta_mask = self._state_attention_delta(
            source_ids,
            state_ids,
            anchor_state_ids,
            condition,
        )
        if self.has_state_feature_delta:
            if (target_state_class_ids is None) != (anchor_state_class_ids is None):
                raise ValueError(
                    "target and anchor state class IDs must be supplied together"
                )
            state_delta_route_mask = (
                target_state_class_ids.ne(anchor_state_class_ids)
                if target_state_class_ids is not None
                and anchor_state_class_ids is not None
                else None
            )
            if (
                state_delta_route_mask is not None
                and state_feature_target_class_gate is not None
            ):
                state_delta_route_mask = state_delta_route_mask & target_state_class_ids.eq(
                    int(state_feature_target_class_gate)
                )
        else:
            state_delta_route_mask = None
        target_state_response_route_mask = (
            target_state_class_ids.ne(anchor_state_class_ids)
            if self.has_target_state_response
            and target_state_class_ids is not None
            and anchor_state_class_ids is not None
            else None
        )
        if state_feature_timestep_max is not None and not (
            0 <= state_feature_timestep_max < len(self.diffusion.betas)
        ):
            raise ValueError("state feature timestep maximum is invalid")

        def denoise(
            noisy: torch.Tensor,
            timesteps: torch.Tensor,
            current: tuple[torch.Tensor, torch.Tensor],
        ) -> torch.Tensor:
            current_state_route = state_delta_route_mask
            if (
                current_state_route is not None
                and state_feature_timestep_max is not None
            ):
                current_state_route = current_state_route & timesteps.le(
                    state_feature_timestep_max
                )
            return self._predict(
                noisy,
                timesteps,
                background_latent,
                mask,
                current,
                factor_delta_context,
                factor_delta_mask,
                target_state_class_ids,
                anchor_state_class_ids,
                current_state_route,
                target_state_response_route_mask,
            )

        return self.diffusion.ddim_sample(
            denoise,
            tuple(background_latent.shape),
            condition,
            cfg_scale=1.0,
            num_inference_steps=steps,
            null_cond=None,
            generator=generator,
            prediction_type=self.prediction_type,
        )


def load_original_v4_high_vae(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[LatentAutoEncoder, dict[str, Any]]:
    payload = load_checkpoint(str(checkpoint_path), map_location="cpu")
    cfg = config_from_checkpoint(payload)
    vae = LatentAutoEncoder(
        3,
        cfg.latent_channels,
        base_ch=cfg.vae_base_ch,
        ch_mult=cfg.vae_ch_mult,
        num_heads=cfg.num_heads,
        gradient_checkpointing=False,
    )
    source = payload.get("model", payload)
    vae_state = {
        key.removeprefix("vae_high."): value
        for key, value in source.items()
        if key.startswith("vae_high.")
    }
    if not vae_state:
        raise ValueError("original checkpoint does not contain vae_high weights")
    vae.load_state_dict(vae_state, strict=True)
    vae.requires_grad_(False).eval().to(device)
    calibration = {
        "enabled": bool(cfg.latent_normalization),
        "mean": tuple(float(value) for value in cfg.latent_mean_high),
        "std": tuple(float(value) for value in cfg.latent_std_high),
        "latent_channels": int(cfg.latent_channels),
        "vae_base_ch": int(cfg.vae_base_ch),
        "vae_ch_mult": tuple(int(value) for value in cfg.vae_ch_mult),
        "num_heads": int(cfg.num_heads),
    }
    return vae, calibration


def normalize_high_latent(
    latent: torch.Tensor, calibration: dict[str, Any]
) -> torch.Tensor:
    if not calibration["enabled"]:
        return latent
    mean = latent.new_tensor(calibration["mean"])[None, :, None, None]
    std = latent.new_tensor(calibration["std"])[None, :, None, None]
    return (latent - mean) / std


def denormalize_high_latent(
    latent: torch.Tensor, calibration: dict[str, Any]
) -> torch.Tensor:
    if not calibration["enabled"]:
        return latent
    mean = latent.new_tensor(calibration["mean"])[None, :, None, None]
    std = latent.new_tensor(calibration["std"])[None, :, None, None]
    return latent * std + mean


def latent_calibrations_match(
    left: dict[str, Any], right: dict[str, Any]
) -> bool:
    """Compare calibration values independent of JSON list/tuple encoding."""
    scalar_fields = (
        "enabled",
        "latent_channels",
        "vae_base_ch",
        "num_heads",
    )
    if any(left.get(key) != right.get(key) for key in scalar_fields):
        return False
    if tuple(int(value) for value in left.get("vae_ch_mult", ())) != tuple(
        int(value) for value in right.get("vae_ch_mult", ())
    ):
        return False
    for key in ("mean", "std"):
        left_values = torch.as_tensor(left.get(key), dtype=torch.float64)
        right_values = torch.as_tensor(right.get(key), dtype=torch.float64)
        if left_values.shape != right_values.shape or not bool(
            torch.equal(left_values, right_values)
        ):
            return False
    return True


def initialize_latent_foreground_state_attention_delta(
    checkpoint: str | Path,
    *,
    rank: int,
) -> tuple[MaskConditionedLatentForegroundDiffusion, dict[str, Any]]:
    """Add zero-initialized state K/V adapters to a frozen Stage-27a model."""
    if rank <= 0:
        raise ValueError("state attention delta rank must be positive")
    base, payload = load_latent_foreground_diffusion(checkpoint)
    if payload.get("schema") != "exp4.latent_foreground_diffusion.stage27a.checkpoint.v1":
        raise ValueError("state attention delta must initialize from Stage-27a")
    config = dict(base.model_config)
    if config.get("state_attention_delta_rank") is not None:
        raise ValueError("initial checkpoint already contains a state adapter")
    config["channel_mult"] = tuple(config["channel_mult"])
    config["state_attention_delta_rank"] = int(rank)
    model = MaskConditionedLatentForegroundDiffusion(**config)
    incompatible = model.load_state_dict(payload["model"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            "unexpected keys while adding state attention delta: "
            + ", ".join(incompatible.unexpected_keys)
        )
    missing = set(incompatible.missing_keys)
    expected = {
        name
        for name in model.state_dict()
        if ".factor_delta_" in name
    }
    if missing != expected or not expected:
        raise ValueError(
            "state attention delta initialization mismatch: "
            f"missing={sorted(missing)} expected={sorted(expected)}"
        )
    return model, payload


def initialize_latent_foreground_state_feature_delta(
    checkpoint: str | Path,
    *,
    num_classes: int,
    allow_stage28a_source: bool = False,
) -> tuple[MaskConditionedLatentForegroundDiffusion, dict[str, Any]]:
    """Add zero-initialized state feature-FiLM experts to Stage-27a."""
    if num_classes <= 1:
        raise ValueError("state feature delta needs at least two classes")
    base, payload = load_latent_foreground_diffusion(checkpoint)
    allowed_schemas = {"exp4.latent_foreground_diffusion.stage27a.checkpoint.v1"}
    if allow_stage28a_source:
        allowed_schemas.add("exp4.latent_foreground_diffusion.stage28a.checkpoint.v1")
    if payload.get("schema") not in allowed_schemas:
        raise ValueError("state feature delta source checkpoint is not supported")
    config = dict(base.model_config)
    if config.get("state_attention_delta_rank") is not None:
        raise ValueError("initial checkpoint already contains a state attention adapter")
    if config.get("state_feature_delta_classes") is not None:
        raise ValueError("initial checkpoint already contains a state feature adapter")
    config["channel_mult"] = tuple(config["channel_mult"])
    config["state_feature_delta_classes"] = int(num_classes)
    model = MaskConditionedLatentForegroundDiffusion(**config)
    incompatible = model.load_state_dict(payload["model"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            "unexpected keys while adding state feature delta: "
            + ", ".join(incompatible.unexpected_keys)
        )
    missing = set(incompatible.missing_keys)
    expected = {
        name for name in model.state_dict() if "state_delta_affines" in name
    }
    if missing != expected or not expected:
        raise ValueError(
            "state feature delta initialization mismatch: "
            f"missing={sorted(missing)} expected={sorted(expected)}"
        )
    return model, payload


def initialize_latent_foreground_target_state_response(
    checkpoint: str | Path,
    *,
    target_class: int,
) -> tuple[MaskConditionedLatentForegroundDiffusion, dict[str, Any]]:
    """Add a zero-initialized target-only response to frozen Stage-36c.

    Unlike the target-minus-anchor adapter, this module is bypassed for every
    other target class.  It can therefore learn the missing 3T3 appearance
    without changing RPE edits that happen to use 3T3 as their anchor.
    """
    base, payload = load_latent_foreground_diffusion(checkpoint)
    if payload.get("schema") != "exp4.latent_foreground_diffusion.stage36b.checkpoint.v1":
        raise ValueError("target response must be initialized from Stage-36b/36c")
    config = dict(base.model_config)
    if config.get("target_state_response_class") is not None:
        raise ValueError("initial checkpoint already contains a target response")
    config["channel_mult"] = tuple(config["channel_mult"])
    config["target_state_response_class"] = int(target_class)
    model = MaskConditionedLatentForegroundDiffusion(**config)
    incompatible = model.load_state_dict(payload["model"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            "unexpected keys while adding target response: "
            + ", ".join(incompatible.unexpected_keys)
        )
    missing = set(incompatible.missing_keys)
    expected = {
        name
        for name in model.state_dict()
        if "target_state_response_affines" in name
    }
    if missing != expected or not expected:
        raise ValueError(
            "target response initialization mismatch: "
            f"missing={sorted(missing)} expected={sorted(expected)}"
        )
    return model, payload


def load_latent_foreground_diffusion(
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[MaskConditionedLatentForegroundDiffusion, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    supported_schemas = {
        "exp4.latent_foreground_diffusion.stage27a.checkpoint.v1",
        "exp4.latent_foreground_diffusion.stage28a.checkpoint.v1",
        "exp4.latent_foreground_diffusion.stage28b.checkpoint.v1",
        "exp4.latent_foreground_diffusion.stage28c.checkpoint.v1",
        "exp4.latent_foreground_diffusion.stage29a.checkpoint.v1",
        "exp4.latent_foreground_diffusion.stage36b.checkpoint.v1",
        "exp4.latent_foreground_diffusion.stage42a.checkpoint.v1",
        "exp4.p11.full_backbone.rpe.checkpoint.v1",
        "exp4.p11.full_backbone.3t3.checkpoint.v1",
    }
    if payload.get("schema") not in supported_schemas:
        raise ValueError(f"unsupported Stage-27a checkpoint: {checkpoint}")
    config = dict(payload["model_config"])
    config["channel_mult"] = tuple(config["channel_mult"])
    model = MaskConditionedLatentForegroundDiffusion(**config)
    model.load_state_dict(payload["model"], strict=True)
    return model, payload
