from __future__ import annotations

from collections.abc import Callable
import hashlib
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .config import MSCDMConfig
from .diffusion import GaussianDiffusion
from .latent_vae import LatentAutoEncoder, vae_kl_loss, vae_reconstruction_loss
from .mask_surrogate import mask_surrogate_loss
from .text_encoder import ContrastiveHead, MultiScaleTextEncoder, contrastive_loss
from .unet import ScaleUNet, timestep_embedding


NUCLEUS_GATED_CONDITIONING_MODES = {
    "subject_state_low_nucleus",
    "subject_state_low_nucleus_adapter",
    "subject_state_low_nucleus_multiscale_adapter",
    "subject_state_low_nucleus_prompt_adapters",
    "hierarchical_nucleus_distill_prompt_adapters",
    "hierarchical_nucleus_distill_output_adapters",
    "subject_state_low_nucleus_temporal_multiscale_adapter",
}
NUCLEUS_DISTILL_PROMPT_ADAPTER_MODE = (
    "hierarchical_nucleus_distill_prompt_adapters"
)
NUCLEUS_DISTILL_OUTPUT_ADAPTER_MODE = (
    "hierarchical_nucleus_distill_output_adapters"
)
NUCLEUS_TEMPORAL_ADAPTER_MODE = (
    "subject_state_low_nucleus_temporal_multiscale_adapter"
)
FACTOR_PROMPT_ADAPTER_MODE = "hierarchical_factor_prompt_adapters"
FACTOR_DELTA_ADAPTER_MODE = "hierarchical_factor_delta_adapters"
FACTOR_ATTENTION_DELTA_ADAPTER_MODE = (
    "hierarchical_factor_attention_delta_adapters"
)
RGB_FACTOR_OUTPUT_ADAPTER_MODE = (
    "hierarchical_rgb_factor_output_adapters"
)
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
SEMANTIC_MULTISCALE_MODE = "semantic_multiscale"
SEMANTIC_STATE_ALLSCALE_MODE = "semantic_state_allscale"
SEMANTIC_STATE_SEQ_LOW_MODE = "semantic_state_seq_low"
SEMANTIC_STATE_FACTOR_RESIDUAL_POOL_MODE = (
    "semantic_state_factor_residual_pool"
)
SEMANTIC_STATE_FACTOR_PARTITION_ATTENTION_MODE = (
    "semantic_state_factor_partition_attention"
)
SEMANTIC_STATE_FACTOR_ISOLATED_COMBINED_MODE = (
    "semantic_state_factor_isolated_combined"
)
SEMANTIC_STATE_SPECIFIC_RESIDUAL_MODE = (
    "semantic_state_specific_residual"
)

CONDITION_MASK_PATTERNS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("full", (0, 0, 0)),
    ("drop_state", (0, 1, 0)),
    ("state_only", (1, 0, 1)),
    ("all_null", (1, 1, 1)),
    ("drop_subject", (1, 0, 0)),
    ("drop_background", (0, 0, 1)),
    ("subject_only", (0, 1, 1)),
    ("background_only", (1, 1, 0)),
)
STRATIFIED_CONDITION_MASK_COUNTS_160: dict[str, int] = {
    "full": 64,
    "drop_state": 24,
    "state_only": 24,
    "all_null": 8,
    "drop_subject": 10,
    "drop_background": 10,
    "subject_only": 10,
    "background_only": 10,
}


def build_isolated_condition_drop_masks(
    *,
    mode: str,
    batch_size: int,
    global_step: int,
    rank: int,
    base_seed: int,
    device: torch.device,
    independent_drop_prob: float = 0.325,
) -> torch.Tensor:
    """Build condition masks without advancing the base diffusion RNG stream.

    The stratified route uses a deterministic 160-sample permutation and five
    consecutive per-rank batches of 32.  Its exact counts are therefore
    64/24/24/8/10/10/10/10 for the eight preregistered mask categories.
    """

    if mode not in {"independent_matched", "stratified"}:
        raise ValueError(f"unsupported isolated condition-mask mode: {mode!r}")
    if batch_size <= 0 or global_step < 0 or rank < 0 or base_seed < 0:
        raise ValueError("condition-mask batch/step/rank/seed must be non-negative")
    generator = torch.Generator(device="cpu")
    if mode == "independent_matched":
        if not 0.0 <= independent_drop_prob <= 1.0:
            raise ValueError("independent condition drop probability must be in [0, 1]")
        generator.manual_seed(base_seed + global_step * 1_000_003 + rank)
        masks = (
            torch.rand((batch_size, 3), generator=generator)
            < independent_drop_prob
        ).to(torch.int64)
        return masks.to(device=device)

    cycle_size = sum(STRATIFIED_CONDITION_MASK_COUNTS_160.values())
    if cycle_size != 160 or batch_size != 32:
        raise ValueError("stratified v1 requires per-rank batch_size=32 and cycle=160")
    cycle_steps = cycle_size // batch_size
    cycle_index, within_cycle = divmod(global_step, cycle_steps)
    generator.manual_seed(base_seed + cycle_index * 1_000_003 + rank)
    patterns_by_name = dict(CONDITION_MASK_PATTERNS)
    cycle = torch.tensor(
        [
            patterns_by_name[name]
            for name, _pattern in CONDITION_MASK_PATTERNS
            for _ in range(STRATIFIED_CONDITION_MASK_COUNTS_160[name])
        ],
        dtype=torch.int64,
    )
    cycle = cycle[torch.randperm(cycle_size, generator=generator)]
    start = within_cycle * batch_size
    return cycle[start : start + batch_size].to(device=device)


def cfg_scale_is_one(scale: float | torch.Tensor) -> bool:
    if isinstance(scale, torch.Tensor):
        return not bool(torch.any(scale != 1.0).item())
    return float(scale) == 1.0


def combine_factor_cfg_predictions(
    full_prediction: torch.Tensor,
    factor_predictions: tuple[
        tuple[torch.Tensor, float | torch.Tensor], ...
    ],
) -> torch.Tensor:
    """Apply independent-factor CFG around the fully conditioned prediction."""
    guided = full_prediction
    batch_size = full_prediction.shape[0]
    for without_factor, scale in factor_predictions:
        if without_factor.shape != full_prediction.shape:
            raise ValueError("factor CFG predictions must have identical shapes")
        if isinstance(scale, torch.Tensor):
            if scale.ndim == 0:
                multiplier = scale.to(
                    device=full_prediction.device,
                    dtype=full_prediction.dtype,
                ) - 1.0
            elif scale.shape == (batch_size,):
                multiplier = scale.to(
                    device=full_prediction.device,
                    dtype=full_prediction.dtype,
                )[:, None, None, None] - 1.0
            else:
                raise ValueError(
                    "factor CFG scale tensor must be scalar or shape [batch]"
                )
        else:
            multiplier = float(scale) - 1.0
        guided = guided + multiplier * (full_prediction - without_factor)
    return guided


def population_mean_logits_and_targets(
    logits: torch.Tensor,
    targets: torch.Tensor,
    population_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate same-target image logits without weakening label checks."""
    if logits.ndim != 2:
        raise ValueError("population logits must have shape [batch,classes]")
    if targets.ndim != 1 or len(targets) != len(logits):
        raise ValueError("population targets must have shape [batch]")
    if population_size <= 0:
        raise ValueError("population size must be positive")
    if len(logits) % population_size != 0:
        raise ValueError(
            "population batch is not divisible by the configured size"
        )
    grouped_targets = targets.view(-1, population_size)
    if not bool((grouped_targets == grouped_targets[:, :1]).all()):
        raise ValueError("each population must share one target class")
    grouped_logits = logits.view(
        -1,
        population_size,
        logits.shape[-1],
    ).mean(dim=1)
    return grouped_logits, grouped_targets[:, 0]


class NucleusLowResidualAdapter(nn.Module):
    """Small zero-initialized residual on the nucleus low-branch prediction."""

    def __init__(
        self,
        latent_channels: int,
        text_dim: int,
        hidden_channels: int = 32,
        time_dim: int = 256,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.input_projection = nn.Conv2d(
            latent_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.condition_projection = nn.Sequential(
            nn.Linear(text_dim + time_dim, hidden_channels * 2),
            nn.SiLU(),
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
        )
        self.output_projection = nn.Conv2d(
            hidden_channels,
            latent_channels,
            kernel_size=3,
            padding=1,
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        prediction: torch.Tensor,
        timesteps: torch.Tensor,
        condition_pool: torch.Tensor,
    ) -> torch.Tensor:
        time_condition = timestep_embedding(timesteps, self.time_dim).to(
            dtype=condition_pool.dtype
        )
        scale, shift = self.condition_projection(
            torch.cat([condition_pool, time_condition], dim=-1)
        ).chunk(2, dim=-1)
        hidden = self.input_projection(prediction)
        hidden = F.silu(
            hidden * (1.0 + scale[:, :, None, None])
            + shift[:, :, None, None]
        )
        return self.output_projection(hidden)


class SpatialSubjectInputAdapter(nn.Module):
    """Zero-initialized spatial condition injected before low-branch synthesis."""

    def __init__(self, latent_channels: int, hidden_channels: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        mask: torch.Tensor,
        spatial_size: tuple[int, int],
    ) -> torch.Tensor:
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("spatial subject mask must have shape [B,1,H,W]")
        resized = F.interpolate(
            mask.float(),
            size=spatial_size,
            mode="bilinear",
            align_corners=False,
        )
        return self.network(resized * 2.0 - 1.0)


class MorphologyLatentBank(nn.Module):
    """Source-neutral, class-indexed multichannel low-latent residuals.

    Explicit round/oval/elongated class IDs select three shared learned
    coordinates. Instance masks never enter the generator. Tokens are zero
    initialized so an untrained bank is exactly equivalent to the frozen base.
    """

    labels = ("round", "oval", "elongated")

    def __init__(self, latent_channels: int, spatial_size: int = 16):
        super().__init__()
        self.tokens = nn.Parameter(
            torch.zeros(3, latent_channels, spatial_size, spatial_size)
        )

    def forward(
        self,
        class_ids: torch.Tensor,
        spatial_size: tuple[int, int],
    ) -> torch.Tensor:
        if class_ids.ndim != 1:
            raise ValueError("morphology class IDs must have shape [batch]")
        if not bool(((class_ids >= 0) & (class_ids < 3)).all()):
            raise ValueError("morphology class IDs must be in [0, 2]")
        selected = self.tokens[class_ids.long()]
        if selected.shape[-2:] != spatial_size:
            selected = F.interpolate(
                selected,
                size=spatial_size,
                mode="bilinear",
                align_corners=False,
            )
        return selected


class PromptGatedTokenAdapter(nn.Module):
    """Zero-initialized low-rank adapter on routed text tokens and pooled text."""

    def __init__(self, text_dim: int, rank: int = 16):
        super().__init__()
        if rank <= 0:
            raise ValueError("prompt adapter rank must be positive")
        self.down = nn.Linear(text_dim, rank, bias=False)
        self.up = nn.Linear(rank, text_dim, bias=False)
        self.gate = nn.Linear(text_dim, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        sequence: torch.Tensor,
        pooled: torch.Tensor,
        route_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence.ndim != 3 or pooled.ndim != 2:
            raise ValueError("prompt adapter expects [B,L,D] sequence and [B,D] pooled")
        if sequence.shape[0] != pooled.shape[0]:
            raise ValueError("prompt adapter sequence/pooled batch sizes differ")
        if route_mask.ndim != 1 or route_mask.shape[0] != sequence.shape[0]:
            raise ValueError("prompt adapter route_mask must have shape [batch]")
        strength = torch.sigmoid(self.gate(pooled))
        strength = strength * route_mask.to(
            device=pooled.device,
            dtype=pooled.dtype,
        )[:, None]
        sequence_delta = self.up(F.silu(self.down(sequence)))
        pooled_delta = self.up(F.silu(self.down(pooled)))
        return (
            sequence + sequence_delta * strength[:, None, :],
            pooled + pooled_delta * strength,
        )


class FactorDeltaTokenAdapter(nn.Module):
    """Edit-only adapter driven by a single factor's target-anchor delta."""

    def __init__(self, text_dim: int, rank: int = 16):
        super().__init__()
        if rank <= 0:
            raise ValueError("factor delta adapter rank must be positive")
        self.down = nn.Linear(text_dim, rank, bias=False)
        self.up = nn.Linear(rank, text_dim, bias=False)
        self.gate = nn.Linear(text_dim, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        target_sequence: torch.Tensor,
        target_pooled: torch.Tensor,
        anchor_sequence: torch.Tensor,
        anchor_pooled: torch.Tensor,
        route_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target_sequence.shape != anchor_sequence.shape:
            raise ValueError("factor delta sequence shapes differ")
        if target_pooled.shape != anchor_pooled.shape:
            raise ValueError("factor delta pooled shapes differ")
        if target_sequence.ndim != 3 or target_pooled.ndim != 2:
            raise ValueError(
                "factor delta adapter expects [B,L,D] sequence and [B,D] pooled"
            )
        if route_mask.ndim != 1 or route_mask.shape[0] != target_sequence.shape[0]:
            raise ValueError("factor delta route_mask must have shape [batch]")
        mask = route_mask.to(device=target_sequence.device, dtype=torch.bool)
        if not bool(mask.any()):
            return target_sequence, target_pooled
        sequence_difference = target_sequence - anchor_sequence
        pooled_difference = target_pooled - anchor_pooled
        sequence_delta = self.up(F.silu(self.down(sequence_difference)))
        pooled_delta = self.up(F.silu(self.down(pooled_difference)))
        strength = torch.sigmoid(self.gate(pooled_difference))
        adapted_sequence = target_sequence + sequence_delta * strength[:, None, :]
        adapted_pooled = target_pooled + pooled_delta * strength
        return (
            torch.where(mask[:, None, None], adapted_sequence, target_sequence),
            torch.where(mask[:, None], adapted_pooled, target_pooled),
        )


class ContextGatedTokenAdapter(nn.Module):
    """Low-rank token adapter with external prompt context and exact bypass."""

    def __init__(self, text_dim: int, rank: int = 16):
        super().__init__()
        if rank <= 0:
            raise ValueError("context prompt adapter rank must be positive")
        self.down = nn.Linear(text_dim, rank, bias=False)
        self.context_down = nn.Linear(text_dim, rank, bias=False)
        self.up = nn.Linear(rank, text_dim, bias=False)
        self.gate = nn.Linear(text_dim, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        sequence: torch.Tensor,
        pooled: torch.Tensor,
        context: torch.Tensor,
        route_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence.ndim != 3 or pooled.ndim != 2 or context.ndim != 2:
            raise ValueError(
                "context prompt adapter expects [B,L,D] sequence and "
                "[B,D] pooled/context"
            )
        if sequence.shape[0] != pooled.shape[0] or pooled.shape != context.shape:
            raise ValueError(
                "context prompt adapter sequence/pooled/context shapes differ"
            )
        if route_mask.ndim != 1 or route_mask.shape[0] != sequence.shape[0]:
            raise ValueError(
                "context prompt adapter route_mask must have shape [batch]"
            )
        mask = route_mask.to(device=sequence.device, dtype=torch.bool)
        if not bool(mask.any()):
            return sequence, pooled
        context_hidden = self.context_down(context)
        sequence_delta = self.up(
            F.silu(self.down(sequence) + context_hidden[:, None, :])
        )
        pooled_delta = self.up(
            F.silu(self.down(pooled) + context_hidden)
        )
        strength = torch.sigmoid(self.gate(context))
        adapted_sequence = sequence + sequence_delta * strength[:, None, :]
        adapted_pooled = pooled + pooled_delta * strength
        return (
            torch.where(mask[:, None, None], adapted_sequence, sequence),
            torch.where(mask[:, None], adapted_pooled, pooled),
        )


class ZeroInitializedLinear(nn.Module):
    """Bias-free linear map whose construction does not consume model RNG."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.weight)


class FactorResidualPooledComposition(nn.Module):
    """Zero-initialized factor-specific residuals on the allscale pooled path.

    The exact allscale mean is retained as the base function.  Each factor is
    normalized independently and compared with its learned null vector before
    a branch-specific zero-initialized linear projection is added.  A dropped
    factor therefore contributes exactly zero residual while the original
    allscale null-conditioned base remains unchanged.
    """

    ROUTES: tuple[tuple[str, str], ...] = (
        ("low", "subject"),
        ("low", "state"),
        ("low", "background"),
        ("mid", "state"),
        ("mid", "background"),
        ("high", "state"),
        ("high", "background"),
    )

    def __init__(self, dim: int):
        super().__init__()
        if dim <= 0:
            raise ValueError("factor residual pooled dimension must be positive")
        self.dim = int(dim)
        self.projections = nn.ModuleDict(
            {
                f"{branch}_{factor}": ZeroInitializedLinear(dim)
                for branch, factor in self.ROUTES
            }
        )

    def forward(
        self,
        branch: str,
        base: torch.Tensor,
        factor_pools: dict[str, torch.Tensor],
        null_pools: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if base.ndim != 2 or base.shape[1] != self.dim:
            raise ValueError("factor residual pooled base must have shape [B,D]")
        output = base
        route_factors = [
            factor for active_branch, factor in self.ROUTES
            if active_branch == branch
        ]
        if not route_factors:
            raise ValueError(f"unsupported factor residual pooled branch: {branch!r}")
        for factor in route_factors:
            pooled = factor_pools[factor]
            null = null_pools[factor]
            if pooled.shape != base.shape:
                raise ValueError(
                    f"{branch}/{factor} pooled shape {tuple(pooled.shape)} "
                    f"does not match base {tuple(base.shape)}"
                )
            if null.ndim != 2 or null.shape[1] != self.dim:
                raise ValueError(
                    f"{branch}/{factor} null pooled must have shape [1 or B,D]"
                )
            null = null.expand_as(pooled)
            normalized_delta = F.layer_norm(
                pooled, (self.dim,)
            ) - F.layer_norm(null, (self.dim,))
            output = output + self.projections[
                f"{branch}_{factor}"
            ](normalized_delta)
        return output


def route_condition_embeddings(
    conditions: tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ],
    mode: str,
    subject_state_low_mask: torch.Tensor | None = None,
    factor_residual_pool: FactorResidualPooledComposition | None = None,
    factor_null_pools: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ] | None = None,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    """Route branch embeddings, including optional model-owned composition."""
    if mode in {
        "hierarchical",
        FACTOR_PROMPT_ADAPTER_MODE,
        FACTOR_DELTA_ADAPTER_MODE,
        FACTOR_ATTENTION_DELTA_ADAPTER_MODE,
        RGB_FACTOR_OUTPUT_ADAPTER_MODE,
        SPATIAL_SUBJECT_INPUT_ADAPTER_MODE,
        MORPHOLOGY_LATENT_BANK_MODE,
        MORPHOLOGY_FEATURE_AFFINE_MODE,
        MORPHOLOGY_SPATIAL_AFFINE_MODE,
    }:
        return conditions
    (sub_seq, sub_pool), (state_seq, state_pool), (bg_seq, bg_pool) = conditions
    if mode == "uniform_all_factors":
        # Reuse the three historical CLIP input fields and their learned heads.
        # Each field is tokenized to max_text_len independently before this
        # route, so concatenation retains all 3 * max_text_len positions.
        # No second tokenizer pass or 64-token truncation is allowed here.
        if not (sub_seq.shape[0] == state_seq.shape[0] == bg_seq.shape[0]):
            raise ValueError("uniform_all_factors requires aligned batches")
        if not (sub_seq.shape[2] == state_seq.shape[2] == bg_seq.shape[2]):
            raise ValueError("uniform_all_factors requires aligned token widths")
        shared = (
            torch.cat((sub_seq, state_seq, bg_seq), dim=1),
            (sub_pool + state_pool + bg_pool) / 3.0,
        )
        return shared, shared, shared
    if mode == "subject_state_low":
        return (
            (
                torch.cat([sub_seq, state_seq], dim=1),
                (sub_pool + state_pool) / 2.0,
            ),
            (state_seq, state_pool),
            (bg_seq, bg_pool),
        )
    if mode == SEMANTIC_MULTISCALE_MODE:
        # RxRx1 v3 的 background 表示观察范围而不只是高频风格：它必须从
        # low 开始决定 nucleus/cell/whole-cell 的可见结构，并在后续尺度
        # 保留染色和纹理。state 同时进入 mid/high，避免只靠级联残差传递。
        return (
            (
                torch.cat([sub_seq, bg_seq], dim=1),
                (sub_pool + bg_pool) / 2.0,
            ),
            (
                torch.cat([state_seq, bg_seq], dim=1),
                (state_pool + bg_pool) / 2.0,
            ),
            (
                torch.cat([state_seq, bg_seq], dim=1),
                (state_pool + bg_pool) / 2.0,
            ),
        )
    if mode in {
        SEMANTIC_STATE_ALLSCALE_MODE,
        SEMANTIC_STATE_FACTOR_PARTITION_ATTENTION_MODE,
        SEMANTIC_STATE_SPECIFIC_RESIDUAL_MODE,
    }:
        # 在已修复 background 的路由上，仅把会改变粗形态的 state 前移到 low。
        return (
            (
                torch.cat([sub_seq, state_seq, bg_seq], dim=1),
                (sub_pool + state_pool + bg_pool) / 3.0,
            ),
            (
                torch.cat([state_seq, bg_seq], dim=1),
                (state_pool + bg_pool) / 2.0,
            ),
            (
                torch.cat([state_seq, bg_seq], dim=1),
                (state_pool + bg_pool) / 2.0,
            ),
        )
    if mode in {
        SEMANTIC_STATE_FACTOR_RESIDUAL_POOL_MODE,
        SEMANTIC_STATE_FACTOR_ISOLATED_COMBINED_MODE,
    }:
        if factor_residual_pool is None or factor_null_pools is None:
            raise ValueError(
                "semantic_state_factor_residual_pool requires the model's "
                "factor residual module and learned null pools"
            )
        null_sub_pool, null_state_pool, null_bg_pool = factor_null_pools
        factor_pools = {
            "subject": sub_pool,
            "state": state_pool,
            "background": bg_pool,
        }
        null_pools = {
            "subject": null_sub_pool,
            "state": null_state_pool,
            "background": null_bg_pool,
        }
        low_base = (sub_pool + state_pool + bg_pool) / 3.0
        mid_high_base = (state_pool + bg_pool) / 2.0
        return (
            (
                torch.cat([sub_seq, state_seq, bg_seq], dim=1),
                factor_residual_pool(
                    "low", low_base, factor_pools, null_pools
                ),
            ),
            (
                torch.cat([state_seq, bg_seq], dim=1),
                factor_residual_pool(
                    "mid", mid_high_base, factor_pools, null_pools
                ),
            ),
            (
                torch.cat([state_seq, bg_seq], dim=1),
                factor_residual_pool(
                    "high", mid_high_base, factor_pools, null_pools
                ),
            ),
        )
    if mode == SEMANTIC_STATE_SEQ_LOW_MODE:
        # The state token sequence remains visible at low, but state is removed
        # from the low pooled FiLM condition.  Mid/high are identical to
        # semantic_state_allscale.  This route adds no trainable parameters.
        return (
            (
                torch.cat([sub_seq, state_seq, bg_seq], dim=1),
                (sub_pool + bg_pool) / 2.0,
            ),
            (
                torch.cat([state_seq, bg_seq], dim=1),
                (state_pool + bg_pool) / 2.0,
            ),
            (
                torch.cat([state_seq, bg_seq], dim=1),
                (state_pool + bg_pool) / 2.0,
            ),
        )
    if mode in NUCLEUS_GATED_CONDITIONING_MODES:
        if subject_state_low_mask is None:
            raise ValueError(
                f"{mode} requires subject_state_low_mask"
            )
        if subject_state_low_mask.ndim != 1:
            raise ValueError("subject_state_low_mask must have shape [batch]")
        if subject_state_low_mask.shape[0] != sub_seq.shape[0]:
            raise ValueError(
                "subject_state_low_mask batch size does not match embeddings"
            )
        if sub_seq.shape != state_seq.shape:
            raise ValueError(
                f"{mode} requires equal subject/state sequence "
                "shapes; personalization is not supported for this route"
            )
        mask = subject_state_low_mask.to(device=sub_seq.device, dtype=torch.bool)
        # The temporal expert preserves the original hierarchical v4 condition
        # path. The object mask is consumed only by the late-step adapters.
        if mode in {
            NUCLEUS_TEMPORAL_ADAPTER_MODE,
            NUCLEUS_DISTILL_PROMPT_ADAPTER_MODE,
            NUCLEUS_DISTILL_OUTPUT_ADAPTER_MODE,
        }:
            return conditions
        if not bool(mask.any()):
            return conditions
        if bool(mask.all()):
            return (
                (
                    torch.cat([sub_seq, state_seq], dim=1),
                    (sub_pool + state_pool) / 2.0,
                ),
                (state_seq, state_pool),
                (bg_seq, bg_pool),
            )
        # Appending an identical copy of the subject tokens leaves cross-attention
        # unchanged for whole-cell samples: duplicate K/V entries split the same
        # softmax mass. Nucleus samples instead append the state tokens.
        second_seq = torch.where(mask[:, None, None], state_seq, sub_seq)
        second_pool = torch.where(mask[:, None], state_pool, sub_pool)
        return (
            (
                torch.cat([sub_seq, second_seq], dim=1),
                (sub_pool + second_pool) / 2.0,
            ),
            (state_seq, state_pool),
            (bg_seq, bg_pool),
        )
    if mode != "cumulative":
        raise ValueError(
            "conditioning_mode must be hierarchical, cumulative, "
            "semantic_multiscale, semantic_state_allscale, semantic_state_seq_low, "
            "semantic_state_factor_residual_pool, "
            "semantic_state_factor_partition_attention, "
            "semantic_state_specific_residual, "
            "subject_state_low, subject_state_low_nucleus, or "
            "subject_state_low_nucleus_adapter, or "
            "subject_state_low_nucleus_multiscale_adapter, or "
            "subject_state_low_nucleus_prompt_adapters, "
            "hierarchical_nucleus_distill_prompt_adapters, "
            "hierarchical_nucleus_distill_output_adapters, "
            "subject_state_low_nucleus_temporal_multiscale_adapter, "
            "hierarchical_factor_prompt_adapters, "
            "hierarchical_factor_delta_adapters, "
            "hierarchical_factor_attention_delta_adapters, "
            "hierarchical_rgb_factor_output_adapters, "
            "hierarchical_spatial_subject_input_adapter, "
            "hierarchical_morphology_latent_bank, "
            "hierarchical_morphology_feature_affine, "
            "hierarchical_morphology_spatial_affine, "
            f"got {mode!r}"
        )
    return (
        (sub_seq, sub_pool),
        (
            torch.cat([sub_seq, state_seq], dim=1),
            (sub_pool + state_pool) / 2.0,
        ),
        (
            torch.cat([sub_seq, state_seq, bg_seq], dim=1),
            (sub_pool + state_pool + bg_pool) / 3.0,
        ),
    )


def nucleus_adapter_route_mask(
    mode: str,
    object_mask: torch.Tensor,
    timesteps: torch.Tensor,
    max_timestep: int,
) -> torch.Tensor:
    """Build the per-sample residual-adapter route, including its time gate."""
    if object_mask.ndim != 1 or timesteps.ndim != 1:
        raise ValueError("object_mask and timesteps must have shape [batch]")
    if object_mask.shape != timesteps.shape:
        raise ValueError("object_mask and timesteps must have identical shapes")
    route = object_mask.to(device=timesteps.device, dtype=torch.bool)
    if mode == NUCLEUS_TEMPORAL_ADAPTER_MODE:
        route = route & (timesteps <= int(max_timestep))
    return route


def foreground_mixed_mse_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    foreground_mask: torch.Tensor | None,
    foreground_mix: float,
) -> torch.Tensor:
    """Return per-sample full-field/foreground mixed MSE."""
    if not 0.0 <= foreground_mix < 1.0:
        raise ValueError("foreground_mix must be in [0, 1)")
    squared_error = (prediction - target).square()
    global_per_sample = squared_error.mean(dim=(1, 2, 3))
    if foreground_mask is None or foreground_mix == 0.0:
        return global_per_sample
    if foreground_mask.ndim != 4 or foreground_mask.shape[1] != 1:
        raise ValueError(
            "foreground_mask must have shape [batch, 1, height, width]"
        )
    if foreground_mask.shape[0] != prediction.shape[0]:
        raise ValueError("foreground_mask batch size does not match prediction")
    latent_mask = F.interpolate(
        foreground_mask.to(device=prediction.device, dtype=prediction.dtype),
        size=prediction.shape[-2:],
        mode="area",
    ).clamp(0.0, 1.0)
    mask_mass = latent_mask.sum(dim=(1, 2, 3))
    if bool((mask_mass <= 0).any()):
        raise ValueError("foreground_mask contains an empty sample")
    foreground_per_sample = (
        (squared_error * latent_mask).sum(dim=(1, 2, 3))
        / (mask_mass * squared_error.shape[1])
    )
    return (
        (1.0 - foreground_mix) * global_per_sample
        + foreground_mix * foreground_per_sample
    )


def foreground_mixed_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    foreground_mask: torch.Tensor | None,
    foreground_mix: float,
) -> torch.Tensor:
    """Blend full-field and foreground-only MSE without changing batch weighting."""
    return foreground_mixed_mse_per_sample(
        prediction,
        target,
        foreground_mask,
        foreground_mix,
    ).mean()


def mix_detached_prior_proxy(
    teacher_prior: torch.Tensor,
    predicted_prior: torch.Tensor,
    proxy_mix: float,
) -> torch.Tensor:
    """Expose later branches to model-generated priors without cross-branch gradients."""
    if not 0.0 <= proxy_mix <= 1.0:
        raise ValueError("proxy_mix must be in [0, 1]")
    if teacher_prior.shape != predicted_prior.shape:
        raise ValueError(
            "teacher_prior and predicted_prior must have identical shapes"
        )
    if proxy_mix == 0.0:
        return teacher_prior
    return torch.lerp(
        teacher_prior,
        predicted_prior.detach(),
        proxy_mix,
    )


CASCADE_PRIOR_GRADIENT_MODES = {"off", "stopgrad", "e2e"}


def cascade_prior_probability(epoch_one_based: int) -> float:
    """Return the frozen GPT-Pro-v2 single-step prior schedule."""
    if epoch_one_based <= 0:
        raise ValueError("epoch_one_based must be positive")
    if epoch_one_based <= 2:
        return 0.25
    if epoch_one_based <= 4:
        return 0.50
    if epoch_one_based <= 6:
        return 0.75
    return 1.00


def build_cascade_prior_source_mask(
    *,
    batch_size: int,
    probability: float,
    global_step: int,
    rank: int,
    base_seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Build an exact-quota mask without advancing the base training RNG.

    The same mask is used for low-to-mid and mid-to-high.  Candidate and
    matched control therefore see the same teacher/self-predicted rows.
    """
    if batch_size <= 0:
        raise ValueError("cascade-prior batch size must be positive")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("cascade-prior probability must be in [0, 1]")
    if min(global_step, rank, base_seed) < 0:
        raise ValueError("cascade-prior step, rank and seed must be non-negative")
    selected = int(round(batch_size * probability))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(base_seed + global_step * 1_000_003 + rank)
    permutation = torch.randperm(batch_size, generator=generator)
    mask = torch.zeros(batch_size, dtype=torch.bool)
    mask[permutation[:selected]] = True
    return mask.to(device=device)


def select_single_step_cascade_prior(
    teacher_prior: torch.Tensor,
    predicted_prior: torch.Tensor,
    use_predicted_mask: torch.Tensor,
    gradient_mode: str,
) -> torch.Tensor:
    """Select clean or single-step self-predicted priors per sample.

    This helper is deliberately *not* called on-policy DDIM training: the
    predicted prior comes from one denoising forward on q(z_t | real z_0).
    The e2e and stopgrad routes have identical forward values and differ only
    in whether downstream losses cross the scale boundary.
    """
    if gradient_mode not in {"stopgrad", "e2e"}:
        raise ValueError("cascade-prior gradient_mode must be stopgrad or e2e")
    if teacher_prior.shape != predicted_prior.shape:
        raise ValueError("teacher and predicted cascade priors must match")
    if use_predicted_mask.shape != (teacher_prior.shape[0],):
        raise ValueError("cascade-prior mask must have shape [batch]")
    if use_predicted_mask.dtype != torch.bool:
        raise ValueError("cascade-prior mask must be boolean")
    predicted_value = (
        predicted_prior.detach()
        if gradient_mode == "stopgrad"
        else predicted_prior
    )
    return torch.where(
        use_predicted_mask[:, None, None, None],
        predicted_value,
        teacher_prior,
    )


def counterfactual_direction_and_anchor_loss(
    anchor_prediction: torch.Tensor,
    target_prediction: torch.Tensor,
    teacher_anchor_prediction: torch.Tensor,
    anchor_target: torch.Tensor,
    target_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match the real pair direction while retaining the original source output."""
    shapes = {
        tuple(value.shape)
        for value in (
            anchor_prediction,
            target_prediction,
            teacher_anchor_prediction,
            anchor_target,
            target_target,
        )
    }
    if len(shapes) != 1:
        raise ValueError("counterfactual direction tensors must share one shape")
    target_delta = target_target - anchor_target
    predicted_delta = target_prediction - anchor_prediction
    flat_target = target_delta.flatten(1)
    flat_predicted = predicted_delta.flatten(1)
    target_rms = flat_target.square().mean(dim=1).sqrt().clamp(min=1e-4)
    cosine = 1.0 - F.cosine_similarity(
        flat_predicted,
        flat_target,
        dim=1,
        eps=1e-6,
    )
    normalized_predicted = predicted_delta / target_rms[:, None, None, None]
    normalized_target = target_delta / target_rms[:, None, None, None]
    magnitude = F.smooth_l1_loss(
        normalized_predicted,
        normalized_target,
        reduction="none",
    ).mean(dim=(1, 2, 3))
    direction = (0.5 * cosine + 0.5 * magnitude).mean()
    anchor = F.mse_loss(anchor_prediction, teacher_anchor_prediction)
    return direction, anchor, target_rms.mean()


def counterfactual_target_orthogonal_loss(
    target_prediction: torch.Tensor,
    teacher_target_prediction: torch.Tensor,
    anchor_target: torch.Tensor,
    target_target: torch.Tensor,
) -> torch.Tensor:
    """Keep target-endpoint updates inside the real factor direction.

    The frozen teacher supplies the original target-condition prediction.
    Student changes parallel to the prototype factor direction remain free;
    only the orthogonal residual is penalized.
    """
    shapes = {
        tuple(value.shape)
        for value in (
            target_prediction,
            teacher_target_prediction,
            anchor_target,
            target_target,
        )
    }
    if len(shapes) != 1:
        raise ValueError(
            "counterfactual target-preservation tensors must share one shape"
        )
    direction = (target_target - anchor_target).flatten(1)
    direction_norm = direction.norm(dim=1, keepdim=True).clamp(min=1e-6)
    unit_direction = direction / direction_norm
    residual = (target_prediction - teacher_target_prediction).flatten(1)
    parallel = (residual * unit_direction).sum(dim=1, keepdim=True)
    orthogonal = residual - parallel * unit_direction
    target_rms = direction.square().mean(dim=1).sqrt().clamp(min=1e-4)
    normalized = orthogonal / target_rms[:, None]
    return normalized.square().mean()


def paired_direction_and_center_loss(
    anchor_prediction: torch.Tensor,
    target_prediction: torch.Tensor,
    base_anchor_prediction: torch.Tensor,
    base_target_prediction: torch.Tensor,
    anchor_target: torch.Tensor,
    target_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match a paired RGB intervention without allowing a generic mean shift.

    The direction term is scale-normalized per pair.  The center term anchors
    the mean of both prompt arms to the frozen formal-E1 predictions, so an
    observation adapter cannot improve the loss merely by amplifying content
    already carried by subject or morphology.
    """
    shapes = {
        tuple(value.shape)
        for value in (
            anchor_prediction,
            target_prediction,
            base_anchor_prediction,
            base_target_prediction,
            anchor_target,
            target_target,
        )
    }
    if len(shapes) != 1:
        raise ValueError("paired direction tensors must share one shape")
    target_delta = target_target - anchor_target
    predicted_delta = target_prediction - anchor_prediction
    flat_target = target_delta.flatten(1)
    flat_predicted = predicted_delta.flatten(1)
    target_rms = flat_target.square().mean(dim=1).sqrt().clamp(min=1e-4)
    cosine = 1.0 - F.cosine_similarity(
        flat_predicted,
        flat_target,
        dim=1,
        eps=1e-6,
    )
    magnitude = F.smooth_l1_loss(
        predicted_delta / target_rms[:, None, None, None],
        target_delta / target_rms[:, None, None, None],
        reduction="none",
    ).mean(dim=(1, 2, 3))
    direction = (0.5 * cosine + 0.5 * magnitude).mean()
    predicted_center = 0.5 * (anchor_prediction + target_prediction)
    base_center = 0.5 * (
        base_anchor_prediction + base_target_prediction
    )
    normalized_center = (
        predicted_center - base_center
    ) / target_rms[:, None, None, None]
    center = normalized_center.square().mean()
    return direction, center, target_rms.mean()


def morphology_pair_preservation_loss(
    anchor_image: torch.Tensor,
    counterfactual_image: torch.Tensor,
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Preserve nuisance appearance while allowing the foreground shape to move.

    The factual image is a stop-gradient anchor generated with the same noise.
    Pixels outside the union of source and counterfactual masks preserve the
    background exactly; foreground channel moments preserve intensity/texture
    statistics without forcing pixel alignment across two different shapes.
    """
    if anchor_image.shape != counterfactual_image.shape:
        raise ValueError("paired morphology images must have identical shapes")
    if source_mask.ndim != 4 or target_mask.ndim != 4:
        raise ValueError("paired morphology masks must have rank four")
    if source_mask.shape[:2] != (anchor_image.shape[0], 1):
        raise ValueError("source morphology mask must have shape [batch,1,H,W]")
    if target_mask.shape[:2] != (anchor_image.shape[0], 1):
        raise ValueError("target morphology mask must have shape [batch,1,H,W]")

    def resize(mask: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            mask.to(device=anchor_image.device, dtype=anchor_image.dtype),
            size=anchor_image.shape[-2:],
            mode="nearest",
        ).clamp(0.0, 1.0)

    source = resize(source_mask)
    target = resize(target_mask)
    source_mass = source.sum(dim=(2, 3)).clamp(min=1.0)
    target_mass = target.sum(dim=(2, 3)).clamp(min=1.0)
    if bool((source.sum(dim=(1, 2, 3)) <= 0).any()) or bool(
        (target.sum(dim=(1, 2, 3)) <= 0).any()
    ):
        raise ValueError("paired morphology masks must be non-empty")

    anchor = anchor_image.detach()
    background = (1.0 - torch.maximum(source, target)).clamp(0.0, 1.0)
    background_mass = background.sum(dim=(1, 2, 3)).clamp(min=1.0)
    background_l1 = (
        ((counterfactual_image - anchor).abs() * background)
        .sum(dim=(1, 2, 3))
        / (background_mass * anchor.shape[1])
    ).mean()

    def moments(
        image: torch.Tensor,
        mask: torch.Tensor,
        mass: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = (image * mask).sum(dim=(2, 3)) / mass
        variance = (
            (image - mean[:, :, None, None]).square() * mask
        ).sum(dim=(2, 3)) / mass
        return mean, variance.clamp(min=1e-8).sqrt()

    anchor_mean, anchor_std = moments(anchor, source, source_mass)
    target_mean, target_std = moments(
        counterfactual_image,
        target,
        target_mass,
    )
    foreground_mean_l1 = F.l1_loss(target_mean, anchor_mean)
    foreground_std_l1 = F.l1_loss(target_std, anchor_std)
    total = (
        background_l1 + foreground_mean_l1 + foreground_std_l1
    ) / 3.0
    return total, {
        "background_l1": background_l1,
        "foreground_mean_l1": foreground_mean_l1,
        "foreground_std_l1": foreground_std_l1,
    }


def state_population_delta_loss(
    predicted_x0: torch.Tensor,
    target_x0: torch.Tensor,
    state_labels: torch.Tensor,
    *,
    pool_size: int = 8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match the real untreated-to-positive population shift in latent space."""
    if predicted_x0.shape != target_x0.shape or predicted_x0.ndim != 4:
        raise ValueError(
            "predicted_x0 and target_x0 must be equal-shape BCHW tensors"
        )
    if state_labels.shape != (predicted_x0.shape[0],):
        raise ValueError("state_labels must have shape [batch]")
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    negative_mask = state_labels == 0
    positive_mask = state_labels == 1
    if bool((~(negative_mask | positive_mask)).any()):
        raise ValueError("state_labels must contain only 0 and 1")
    negative_count = int(negative_mask.sum().item())
    positive_count = int(positive_mask.sum().item())
    if negative_count == 0 or negative_count != positive_count:
        raise ValueError(
            "state population batch must contain equal non-empty state groups"
        )

    predicted_features = F.adaptive_avg_pool2d(
        predicted_x0.float(),
        (pool_size, pool_size),
    )
    target_features = F.adaptive_avg_pool2d(
        target_x0.detach().float(),
        (pool_size, pool_size),
    )
    predicted_delta = (
        predicted_features[positive_mask].mean(dim=0)
        - predicted_features[negative_mask].mean(dim=0)
    )
    target_delta = (
        target_features[positive_mask].mean(dim=0)
        - target_features[negative_mask].mean(dim=0)
    )
    target_rms = target_delta.square().mean().sqrt().detach().clamp_min(1e-3)
    normalized_prediction = predicted_delta / target_rms
    normalized_target = target_delta / target_rms
    loss = F.smooth_l1_loss(
        normalized_prediction,
        normalized_target,
        beta=0.25,
    )
    cosine = F.cosine_similarity(
        predicted_delta.flatten().unsqueeze(0),
        target_delta.flatten().unsqueeze(0),
        dim=1,
        eps=1e-8,
    ).mean()
    return loss, {
        "cosine": cosine.detach(),
        "predicted_rms": predicted_delta.square().mean().sqrt().detach(),
        "target_rms": target_rms,
    }


def state_prompt_causal_neutral_inputs(
    carrier: torch.Tensor,
    upstream_prior: torch.Tensor,
    state_labels: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build equal-carrier, equal-prior and paired-noise state prompt arms."""
    if carrier.ndim != 4 or upstream_prior.ndim != 4:
        raise ValueError("carrier and upstream prior must be BCHW tensors")
    if carrier.shape[0] != upstream_prior.shape[0]:
        raise ValueError("carrier and upstream prior batch sizes must match")
    if state_labels.shape != (carrier.shape[0],):
        raise ValueError("state labels must have shape [batch]")
    negative_mask = state_labels == 0
    positive_mask = state_labels == 1
    negative_count = int(negative_mask.sum().item())
    positive_count = int(positive_mask.sum().item())
    if negative_count == 0 or negative_count != positive_count:
        raise ValueError("causal prompt arms must be balanced and non-empty")
    if bool((~(negative_mask | positive_mask)).any()):
        raise ValueError("state labels must contain only 0 and 1")

    neutral_carrier = carrier.detach().float().mean(
        dim=0,
        keepdim=True,
    ).to(dtype=carrier.dtype).expand_as(carrier)
    neutral_upstream = upstream_prior.detach().float().mean(
        dim=0,
        keepdim=True,
    ).to(dtype=upstream_prior.dtype)
    paired_noise = torch.randn_like(carrier[negative_mask])
    causal_noise = torch.empty_like(carrier)
    causal_noise[negative_mask] = paired_noise
    causal_noise[positive_mask] = paired_noise
    return (
        neutral_carrier,
        neutral_upstream,
        causal_noise,
        negative_mask,
        positive_mask,
    )


def state_population_centroid_loss(
    predicted_x0: torch.Tensor,
    target_x0: torch.Tensor,
    state_labels: torch.Tensor,
    *,
    pool_size: int = 8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match both real state centroids, not only their relative direction."""
    if predicted_x0.shape != target_x0.shape or predicted_x0.ndim != 4:
        raise ValueError(
            "predicted_x0 and target_x0 must be equal-shape BCHW tensors"
        )
    if state_labels.shape != (predicted_x0.shape[0],):
        raise ValueError("state_labels must have shape [batch]")
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    negative_mask = state_labels == 0
    positive_mask = state_labels == 1
    if bool((~(negative_mask | positive_mask)).any()):
        raise ValueError("state_labels must contain only 0 and 1")
    if (
        int(negative_mask.sum().item()) == 0
        or int(negative_mask.sum().item())
        != int(positive_mask.sum().item())
    ):
        raise ValueError("state centroid groups must be balanced and non-empty")

    predicted_features = F.adaptive_avg_pool2d(
        predicted_x0.float(),
        (pool_size, pool_size),
    )
    target_features = F.adaptive_avg_pool2d(
        target_x0.detach().float(),
        (pool_size, pool_size),
    )
    predicted_negative = predicted_features[negative_mask].mean(dim=0)
    predicted_positive = predicted_features[positive_mask].mean(dim=0)
    target_negative = target_features[negative_mask].mean(dim=0)
    target_positive = target_features[positive_mask].mean(dim=0)
    target_delta = target_positive - target_negative
    predicted_delta = predicted_positive - predicted_negative
    target_delta_rms = target_delta.square().mean().sqrt().detach().clamp_min(
        1e-3
    )
    negative_loss = F.smooth_l1_loss(
        (predicted_negative - target_negative) / target_delta_rms,
        torch.zeros_like(target_negative),
        beta=0.25,
    )
    positive_loss = F.smooth_l1_loss(
        (predicted_positive - target_positive) / target_delta_rms,
        torch.zeros_like(target_positive),
        beta=0.25,
    )
    loss = 0.5 * (negative_loss + positive_loss)
    cosine = F.cosine_similarity(
        predicted_delta.flatten().unsqueeze(0),
        target_delta.flatten().unsqueeze(0),
        dim=1,
        eps=1e-8,
    ).mean()
    predicted_midpoint = 0.5 * (
        predicted_negative + predicted_positive
    )
    target_midpoint = 0.5 * (target_negative + target_positive)
    return loss, {
        "cosine": cosine.detach(),
        "predicted_delta_rms": predicted_delta.square().mean().sqrt().detach(),
        "target_delta_rms": target_delta_rms,
        "midpoint_error_rms": (
            predicted_midpoint - target_midpoint
        ).square().mean().sqrt().detach(),
        "negative_error_rms": (
            predicted_negative - target_negative
        ).square().mean().sqrt().detach(),
        "positive_error_rms": (
            predicted_positive - target_positive
        ).square().mean().sqrt().detach(),
    }


def state_multiclass_centroid_loss(
    predicted_x0: torch.Tensor,
    target_x0: torch.Tensor,
    state_labels: torch.Tensor,
    *,
    pool_size: int = 8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match every source-local state centroid with equal class weight."""
    if predicted_x0.shape != target_x0.shape or predicted_x0.ndim != 4:
        raise ValueError(
            "predicted_x0 and target_x0 must be equal-shape BCHW tensors"
        )
    if state_labels.shape != (predicted_x0.shape[0],):
        raise ValueError("state_labels must have shape [batch]")
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    classes = torch.unique(state_labels, sorted=True)
    if classes.numel() not in {2, 3}:
        raise ValueError("multiclass state batches must contain 2 or 3 classes")
    expected = torch.arange(
        classes.numel(), device=classes.device, dtype=classes.dtype
    )
    if not torch.equal(classes, expected):
        raise ValueError("state classes must be contiguous and start at zero")
    counts = torch.stack([(state_labels == value).sum() for value in classes])
    if int(counts.min().item()) <= 0 or not bool(torch.all(counts == counts[0])):
        raise ValueError("state classes must be balanced and non-empty")

    predicted_features = F.adaptive_avg_pool2d(
        predicted_x0.float(), (pool_size, pool_size)
    )
    target_features = F.adaptive_avg_pool2d(
        target_x0.detach().float(), (pool_size, pool_size)
    )
    predicted_centroids = torch.stack(
        [predicted_features[state_labels == value].mean(dim=0) for value in classes]
    )
    target_centroids = torch.stack(
        [target_features[state_labels == value].mean(dim=0) for value in classes]
    )
    pairwise_rms = []
    for left in range(target_centroids.shape[0]):
        for right in range(left + 1, target_centroids.shape[0]):
            pairwise_rms.append(
                (target_centroids[left] - target_centroids[right])
                .square()
                .mean()
                .sqrt()
            )
    scale = torch.stack(pairwise_rms).mean().detach().clamp_min(1e-3)
    normalized_error = (predicted_centroids - target_centroids) / scale
    loss = F.smooth_l1_loss(
        normalized_error,
        torch.zeros_like(normalized_error),
        beta=0.25,
    )
    predicted_pairwise = []
    target_pairwise = []
    for left in range(target_centroids.shape[0]):
        for right in range(left + 1, target_centroids.shape[0]):
            predicted_pairwise.append(
                (predicted_centroids[left] - predicted_centroids[right])
                .square()
                .mean()
                .sqrt()
            )
            target_pairwise.append(
                (target_centroids[left] - target_centroids[right])
                .square()
                .mean()
                .sqrt()
            )
    predicted_separation = torch.stack(predicted_pairwise).mean()
    target_separation = torch.stack(target_pairwise).mean().detach()
    per_class_rms = normalized_error.square().mean(dim=(1, 2, 3)).sqrt()
    return loss, {
        "class_count": classes.new_tensor(
            float(classes.numel()), dtype=torch.float32
        ),
        "target_pairwise_rms": target_separation,
        "predicted_pairwise_rms": predicted_separation.detach(),
        "separation_ratio": (
            predicted_separation / target_separation.clamp_min(1e-3)
        ).detach(),
        "mean_normalized_class_rms": per_class_rms.mean().detach(),
        "max_normalized_class_rms": per_class_rms.max().detach(),
    }


def weighted_image_text_contrastive_loss(
    image: torch.Tensor,
    pools: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    head: nn.Module,
    weight: float,
    tau: float,
) -> torch.Tensor:
    """Keep the original H objective; skip C's unused head entirely at weight zero."""
    if weight < 0.0:
        raise ValueError("contrastive weight must be nonnegative")
    if weight == 0.0:
        return image.new_zeros(())
    z_txt = F.normalize(sum(pools) / 3.0, dim=-1)
    z_img = head(image)
    return weight * contrastive_loss(z_img, z_txt, tau)


class MSCDM(nn.Module):
    """
    多尺度条件扩散生成模型（MSCDM）：
    三尺度潜空间扩散 + 分层条件注入 + FPN 式跨尺度先验拼接。
    """

    def __init__(self, cfg: MSCDMConfig):
        super().__init__()
        if cfg.prediction_type not in ("epsilon", "v"):
            raise ValueError(
                f"prediction_type 必须为 epsilon 或 v，实际为 {cfg.prediction_type!r}"
            )
        if cfg.conditioning_mode not in (
            "hierarchical",
            "uniform_all_factors",
            "cumulative",
            "subject_state_low",
            "subject_state_low_nucleus",
            "subject_state_low_nucleus_adapter",
            "subject_state_low_nucleus_multiscale_adapter",
            "subject_state_low_nucleus_prompt_adapters",
            NUCLEUS_DISTILL_PROMPT_ADAPTER_MODE,
            NUCLEUS_DISTILL_OUTPUT_ADAPTER_MODE,
            NUCLEUS_TEMPORAL_ADAPTER_MODE,
            FACTOR_PROMPT_ADAPTER_MODE,
            FACTOR_DELTA_ADAPTER_MODE,
            FACTOR_ATTENTION_DELTA_ADAPTER_MODE,
            RGB_FACTOR_OUTPUT_ADAPTER_MODE,
            SPATIAL_SUBJECT_INPUT_ADAPTER_MODE,
            MORPHOLOGY_LATENT_BANK_MODE,
            MORPHOLOGY_FEATURE_AFFINE_MODE,
            MORPHOLOGY_SPATIAL_AFFINE_MODE,
            SEMANTIC_MULTISCALE_MODE,
            SEMANTIC_STATE_ALLSCALE_MODE,
            SEMANTIC_STATE_SEQ_LOW_MODE,
            SEMANTIC_STATE_FACTOR_RESIDUAL_POOL_MODE,
            SEMANTIC_STATE_FACTOR_PARTITION_ATTENTION_MODE,
            SEMANTIC_STATE_FACTOR_ISOLATED_COMBINED_MODE,
            SEMANTIC_STATE_SPECIFIC_RESIDUAL_MODE,
        ):
            raise ValueError(
                "conditioning_mode 必须为 hierarchical、cumulative "
                "、subject_state_low、subject_state_low_nucleus "
                "、subject_state_low_nucleus_adapter 或 "
                "subject_state_low_nucleus_multiscale_adapter 或 "
                "subject_state_low_nucleus_prompt_adapters 或 "
                "hierarchical_nucleus_distill_prompt_adapters 或 "
                "hierarchical_nucleus_distill_output_adapters 或 "
                "subject_state_low_nucleus_temporal_multiscale_adapter，"
                "hierarchical_factor_prompt_adapters，"
                "hierarchical_factor_delta_adapters，"
                "hierarchical_factor_attention_delta_adapters，"
                "hierarchical_rgb_factor_output_adapters，"
                "hierarchical_spatial_subject_input_adapter，"
                "hierarchical_morphology_latent_bank，"
                "hierarchical_morphology_feature_affine，"
                "hierarchical_morphology_spatial_affine，"
                "semantic_multiscale，"
                "semantic_state_allscale，semantic_state_seq_low，"
                "semantic_state_factor_residual_pool，"
                "semantic_state_factor_partition_attention，"
                "semantic_state_factor_isolated_combined，"
                "semantic_state_specific_residual，"
                f"实际为 {cfg.conditioning_mode!r}"
            )
        if (
            cfg.conditioning_mode == NUCLEUS_TEMPORAL_ADAPTER_MODE
            and not 0 <= int(cfg.nucleus_temporal_adapter_max_timestep) < int(
                cfg.num_train_timesteps
            )
        ):
            raise ValueError(
                "nucleus_temporal_adapter_max_timestep 必须位于 "
                f"[0, {cfg.num_train_timesteps - 1}]"
            )
        self.cfg = cfg
        z = cfg.latent_channels
        self.text_enc = MultiScaleTextEncoder(
            cfg.text_dim,
            max_len=cfg.max_text_len,
            pretrained_name=cfg.text_pretrained_name,
            local_files_only=cfg.text_local_files_only,
            freeze_backbone=cfg.freeze_clip_backbone,
            state_specific_residual_rank=(
                cfg.state_specific_residual_rank
                if cfg.conditioning_mode
                == SEMANTIC_STATE_SPECIFIC_RESIDUAL_MODE
                else None
            ),
        )
        self.factor_residual_pool = (
            FactorResidualPooledComposition(cfg.text_dim)
            if cfg.conditioning_mode
            in {
                SEMANTIC_STATE_FACTOR_RESIDUAL_POOL_MODE,
                SEMANTIC_STATE_FACTOR_ISOLATED_COMBINED_MODE,
            }
            else None
        )
        vae_kw = dict(
            base_ch=cfg.vae_base_ch,
            ch_mult=cfg.vae_ch_mult,
            num_heads=cfg.num_heads,
            gradient_checkpointing=cfg.vae_gradient_checkpointing,
        )
        self.vae_low = LatentAutoEncoder(3, z, **vae_kw)
        self.vae_mid = LatentAutoEncoder(3, z, **vae_kw)
        self.vae_high = LatentAutoEncoder(3, z, **vae_kw)
        for branch in ("low", "mid", "high"):
            mean = tuple(getattr(cfg, f"latent_mean_{branch}"))
            std = tuple(getattr(cfg, f"latent_std_{branch}"))
            if len(mean) != z or len(std) != z:
                raise ValueError(
                    f"{branch} latent calibration 必须各含 {z} 个通道，"
                    f"实际 mean={len(mean)} std={len(std)}"
                )
            if any(value <= 0 for value in std):
                raise ValueError(f"{branch} latent std 必须全部大于 0")
            # calibration 已保存在 cfg；buffer 不进入 state_dict，兼容 v1 checkpoint。
            self.register_buffer(
                f"_latent_mean_{branch}",
                torch.tensor(mean, dtype=torch.float32).view(1, z, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                f"_latent_std_{branch}",
                torch.tensor(std, dtype=torch.float32).view(1, z, 1, 1),
                persistent=False,
            )

        bl, bm, bh = cfg.base_channels
        rev_kwargs = dict(
            channel_mult=cfg.channel_mult,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=cfg.attention_resolutions,
            num_heads=cfg.num_heads,
            context_dim=cfg.unet_context_dim,
            use_reversible=cfg.use_reversible,
            reversible_save_memory=cfg.reversible_save_memory,
            factor_delta_attention_rank=(
                cfg.factor_attention_delta_rank
                if cfg.conditioning_mode
                == FACTOR_ATTENTION_DELTA_ADAPTER_MODE
                else None
            ),
        )
        self.unet_low = ScaleUNet(
            z,
            z,
            bl,
            cfg.text_dim,
            input_resolution=cfg.latent_hw[0][0],
            morphology_feature_affine=(
                cfg.conditioning_mode == MORPHOLOGY_FEATURE_AFFINE_MODE
            ),
            morphology_spatial_affine=(
                cfg.conditioning_mode == MORPHOLOGY_SPATIAL_AFFINE_MODE
            ),
            factor_partition_count=(
                3
                if cfg.conditioning_mode
                in {
                    SEMANTIC_STATE_FACTOR_PARTITION_ATTENTION_MODE,
                    SEMANTIC_STATE_FACTOR_ISOLATED_COMBINED_MODE,
                }
                else None
            ),
            **rev_kwargs,
        )
        self.unet_mid = ScaleUNet(
            z * 2,
            z,
            bm,
            cfg.text_dim,
            input_resolution=cfg.latent_hw[1][0],
            factor_partition_count=(
                2
                if cfg.conditioning_mode
                in {
                    SEMANTIC_STATE_FACTOR_PARTITION_ATTENTION_MODE,
                    SEMANTIC_STATE_FACTOR_ISOLATED_COMBINED_MODE,
                }
                else None
            ),
            **rev_kwargs,
        )
        self.unet_high = ScaleUNet(
            z * 2,
            z,
            bh,
            cfg.text_dim,
            input_resolution=cfg.latent_hw[2][0],
            factor_partition_count=(
                2
                if cfg.conditioning_mode
                in {
                    SEMANTIC_STATE_FACTOR_PARTITION_ATTENTION_MODE,
                    SEMANTIC_STATE_FACTOR_ISOLATED_COMBINED_MODE,
                }
                else None
            ),
            **rev_kwargs,
        )
        self.nucleus_low_adapter = (
            NucleusLowResidualAdapter(z, cfg.text_dim)
            if cfg.conditioning_mode
            in {
                "subject_state_low_nucleus_adapter",
                "subject_state_low_nucleus_multiscale_adapter",
                NUCLEUS_DISTILL_OUTPUT_ADAPTER_MODE,
                NUCLEUS_TEMPORAL_ADAPTER_MODE,
            }
            else None
        )
        self.nucleus_mid_adapter = (
            NucleusLowResidualAdapter(z, cfg.text_dim)
            if cfg.conditioning_mode in {
                "subject_state_low_nucleus_multiscale_adapter",
                NUCLEUS_DISTILL_OUTPUT_ADAPTER_MODE,
                NUCLEUS_TEMPORAL_ADAPTER_MODE,
            }
            else None
        )
        self.nucleus_high_adapter = (
            NucleusLowResidualAdapter(z, cfg.text_dim)
            if cfg.conditioning_mode in {
                "subject_state_low_nucleus_multiscale_adapter",
                NUCLEUS_DISTILL_OUTPUT_ADAPTER_MODE,
                NUCLEUS_TEMPORAL_ADAPTER_MODE,
            }
            else None
        )
        self.nucleus_prompt_adapters = (
            nn.ModuleDict(
                {
                    "subject_state_low": PromptGatedTokenAdapter(cfg.text_dim),
                    "state_mid": PromptGatedTokenAdapter(cfg.text_dim),
                    "background_high": PromptGatedTokenAdapter(cfg.text_dim),
                }
            )
            if cfg.conditioning_mode
            == "subject_state_low_nucleus_prompt_adapters"
            else None
        )
        self.nucleus_distill_prompt_adapters = (
            nn.ModuleDict(
                {
                    "subject_low": ContextGatedTokenAdapter(cfg.text_dim),
                    "state_mid": ContextGatedTokenAdapter(cfg.text_dim),
                    "background_high": ContextGatedTokenAdapter(cfg.text_dim),
                }
            )
            if cfg.conditioning_mode
            == NUCLEUS_DISTILL_PROMPT_ADAPTER_MODE
            else None
        )
        self.factor_prompt_adapters = (
            nn.ModuleDict(
                {
                    "subject_low": PromptGatedTokenAdapter(cfg.text_dim),
                    "state_mid": PromptGatedTokenAdapter(cfg.text_dim),
                    "background_high": PromptGatedTokenAdapter(cfg.text_dim),
                }
            )
            if cfg.conditioning_mode == FACTOR_PROMPT_ADAPTER_MODE
            else None
        )
        self.factor_delta_adapters = (
            nn.ModuleDict(
                {
                    "subject_low": FactorDeltaTokenAdapter(cfg.text_dim),
                    "state_mid": FactorDeltaTokenAdapter(cfg.text_dim),
                    "background_high": FactorDeltaTokenAdapter(cfg.text_dim),
                }
            )
            if cfg.conditioning_mode == FACTOR_DELTA_ADAPTER_MODE
            else None
        )
        self.rgb_factor_output_adapters = (
            nn.ModuleDict(
                {
                    f"{branch}_{factor}": NucleusLowResidualAdapter(
                        z,
                        cfg.text_dim,
                    )
                    for branch in ("low", "mid", "high")
                    for factor in ("morphology", "observation")
                }
            )
            if cfg.conditioning_mode == RGB_FACTOR_OUTPUT_ADAPTER_MODE
            else None
        )
        self.spatial_subject_input_adapter = (
            SpatialSubjectInputAdapter(z)
            if cfg.conditioning_mode == SPATIAL_SUBJECT_INPUT_ADAPTER_MODE
            else None
        )
        self.morphology_latent_bank = (
            MorphologyLatentBank(z, cfg.latent_hw[0][0])
            if cfg.conditioning_mode == MORPHOLOGY_LATENT_BANK_MODE
            else None
        )

        self.diffusion = GaussianDiffusion(
            cfg.num_train_timesteps, cfg.beta_schedule, cfg.beta_start, cfg.beta_end
        )

        self.null_subj_seq = nn.Parameter(torch.zeros(1, 1, cfg.text_dim))
        self.null_state_seq = nn.Parameter(torch.zeros(1, 1, cfg.text_dim))
        self.null_bg_seq = nn.Parameter(torch.zeros(1, 1, cfg.text_dim))
        self.null_subj = nn.Parameter(torch.zeros(1, cfg.text_dim))
        self.null_state = nn.Parameter(torch.zeros(1, cfg.text_dim))
        self.null_bg = nn.Parameter(torch.zeros(1, cfg.text_dim))

        self.contrastive = ContrastiveHead(3, cfg.text_dim)
        self.personalization = nn.Parameter(torch.randn(1, cfg.personalization_token_dim))
        self.proj_pers = nn.Linear(cfg.personalization_token_dim, cfg.text_dim)

        if cfg.freeze_vae:
            self.freeze_vae()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.cfg.freeze_vae:
            for vae in (self.vae_low, self.vae_mid, self.vae_high):
                vae.eval()
        return self

    def _route_conditions(
        self,
        conditions: tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ],
        subject_state_low_mask: torch.Tensor | None = None,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """Route raw factor conditions with any model-owned composition module."""
        return route_condition_embeddings(
            conditions,
            self.cfg.conditioning_mode,
            subject_state_low_mask=subject_state_low_mask,
            factor_residual_pool=self.factor_residual_pool,
            factor_null_pools=(
                self.null_subj,
                self.null_state,
                self.null_bg,
            ),
        )

    def freeze_vae(self) -> None:
        """冻结已验证的三路 VAE；训练阶段仅学习标准化潜空间中的扩散网络。"""
        self.cfg.freeze_vae = True
        for vae in (self.vae_low, self.vae_mid, self.vae_high):
            vae.requires_grad_(False)
            vae.eval()

    def _normalize_latent(self, z: torch.Tensor, branch: str) -> torch.Tensor:
        if not self.cfg.latent_normalization:
            return z
        mean = getattr(self, f"_latent_mean_{branch}").to(dtype=z.dtype)
        std = getattr(self, f"_latent_std_{branch}").to(dtype=z.dtype)
        return (z - mean) / std

    def _denormalize_latent(self, z: torch.Tensor, branch: str) -> torch.Tensor:
        if not self.cfg.latent_normalization:
            return z
        mean = getattr(self, f"_latent_mean_{branch}").to(dtype=z.dtype)
        std = getattr(self, f"_latent_std_{branch}").to(dtype=z.dtype)
        return z * std + mean

    def _encode_branch(
        self,
        vae: LatentAutoEncoder,
        x: torch.Tensor,
        branch: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        if self.cfg.freeze_vae:
            with torch.no_grad():
                mean, logvar = vae.encode(x)
                z_raw = (
                    mean
                    if self.cfg.frozen_vae_use_mean
                    else vae.reparameterize(mean, logvar)
                )
            return self._normalize_latent(z_raw, branch), None, mean, logvar

        z_raw, reconstruction, mean, logvar = vae(x)
        return self._normalize_latent(z_raw, branch), reconstruction, mean, logvar

    def _maybe_dropout_cond(
        self,
        conds,
        p: float,
        *,
        independent: bool = False,
        drop_masks: torch.Tensor | None = None,
    ):
        (s_seq, s_pool), (m_seq, m_pool), (b_seq, b_pool) = conds
        if not self.training or p <= 0:
            return conds
        bsz = s_pool.shape[0]
        device = s_pool.device
        if drop_masks is not None:
            if drop_masks.shape != (bsz, 3):
                raise ValueError(
                    "condition drop mask must have shape [batch, 3], "
                    f"got {tuple(drop_masks.shape)}"
                )
            if not bool(torch.all((drop_masks == 0) | (drop_masks == 1)).item()):
                raise ValueError("condition drop mask must contain only 0/1 values")
            masks = [
                drop_masks[:, index].to(device=device, dtype=torch.float32)[:, None]
                for index in range(3)
            ]
        elif independent:
            masks = [
                (torch.rand(bsz, device=device) < p).float()[:, None]
                for _ in range(3)
            ]
        else:
            shared = (torch.rand(bsz, device=device) < p).float()[:, None]
            masks = [shared, shared, shared]
        ns_seq = self.null_subj_seq.expand(bsz, -1, -1)
        nst_seq = self.null_state_seq.expand(bsz, -1, -1)
        nbg_seq = self.null_bg_seq.expand(bsz, -1, -1)
        ns_pool = self.null_subj.expand(bsz, -1)
        nst_pool = self.null_state.expand(bsz, -1)
        nbg_pool = self.null_bg.expand(bsz, -1)
        outputs = []
        for (sequence, pooled), null_sequence, null_pooled, mask in zip(
            ((s_seq, s_pool), (m_seq, m_pool), (b_seq, b_pool)),
            (ns_seq, nst_seq, nbg_seq),
            (ns_pool, nst_pool, nbg_pool),
            masks,
        ):
            sequence_mask = mask.unsqueeze(-1)
            outputs.append(
                (
                    sequence * (1 - sequence_mask)
                    + null_sequence * sequence_mask,
                    pooled * (1 - mask) + null_pooled * mask,
                )
            )
        return tuple(outputs)

    def _add_personalization(self, c_sub_seq: torch.Tensor, c_sub_pool: torch.Tensor, use_token: bool):
        if not use_token:
            return c_sub_seq, c_sub_pool
        t = self.proj_pers(self.personalization).expand(c_sub_pool.shape[0], -1)
        t_seq = t[:, None, :]
        return torch.cat([t_seq, c_sub_seq], dim=1), c_sub_pool + t

    def _apply_nucleus_prompt_adapters(
        self,
        conditions: tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ],
        subject_state_low_mask: torch.Tensor | None,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        adapters = self.nucleus_prompt_adapters
        distill_adapters = self.nucleus_distill_prompt_adapters
        factor_adapters = self.factor_prompt_adapters
        if (
            adapters is None
            and distill_adapters is None
            and factor_adapters is None
        ):
            return conditions
        if factor_adapters is not None:
            route_mask = torch.ones(
                conditions[0][0].shape[0],
                device=conditions[0][0].device,
                dtype=torch.bool,
            )
            return (
                factor_adapters["subject_low"](
                    *conditions[0],
                    route_mask,
                ),
                factor_adapters["state_mid"](
                    *conditions[1],
                    route_mask,
                ),
                factor_adapters["background_high"](
                    *conditions[2],
                    route_mask,
                ),
            )
        if subject_state_low_mask is None:
            raise ValueError("nucleus prompt adapters require subject_state_low_mask")
        if distill_adapters is not None:
            subject_pool = conditions[0][1]
            state_pool = conditions[1][1]
            background_pool = conditions[2][1]
            return (
                distill_adapters["subject_low"](
                    *conditions[0],
                    (subject_pool + state_pool) / 2.0,
                    subject_state_low_mask,
                ),
                distill_adapters["state_mid"](
                    *conditions[1],
                    state_pool,
                    subject_state_low_mask,
                ),
                distill_adapters["background_high"](
                    *conditions[2],
                    background_pool,
                    subject_state_low_mask,
                ),
            )
        return (
            adapters["subject_state_low"](
                *conditions[0],
                subject_state_low_mask,
            ),
            adapters["state_mid"](
                *conditions[1],
                subject_state_low_mask,
            ),
            adapters["background_high"](
                *conditions[2],
                subject_state_low_mask,
            ),
        )

    def _apply_factor_delta_adapters(
        self,
        target_conditions: tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ],
        anchor_conditions: tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ],
        intervention_factor: int | torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        adapters = self.factor_delta_adapters
        if adapters is None:
            return target_conditions
        batch_size = target_conditions[0][0].shape[0]
        if isinstance(intervention_factor, int):
            if intervention_factor not in {0, 1, 2}:
                raise ValueError("intervention factor must be 0, 1, or 2")
            factor_ids = torch.full(
                (batch_size,),
                intervention_factor,
                device=target_conditions[0][0].device,
                dtype=torch.long,
            )
        else:
            factor_ids = intervention_factor.to(
                device=target_conditions[0][0].device,
                dtype=torch.long,
            )
            if factor_ids.shape != (batch_size,):
                raise ValueError(
                    "intervention_factor tensor must have shape [batch]"
                )
            if bool(((factor_ids < 0) | (factor_ids > 2)).any()):
                raise ValueError("intervention factor values must be 0, 1, or 2")
        keys = ("subject_low", "state_mid", "background_high")
        return tuple(
            adapters[key](
                *target_conditions[index],
                *anchor_conditions[index],
                factor_ids == index,
            )
            for index, key in enumerate(keys)
        )

    def _apply_rgb_factor_output_adapters(
        self,
        prediction: torch.Tensor,
        timesteps: torch.Tensor,
        morphology_pool: torch.Tensor,
        observation_pool: torch.Tensor,
        branch: str,
        observation_scale: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """Add RGB-only morphology and observation residuals at one scale.

        The two factors remain separate and the frozen hierarchical prediction
        is preserved exactly at initialization because both output projections
        are zero initialized.  These adapters act on RGB diffusion latents;
        they never expose or reconstruct raw microscopy channels.
        """
        adapters = self.rgb_factor_output_adapters
        if adapters is None:
            return prediction
        if branch not in {"low", "mid", "high"}:
            raise ValueError(f"unsupported RGB factor adapter branch: {branch!r}")
        morphology_delta = adapters[f"{branch}_morphology"](
            prediction,
            timesteps,
            morphology_pool,
        )
        observation_delta = adapters[f"{branch}_observation"](
            prediction,
            timesteps,
            observation_pool,
        )
        scale = torch.as_tensor(
            observation_scale,
            device=observation_delta.device,
            dtype=observation_delta.dtype,
        )
        if scale.ndim == 0:
            pass
        elif scale.ndim == 1 and scale.shape[0] == observation_delta.shape[0]:
            scale = scale[:, None, None, None]
        else:
            raise ValueError(
                "RGB observation adapter scale must be scalar or have shape [batch]"
            )
        if not bool(torch.isfinite(scale).all()):
            raise ValueError("RGB observation adapter scale must be finite")
        if bool(((scale < 0.0) | (scale > 1.0)).any()):
            raise ValueError("RGB observation adapter scale must lie in [0, 1]")
        return prediction + morphology_delta + scale * observation_delta

    @staticmethod
    def _factor_route_masks(
        intervention_factor: int | torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(intervention_factor, int):
            if intervention_factor not in {0, 1, 2}:
                raise ValueError("intervention factor must be 0, 1, or 2")
            factor_ids = torch.full(
                (batch_size,),
                intervention_factor,
                device=device,
                dtype=torch.long,
            )
        else:
            factor_ids = intervention_factor.to(
                device=device,
                dtype=torch.long,
            )
            if factor_ids.shape != (batch_size,):
                raise ValueError(
                    "intervention_factor tensor must have shape [batch]"
                )
            if bool(((factor_ids < 0) | (factor_ids > 2)).any()):
                raise ValueError(
                    "intervention factor values must be 0, 1, or 2"
                )
        return tuple(factor_ids == index for index in range(3))

    def encode_triplet(self, x128: torch.Tensor, x256: torch.Tensor, x512: torch.Tensor):
        z_l, xr_l, mu_l, lv_l = self._encode_branch(self.vae_low, x128, "low")
        z_m, xr_m, mu_m, lv_m = self._encode_branch(self.vae_mid, x256, "mid")
        z_h, xr_h, mu_h, lv_h = self._encode_branch(self.vae_high, x512, "high")
        return (z_l, z_m, z_h), (xr_l, xr_m, xr_h), (mu_l, mu_m, mu_h), (lv_l, lv_m, lv_h)

    def counterfactual_pair_losses(
        self,
        anchor_x512: torch.Tensor,
        target_x512: torch.Tensor,
        anchor_subject_ids: torch.Tensor,
        anchor_state_ids: torch.Tensor,
        anchor_bg_ids: torch.Tensor,
        target_subject_ids: torch.Tensor,
        target_state_ids: torch.Tensor,
        target_bg_ids: torch.Tensor,
        *,
        factor: int,
        teacher_unets: tuple[nn.Module, nn.Module],
        timestep_min: int,
        timestep_max: int,
    ) -> dict[str, torch.Tensor]:
        """Supervise a single-factor intervention from matched real exemplars.

        Both conditions see one shared noisy midpoint.  The change in model
        output must match the real latent endpoint difference, while the source
        endpoint stays anchored to the frozen original-v4 branch.
        """
        if self.cfg.conditioning_mode != "hierarchical":
            raise ValueError(
                "counterfactual pair supervision currently requires the "
                "original hierarchical conditioning path"
            )
        if factor not in {0, 1}:
            raise ValueError("counterfactual factor must be 0 (subject) or 1 (state)")
        if target_x512.ndim != 5:
            raise ValueError(
                "counterfactual targets must have shape "
                "[batch, exemplars, channels, height, width]"
            )
        if anchor_x512.ndim == 4:
            anchor_x512 = anchor_x512[:, None]
        if (
            anchor_x512.ndim != 5
            or anchor_x512.shape[0] != target_x512.shape[0]
            or anchor_x512.shape[2:] != target_x512.shape[2:]
        ):
            raise ValueError("counterfactual image pair dimensions do not match")
        if len(teacher_unets) != 2:
            raise ValueError("teacher_unets must contain frozen low and mid U-Nets")
        if not 0 <= timestep_min <= timestep_max < self.cfg.num_train_timesteps:
            raise ValueError("invalid counterfactual timestep interval")

        pair_batch, anchor_exemplar_count = anchor_x512.shape[:2]
        flat_anchor_x512 = anchor_x512.flatten(0, 1)
        anchor_x128 = F.interpolate(
            flat_anchor_x512,
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        )
        target_exemplar_count = target_x512.shape[1]
        flat_target_x512 = target_x512.flatten(0, 1)
        target_x128 = F.interpolate(
            flat_target_x512,
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        )
        anchor_z_l_all = self._encode_branch(
            self.vae_low,
            anchor_x128,
            "low",
        )[0]
        anchor_z_l = anchor_z_l_all.unflatten(
            0,
            (pair_batch, anchor_exemplar_count),
        ).mean(dim=1)
        target_z_l_all = self._encode_branch(
            self.vae_low,
            target_x128,
            "low",
        )[0]
        target_z_l = target_z_l_all.unflatten(
            0,
            (pair_batch, target_exemplar_count),
        ).mean(dim=1)

        anchor_conditions = self.text_enc(
            anchor_subject_ids,
            anchor_state_ids,
            anchor_bg_ids,
        )
        target_conditions = self.text_enc(
            target_subject_ids,
            target_state_ids,
            target_bg_ids,
        )

        if factor == 0:
            anchor_z = anchor_z_l
            target_z = target_z_l
            prior = None
            unet = self.unet_low
            teacher_unet = teacher_unets[0]
            anchor_condition = anchor_conditions[0]
            target_condition = target_conditions[0]
        else:
            anchor_x256 = F.interpolate(
                flat_anchor_x512,
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            )
            target_x256 = F.interpolate(
                flat_target_x512,
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            )
            anchor_z_all = self._encode_branch(
                self.vae_mid,
                anchor_x256,
                "mid",
            )[0]
            anchor_z = anchor_z_all.unflatten(
                0,
                (pair_batch, anchor_exemplar_count),
            ).mean(dim=1)
            target_z_all = self._encode_branch(
                self.vae_mid,
                target_x256,
                "mid",
            )[0]
            target_z = target_z_all.unflatten(
                0,
                (pair_batch, target_exemplar_count),
            ).mean(dim=1)
            shared_low = 0.5 * (anchor_z_l + target_z_l)
            prior = F.interpolate(
                shared_low,
                size=anchor_z.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            unet = self.unet_mid
            teacher_unet = teacher_unets[1]
            anchor_condition = anchor_conditions[1]
            target_condition = target_conditions[1]

        batch_size = anchor_z.shape[0]
        timesteps = torch.randint(
            timestep_min,
            timestep_max + 1,
            (batch_size,),
            device=anchor_z.device,
        )
        midpoint = 0.5 * (anchor_z + target_z)
        noise = torch.randn_like(midpoint)
        x_t = self.diffusion.q_sample(midpoint, timesteps, noise)
        model_input = (
            torch.cat([x_t, prior], dim=1)
            if prior is not None
            else x_t
        )
        anchor_prediction = unet(
            model_input,
            timesteps,
            *anchor_condition,
        )
        target_prediction = unet(
            model_input,
            timesteps,
            *target_condition,
        )
        with torch.no_grad():
            teacher_anchor_prediction = teacher_unet(
                model_input,
                timesteps,
                *anchor_condition,
            )
            teacher_target_prediction = teacher_unet(
                model_input,
                timesteps,
                *target_condition,
            )

        alpha = self.diffusion.sqrt_ac[timesteps][:, None, None, None]
        sigma = self.diffusion.sqrt_om_ac[timesteps][:, None, None, None]
        anchor_noise = (x_t - alpha * anchor_z) / sigma.clamp(min=1e-6)
        target_noise = (x_t - alpha * target_z) / sigma.clamp(min=1e-6)
        anchor_output_target = self.diffusion.training_target(
            anchor_z,
            timesteps,
            anchor_noise,
            self.cfg.prediction_type,
        )
        target_output_target = self.diffusion.training_target(
            target_z,
            timesteps,
            target_noise,
            self.cfg.prediction_type,
        )
        direction, anchor, target_rms = (
            counterfactual_direction_and_anchor_loss(
                anchor_prediction,
                target_prediction,
                teacher_anchor_prediction,
                anchor_output_target,
                target_output_target,
            )
        )
        target_orthogonal = counterfactual_target_orthogonal_loss(
            target_prediction,
            teacher_target_prediction,
            anchor_output_target,
            target_output_target,
        )
        return {
            "direction": direction,
            "teacher_anchor": anchor,
            "teacher_target_orthogonal": target_orthogonal,
            "target_direction_rms": target_rms.detach(),
        }

    def observation_rgb_pair_losses(
        self,
        anchor_x512: torch.Tensor,
        target_x512: torch.Tensor,
        anchor_subject_ids: torch.Tensor,
        anchor_state_ids: torch.Tensor,
        anchor_bg_ids: torch.Tensor,
        target_subject_ids: torch.Tensor,
        target_state_ids: torch.Tensor,
        target_bg_ids: torch.Tensor,
        *,
        timestep_min: int,
        timestep_max: int,
    ) -> dict[str, torch.Tensor]:
        """Teach an observation-prompt switch from paired pseudocolor RGB.

        Both prompt arms see one shared noisy midpoint at each VAE scale.  The
        only trainable path in P17 is the observation output adapter.  The
        target is the real rendered-RGB latent difference, not any raw channel
        tensor or cross-channel reconstruction objective.
        """
        if self.cfg.conditioning_mode != RGB_FACTOR_OUTPUT_ADAPTER_MODE:
            raise ValueError(
                "observation RGB pair loss requires RGB factor output adapters"
            )
        if self.rgb_factor_output_adapters is None:
            raise ValueError("RGB factor output adapters are not initialized")
        if anchor_x512.ndim == 5:
            if anchor_x512.shape[1] != 1:
                raise ValueError("P17 requires exactly one anchor RGB image")
            anchor_x512 = anchor_x512[:, 0]
        if target_x512.ndim == 5:
            if target_x512.shape[1] != 1:
                raise ValueError("P17 requires exactly one target RGB image")
            target_x512 = target_x512[:, 0]
        if (
            anchor_x512.ndim != 4
            or target_x512.ndim != 4
            or anchor_x512.shape != target_x512.shape
        ):
            raise ValueError(
                "P17 observation RGB pairs must be equal-shape BCHW tensors"
            )
        if not (
            0
            <= timestep_min
            <= timestep_max
            < self.cfg.num_train_timesteps
        ):
            raise ValueError("invalid observation RGB pair timestep interval")

        anchor_x256 = F.interpolate(
            anchor_x512,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )
        target_x256 = F.interpolate(
            target_x512,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )
        anchor_x128 = F.interpolate(
            anchor_x512,
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        )
        target_x128 = F.interpolate(
            target_x512,
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        )
        with torch.no_grad():
            anchor_latents = (
                self._encode_branch(self.vae_low, anchor_x128, "low")[0],
                self._encode_branch(self.vae_mid, anchor_x256, "mid")[0],
                self._encode_branch(self.vae_high, anchor_x512, "high")[0],
            )
            target_latents = (
                self._encode_branch(self.vae_low, target_x128, "low")[0],
                self._encode_branch(self.vae_mid, target_x256, "mid")[0],
                self._encode_branch(self.vae_high, target_x512, "high")[0],
            )
            anchor_conditions = self._route_conditions(
                self.text_enc(
                    anchor_subject_ids,
                    anchor_state_ids,
                    anchor_bg_ids,
                )
            )
            target_conditions = self._route_conditions(
                self.text_enc(
                    target_subject_ids,
                    target_state_ids,
                    target_bg_ids,
                )
            )

        midpoint_low = 0.5 * (
            anchor_latents[0] + target_latents[0]
        )
        midpoint_mid = 0.5 * (
            anchor_latents[1] + target_latents[1]
        )
        priors = (
            None,
            F.interpolate(
                midpoint_low,
                size=anchor_latents[1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            ),
            F.interpolate(
                midpoint_mid,
                size=anchor_latents[2].shape[-2:],
                mode="bilinear",
                align_corners=False,
            ),
        )
        branches = (
            ("low", self.unet_low, 0),
            ("mid", self.unet_mid, 1),
            ("high", self.unet_high, 2),
        )
        weights = self.cfg.scale_weights()
        weight_sum = float(sum(weights))
        direction_total = anchor_x512.new_zeros(())
        center_total = anchor_x512.new_zeros(())
        output: dict[str, torch.Tensor] = {}
        batch_size = anchor_x512.shape[0]
        for branch_index, (branch, unet, condition_index) in enumerate(
            branches
        ):
            anchor_z = anchor_latents[branch_index]
            target_z = target_latents[branch_index]
            midpoint = 0.5 * (anchor_z + target_z)
            timesteps = torch.randint(
                timestep_min,
                timestep_max + 1,
                (batch_size,),
                device=midpoint.device,
            )
            noise = torch.randn_like(midpoint)
            x_t = self.diffusion.q_sample(midpoint, timesteps, noise)
            prior = priors[branch_index]
            model_input = (
                torch.cat([x_t, prior], dim=1)
                if prior is not None
                else x_t
            )
            with torch.no_grad():
                base_anchor = unet(
                    model_input,
                    timesteps,
                    *anchor_conditions[condition_index],
                )
                base_target = unet(
                    model_input,
                    timesteps,
                    *target_conditions[condition_index],
                )
            adapter = self.rgb_factor_output_adapters[
                f"{branch}_observation"
            ]
            anchor_prediction = base_anchor + adapter(
                base_anchor,
                timesteps,
                anchor_conditions[2][1],
            )
            target_prediction = base_target + adapter(
                base_target,
                timesteps,
                target_conditions[2][1],
            )
            alpha = self.diffusion.sqrt_ac[timesteps][
                :, None, None, None
            ]
            sigma = self.diffusion.sqrt_om_ac[timesteps][
                :, None, None, None
            ]
            anchor_noise = (
                x_t - alpha * anchor_z
            ) / sigma.clamp(min=1e-6)
            target_noise = (
                x_t - alpha * target_z
            ) / sigma.clamp(min=1e-6)
            anchor_output_target = self.diffusion.training_target(
                anchor_z,
                timesteps,
                anchor_noise,
                self.cfg.prediction_type,
            )
            target_output_target = self.diffusion.training_target(
                target_z,
                timesteps,
                target_noise,
                self.cfg.prediction_type,
            )
            direction, center, target_rms = (
                paired_direction_and_center_loss(
                    anchor_prediction,
                    target_prediction,
                    base_anchor,
                    base_target,
                    anchor_output_target,
                    target_output_target,
                )
            )
            normalized_weight = float(weights[branch_index]) / weight_sum
            direction_total = (
                direction_total + normalized_weight * direction
            )
            center_total = center_total + normalized_weight * center
            output[f"direction_{branch}"] = direction.detach()
            output[f"center_{branch}"] = center.detach()
            output[f"target_rms_{branch}"] = target_rms.detach()
        output["direction"] = direction_total
        output["center"] = center_total
        return output

    def state_early_ddim_low_trajectory(
        self,
        initial_noise: torch.Tensor,
        raw_conditions: tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ],
        *,
        inference_steps: int = 25,
        trace_steps: int = 2,
        subject_cfg: float = 2.0,
        subject_state_low_mask: torch.Tensor | None = None,
        detach_conditions: bool = False,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Run the differentiable low-branch prefix used by formal sampling.

        The helper is shared by the training objective and the server parity
        audit.  Keeping condition routing, factor CFG and DDIM updates here
        prevents a silent second implementation from drifting away from the
        inference path.
        """
        if self.cfg.conditioning_mode != SEMANTIC_STATE_ALLSCALE_MODE:
            raise ValueError(
                "early DDIM low trajectory requires semantic_state_allscale"
            )
        if initial_noise.ndim != 4 or initial_noise.dtype != torch.float32:
            raise ValueError(
                "early DDIM initial noise must be a BCHW float32 tensor"
            )
        if inference_steps != 25 or trace_steps != 2 or subject_cfg != 2.0:
            raise ValueError(
                "early DDIM low trajectory is frozen to 25 steps, a two-step "
                "prefix and subject CFG 2"
            )
        batch_size = initial_noise.shape[0]
        if any(sequence.shape[0] != batch_size for sequence, _ in raw_conditions):
            raise ValueError("condition batch size does not match initial noise")

        # Keep the sampled state in FP32 exactly like diffusion.ddim_sample.
        # Autocast may lower individual U-Net kernels, but the DDIM carrier is
        # not pre-cast before the model call.
        current = initial_noise
        full_low = self._route_conditions(
            raw_conditions,
            subject_state_low_mask=subject_state_low_mask,
        )[0]
        raw_without_subject = (
            (
                self.null_subj_seq.expand(
                    batch_size, raw_conditions[0][0].shape[1], -1
                ),
                self.null_subj.expand(batch_size, -1),
            ),
            raw_conditions[1],
            raw_conditions[2],
        )
        without_subject_low = self._route_conditions(
            raw_without_subject,
            subject_state_low_mask=subject_state_low_mask,
        )[0]
        if detach_conditions:
            full_low = tuple(value.detach() for value in full_low)
            without_subject_low = tuple(
                value.detach() for value in without_subject_low
            )

        timesteps = self.diffusion.ddim_timesteps(inference_steps)
        predicted_x0_values: list[torch.Tensor] = []
        guided_predictions: list[torch.Tensor] = []
        for step_index in range(trace_steps):
            timestep = timesteps[step_index].expand(batch_size)
            full_prediction = self.unet_low(
                current,
                timestep,
                *full_low,
            )
            without_subject_prediction = self.unet_low(
                current,
                timestep,
                *without_subject_low,
            )
            guided_prediction = combine_factor_cfg_predictions(
                full_prediction,
                ((without_subject_prediction, subject_cfg),),
            )
            predicted_x0, _predicted_eps, current = (
                self.diffusion.ddim_eta0_step(
                    current,
                    timestep,
                    guided_prediction,
                    prediction_type=self.cfg.prediction_type,
                    t_next=timesteps[step_index + 1].expand(batch_size),
                )
            )
            predicted_x0_values.append(predicted_x0)
            guided_predictions.append(guided_prediction)
        return tuple(predicted_x0_values), tuple(guided_predictions)

    def training_losses(
        self,
        x512: torch.Tensor,
        subject_ids: torch.Tensor,
        state_ids: torch.Tensor,
        bg_ids: torch.Tensor,
        cond_drop_prob: float = 0.1,
        cond_drop_mode: str = "joint",
        cond_drop_masks: torch.Tensor | None = None,
        vae_weight: float = 0.1,
        use_personalization: bool = False,
        negative_subject_ids: torch.Tensor | None = None,
        negative_state_ids: torch.Tensor | None = None,
        negative_bg_ids: torch.Tensor | None = None,
        control_rank_enabled: torch.Tensor | None = None,
        control_rank_factor: int = 0,
        control_rank_weight: float = 0.0,
        control_rank_margin: float = 0.01,
        control_rank_scope: str = "single_branch",
        factor_surrogate_bundle: nn.Module | None = None,
        factor_surrogate_target_ids: torch.Tensor | None = None,
        factor_surrogate_weight: float = 0.0,
        factor_surrogate_timestep_min: int = 0,
        factor_surrogate_timestep_max: int = 249,
        factor_surrogate_state_population_size: int = 1,
        state_population_labels: torch.Tensor | None = None,
        state_population_delta_weight: float = 0.0,
        state_population_size: int = 1,
        state_population_timestep_min: int = 750,
        state_population_timestep_max: int = 999,
        state_population_pool_size: int = 8,
        state_prompt_causal_mid_weight: float = 0.0,
        state_prompt_causal_timestep_min: int = 750,
        state_prompt_causal_timestep_max: int = 999,
        state_prompt_causal_pool_size: int = 8,
        state_prompt_causal_high_centroid_weight: float = 0.0,
        state_prompt_causal_high_timestep_min: int = 750,
        state_prompt_causal_high_timestep_max: int = 999,
        state_prompt_causal_high_pool_size: int = 8,
        state_trajectory_centroid_weight: float = 0.0,
        state_trajectory_population_size: int = 0,
        state_trajectory_pool_size: int = 8,
        state_trajectory_timestep: int = 999,
        state_trajectory_generator: torch.Generator | None = None,
        state_early_ddim_centroid_weight: float = 0.0,
        state_early_ddim_population_size: int = 0,
        state_early_ddim_pool_size: int = 8,
        state_early_ddim_inference_steps: int = 25,
        state_early_ddim_trace_steps: int = 2,
        state_early_ddim_subject_cfg: float = 2.0,
        state_early_ddim_generator: torch.Generator | None = None,
        state_early_ddim_probe_gradients: bool = False,
        morphology_mask_surrogate_bundle: nn.Module | None = None,
        morphology_mask_target: torch.Tensor | None = None,
        morphology_mask_source_ids: torch.Tensor | None = None,
        morphology_mask_surrogate_weight: float = 0.0,
        morphology_mask_surrogate_timestep_min: int = 0,
        morphology_mask_surrogate_timestep_max: int = 249,
        morphology_pair_preservation_weight: float = 0.0,
        foreground_mask: torch.Tensor | None = None,
        foreground_loss_mix: float = 0.0,
        prior_proxy_mix: float = 0.0,
        cascade_prior_source_mask: torch.Tensor | None = None,
        cascade_prior_gradient_mode: str = "off",
        cascade_prior_capture_forward: bool = False,
        subject_state_low_mask: torch.Tensor | None = None,
        spatial_subject_mask: torch.Tensor | None = None,
        spatial_subject_route_mask: torch.Tensor | None = None,
        morphology_class_ids: torch.Tensor | None = None,
        morphology_counterfactual_class_ids: torch.Tensor | None = None,
        counterfactual_anchor_x512: torch.Tensor | None = None,
        counterfactual_target_x512: torch.Tensor | None = None,
        counterfactual_anchor_subject_ids: torch.Tensor | None = None,
        counterfactual_anchor_state_ids: torch.Tensor | None = None,
        counterfactual_anchor_bg_ids: torch.Tensor | None = None,
        counterfactual_target_subject_ids: torch.Tensor | None = None,
        counterfactual_target_state_ids: torch.Tensor | None = None,
        counterfactual_target_bg_ids: torch.Tensor | None = None,
        counterfactual_factor: int = 0,
        counterfactual_pair_weight: float = 0.0,
        counterfactual_teacher_anchor_mix: float = 1.0,
        counterfactual_target_orthogonal_mix: float = 0.0,
        counterfactual_timestep_min: int = 500,
        counterfactual_timestep_max: int = 999,
        counterfactual_teacher_unets: tuple[nn.Module, nn.Module] | None = None,
        observation_rgb_anchor_x512: torch.Tensor | None = None,
        observation_rgb_target_x512: torch.Tensor | None = None,
        observation_rgb_anchor_subject_ids: torch.Tensor | None = None,
        observation_rgb_anchor_state_ids: torch.Tensor | None = None,
        observation_rgb_anchor_bg_ids: torch.Tensor | None = None,
        observation_rgb_target_subject_ids: torch.Tensor | None = None,
        observation_rgb_target_state_ids: torch.Tensor | None = None,
        observation_rgb_target_bg_ids: torch.Tensor | None = None,
        observation_rgb_pair_weight: float = 0.0,
        observation_rgb_pair_center_weight: float = 0.25,
        observation_rgb_pair_timestep_min: int = 500,
        observation_rgb_pair_timestep_max: int = 999,
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        self._last_state_trajectory_activation_gradient_rms: dict[str, float] = {}
        self._last_state_early_ddim_activation_gradient_rms: dict[
            str, float
        ] = {}
        self._last_state_early_ddim_parameter_gradient_stats: dict[
            str, torch.Tensor
        ] = {}
        self._last_cascade_prior_activation_gradient_rms: dict[str, float] = {
            "low_from_mid_high": 0.0,
            "mid_from_high": 0.0,
        }
        self._last_cascade_prior_forward_fingerprints: dict[str, Any] = {}
        if cascade_prior_gradient_mode not in CASCADE_PRIOR_GRADIENT_MODES:
            raise ValueError(
                "cascade_prior_gradient_mode must be off, stopgrad, or e2e"
            )
        cascade_prior_active = cascade_prior_gradient_mode != "off"
        if cascade_prior_active:
            if cfg.conditioning_mode != SEMANTIC_STATE_ALLSCALE_MODE:
                raise ValueError(
                    "single-step cascade prior requires semantic_state_allscale"
                )
            if prior_proxy_mix != 0.0:
                raise ValueError(
                    "single-step cascade prior cannot be combined with prior_proxy_mix"
                )
            if cascade_prior_source_mask is None:
                raise ValueError("active cascade prior requires a source mask")
            if cascade_prior_source_mask.shape != (x512.shape[0],):
                raise ValueError("cascade-prior source mask batch mismatch")
            if cascade_prior_source_mask.dtype != torch.bool:
                raise ValueError("cascade-prior source mask must be boolean")
        elif cascade_prior_source_mask is not None:
            raise ValueError("cascade-prior source mask requires an active mode")
        if cond_drop_mode not in {
            "joint",
            "independent",
            "independent_matched",
            "stratified",
        }:
            raise ValueError(
                "cond_drop_mode must be joint, independent, "
                "independent_matched, or stratified, "
                f"got {cond_drop_mode!r}"
            )
        if cond_drop_mode in {"independent_matched", "stratified"}:
            if cond_drop_masks is None:
                raise ValueError(
                    f"{cond_drop_mode} requires an explicit isolated condition mask"
                )
        elif cond_drop_masks is not None:
            raise ValueError(
                "explicit condition masks are reserved for matched/stratified modes"
            )
        if control_rank_weight < 0.0 or control_rank_margin < 0.0:
            raise ValueError(
                "control rank weight and margin must be non-negative"
            )
        if control_rank_scope not in {"single_branch", "routed_all"}:
            raise ValueError(
                "control_rank_scope must be single_branch or routed_all"
            )
        if factor_surrogate_weight < 0.0:
            raise ValueError("factor surrogate weight must be non-negative")
        if state_population_delta_weight < 0.0:
            raise ValueError(
                "state population delta weight must be non-negative"
            )
        if state_prompt_causal_mid_weight < 0.0:
            raise ValueError(
                "state prompt causal mid weight must be non-negative"
            )
        if state_prompt_causal_high_centroid_weight < 0.0:
            raise ValueError(
                "state prompt causal high centroid weight must be non-negative"
            )
        if state_trajectory_centroid_weight < 0.0:
            raise ValueError(
                "state trajectory centroid weight must be non-negative"
            )
        if state_trajectory_population_size < 0:
            raise ValueError(
                "state trajectory population size must be non-negative"
            )
        if (
            state_trajectory_centroid_weight > 0.0
            and state_trajectory_population_size <= 0
        ):
            raise ValueError(
                "state trajectory centroid loss requires population batches"
            )
        if state_early_ddim_centroid_weight < 0.0:
            raise ValueError(
                "state early DDIM centroid weight must be non-negative"
            )
        if state_early_ddim_population_size < 0:
            raise ValueError(
                "state early DDIM population size must be non-negative"
            )
        if (
            state_early_ddim_centroid_weight > 0.0
            and state_early_ddim_population_size <= 0
        ):
            raise ValueError(
                "state early DDIM centroid loss requires population batches"
            )
        if state_early_ddim_pool_size <= 0:
            raise ValueError("state early DDIM pool size must be positive")
        if state_early_ddim_inference_steps != 25:
            raise ValueError(
                "state early DDIM objective is frozen to a 25-step sampler"
            )
        if state_early_ddim_trace_steps != 2:
            raise ValueError(
                "state early DDIM objective is frozen to the first two steps"
            )
        if state_early_ddim_subject_cfg != 2.0:
            raise ValueError(
                "state early DDIM objective is frozen to subject CFG 2"
            )
        if (
            state_early_ddim_probe_gradients
            and state_early_ddim_centroid_weight <= 0.0
        ):
            raise ValueError(
                "state early DDIM gradient probe requires a positive raw-loss weight"
            )
        if (
            state_trajectory_population_size > 0
            and state_early_ddim_population_size > 0
        ):
            raise ValueError(
                "old trajectory and early DDIM population objectives cannot mix"
            )
        if state_population_size <= 0:
            raise ValueError("state population size must be positive")
        if (
            state_population_delta_weight > 0.0
            or state_prompt_causal_mid_weight > 0.0
            or state_prompt_causal_high_centroid_weight > 0.0
        ) and state_population_size <= 1:
            raise ValueError(
                "state population delta loss requires population size > 1"
            )
        if state_population_pool_size <= 0:
            raise ValueError("state population pool size must be positive")
        if not (
            0
            <= state_population_timestep_min
            <= state_population_timestep_max
            < cfg.num_train_timesteps
        ):
            raise ValueError("invalid state population timestep interval")
        if state_prompt_causal_pool_size <= 0:
            raise ValueError("state prompt causal pool size must be positive")
        if not (
            0
            <= state_prompt_causal_timestep_min
            <= state_prompt_causal_timestep_max
            < cfg.num_train_timesteps
        ):
            raise ValueError("invalid state prompt causal timestep interval")
        if state_prompt_causal_high_pool_size <= 0:
            raise ValueError(
                "state prompt causal high pool size must be positive"
            )
        if not (
            0
            <= state_prompt_causal_high_timestep_min
            <= state_prompt_causal_high_timestep_max
            < cfg.num_train_timesteps
        ):
            raise ValueError(
                "invalid state prompt causal high timestep interval"
            )
        if state_trajectory_pool_size <= 0:
            raise ValueError("state trajectory pool size must be positive")
        if not 0 <= state_trajectory_timestep < cfg.num_train_timesteps:
            raise ValueError("state trajectory timestep is invalid")
        if morphology_mask_surrogate_weight < 0.0:
            raise ValueError(
                "morphology mask surrogate weight must be non-negative"
            )
        if morphology_pair_preservation_weight < 0.0:
            raise ValueError(
                "morphology pair preservation weight must be non-negative"
            )
        if (
            morphology_pair_preservation_weight > 0.0
            and morphology_mask_surrogate_weight <= 0.0
        ):
            raise ValueError(
                "morphology pair preservation requires geometry supervision"
            )
        if factor_surrogate_state_population_size <= 0:
            raise ValueError(
                "factor surrogate state population size must be positive"
            )
        if not (
            0
            <= factor_surrogate_timestep_min
            <= factor_surrogate_timestep_max
            < cfg.num_train_timesteps
        ):
            raise ValueError("invalid factor surrogate timestep interval")
        if not (
            0
            <= morphology_mask_surrogate_timestep_min
            <= morphology_mask_surrogate_timestep_max
            < cfg.num_train_timesteps
        ):
            raise ValueError(
                "invalid morphology mask surrogate timestep interval"
            )
        if (
            morphology_mask_surrogate_weight > 0.0
            and control_rank_factor != 0
        ):
            raise ValueError(
                "morphology mask supervision is defined only for subject factor"
            )
        if control_rank_factor not in {0, 1, 2}:
            raise ValueError("control_rank_factor must be 0, 1, or 2")
        if counterfactual_pair_weight < 0.0:
            raise ValueError("counterfactual_pair_weight must be non-negative")
        if counterfactual_teacher_anchor_mix < 0.0:
            raise ValueError(
                "counterfactual_teacher_anchor_mix must be non-negative"
            )
        if counterfactual_target_orthogonal_mix < 0.0:
            raise ValueError(
                "counterfactual_target_orthogonal_mix must be non-negative"
            )
        if observation_rgb_pair_weight < 0.0:
            raise ValueError(
                "observation RGB pair weight must be non-negative"
            )
        if observation_rgb_pair_center_weight < 0.0:
            raise ValueError(
                "observation RGB pair center weight must be non-negative"
            )
        if not (
            0
            <= observation_rgb_pair_timestep_min
            <= observation_rgb_pair_timestep_max
            < cfg.num_train_timesteps
        ):
            raise ValueError(
                "invalid observation RGB pair timestep interval"
            )
        x256 = F.interpolate(x512, size=(256, 256), mode="bilinear", align_corners=False)
        x128 = F.interpolate(x512, size=(128, 128), mode="bilinear", align_corners=False)

        (z_l, z_m, z_h), (xr_l, xr_m, xr_h), (mu_l, mu_m, mu_h), (lv_l, lv_m, lv_h) = self.encode_triplet(
            x128, x256, x512
        )

        raw_conds = self.text_enc(subject_ids, state_ids, bg_ids)
        raw_routed_conds = self._route_conditions(
            raw_conds,
            subject_state_low_mask=subject_state_low_mask,
        )
        raw_routed_conds = self._apply_nucleus_prompt_adapters(
            raw_routed_conds,
            subject_state_low_mask,
        )
        # Classifier-free condition dropout is a diffusion-training device.
        # The image-text contrastive target must keep the complete canonical
        # triplet; otherwise independent dropout would make at least one
        # semantic factor disappear from about 27% of contrastive targets when
        # p=0.1, encouraging the factor embeddings themselves to blur.
        contrastive_pools = tuple(
            pooled for _sequence, pooled in raw_conds
        )
        conds = self._maybe_dropout_cond(
            raw_conds,
            cond_drop_prob,
            independent=cond_drop_mode == "independent",
            drop_masks=cond_drop_masks,
        )
        (c_sub_seq, c_sub_pool), (c_st_seq, c_st_pool), (c_bg_seq, c_bg_pool) = conds
        c_sub_seq, c_sub_pool = self._add_personalization(c_sub_seq, c_sub_pool, use_personalization)
        (
            (c_sub_seq, c_sub_pool),
            (c_st_seq, c_st_pool),
            (c_bg_seq, c_bg_pool),
        ) = self._route_conditions(
            (
                (c_sub_seq, c_sub_pool),
                (c_st_seq, c_st_pool),
                (c_bg_seq, c_bg_pool),
            ),
            subject_state_low_mask=subject_state_low_mask,
        )
        (
            (c_sub_seq, c_sub_pool),
            (c_st_seq, c_st_pool),
            (c_bg_seq, c_bg_pool),
        ) = self._apply_nucleus_prompt_adapters(
            (
                (c_sub_seq, c_sub_pool),
                (c_st_seq, c_st_pool),
                (c_bg_seq, c_bg_pool),
            ),
            subject_state_low_mask,
        )

        w_l, w_m, w_h = cfg.scale_weights()
        diff = self.diffusion
        losses: dict[str, torch.Tensor] = {}
        if cond_drop_masks is not None:
            metric_masks = cond_drop_masks.to(
                device=z_l.device,
                dtype=torch.int64,
            )
            for pattern_name, values in CONDITION_MASK_PATTERNS:
                pattern = torch.tensor(
                    values,
                    device=metric_masks.device,
                    dtype=metric_masks.dtype,
                )
                losses[f"metric_condition_mask_{pattern_name}_fraction"] = (
                    (metric_masks == pattern).all(dim=1).float().mean()
                )
            losses["metric_condition_mask_subject_retained_fraction"] = (
                1.0 - metric_masks[:, 0].float().mean()
            )
            losses["metric_condition_mask_state_retained_fraction"] = (
                1.0 - metric_masks[:, 1].float().mean()
            )
            losses["metric_condition_mask_background_retained_fraction"] = (
                1.0 - metric_masks[:, 2].float().mean()
            )

        def branch_prediction(
            x_t: torch.Tensor,
            unet: nn.Module,
            t: torch.Tensor,
            cond_seq: torch.Tensor,
            cond_pool: torch.Tensor,
            prior: torch.Tensor | None,
            factor_delta_context: torch.Tensor | None = None,
            factor_delta_mask: torch.Tensor | None = None,
            morphology_class_ids_override: torch.Tensor | None = None,
            morphology_route_mask_override: torch.Tensor | None = None,
            morphology_spatial_mask_override: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if (
                unet is self.unet_low
                and (
                    self.spatial_subject_input_adapter is not None
                    or self.morphology_latent_bank is not None
                )
                and (
                    spatial_subject_mask is not None
                    or morphology_class_ids is not None
                )
            ):
                structure_batch = (
                    spatial_subject_mask.shape[0]
                    if spatial_subject_mask is not None
                    else morphology_class_ids.shape[0]
                )
                if structure_batch != x_t.shape[0]:
                    raise ValueError(
                        "spatial subject mask batch size does not match low latent"
                    )
                route = spatial_subject_route_mask
                if route is None:
                    route = torch.ones(
                        x_t.shape[0],
                        device=x_t.device,
                        dtype=torch.bool,
                    )
                if route.ndim != 1 or route.shape[0] != x_t.shape[0]:
                    raise ValueError(
                        "spatial subject route mask must have shape [batch]"
                    )
                if self.spatial_subject_input_adapter is not None:
                    assert spatial_subject_mask is not None
                    spatial_residual = self.spatial_subject_input_adapter(
                        spatial_subject_mask.to(device=x_t.device),
                        x_t.shape[-2:],
                    )
                else:
                    assert self.morphology_latent_bank is not None
                    assert morphology_class_ids is not None
                    spatial_residual = self.morphology_latent_bank(
                        morphology_class_ids.to(device=x_t.device),
                        x_t.shape[-2:],
                    )
                spatial_residual = spatial_residual.to(dtype=x_t.dtype)
                x_t = torch.where(
                    route.to(device=x_t.device, dtype=torch.bool)[
                        :, None, None, None
                    ],
                    x_t + spatial_residual,
                    x_t,
                )
            if prior is not None:
                x_in = torch.cat([x_t, prior], dim=1)
            else:
                x_in = x_t
            active_morphology_class_ids = (
                morphology_class_ids_override
                if morphology_class_ids_override is not None
                else morphology_class_ids
            )
            active_morphology_route_mask = (
                morphology_route_mask_override
                if morphology_route_mask_override is not None
                else spatial_subject_route_mask
            )
            active_morphology_spatial_mask = (
                morphology_spatial_mask_override
                if morphology_spatial_mask_override is not None
                else spatial_subject_mask
            )
            pred = unet(
                x_in,
                t,
                cond_seq,
                cond_pool,
                factor_delta_context=factor_delta_context,
                factor_delta_mask=factor_delta_mask,
                morphology_class_ids=(
                    active_morphology_class_ids
                    if unet is self.unet_low
                    and cfg.conditioning_mode
                    == MORPHOLOGY_FEATURE_AFFINE_MODE
                    else None
                ),
                morphology_route_mask=(
                    active_morphology_route_mask
                    if unet is self.unet_low
                    and cfg.conditioning_mode
                    in {
                        MORPHOLOGY_FEATURE_AFFINE_MODE,
                        MORPHOLOGY_SPATIAL_AFFINE_MODE,
                    }
                    else None
                ),
                morphology_spatial_mask=(
                    active_morphology_spatial_mask
                    if unet is self.unet_low
                    and cfg.conditioning_mode
                    == MORPHOLOGY_SPATIAL_AFFINE_MODE
                    else None
                ),
            )
            if self.rgb_factor_output_adapters is not None:
                if unet is self.unet_low:
                    rgb_branch = "low"
                elif unet is self.unet_mid:
                    rgb_branch = "mid"
                elif unet is self.unet_high:
                    rgb_branch = "high"
                else:
                    raise ValueError("unknown U-Net for RGB factor adapters")
                pred = self._apply_rgb_factor_output_adapters(
                    pred,
                    t,
                    c_st_pool,
                    c_bg_pool,
                    rgb_branch,
                )
            nucleus_adapter = None
            if unet is self.unet_low:
                nucleus_adapter = self.nucleus_low_adapter
            elif unet is self.unet_mid:
                nucleus_adapter = self.nucleus_mid_adapter
            elif unet is self.unet_high:
                nucleus_adapter = self.nucleus_high_adapter
            if nucleus_adapter is not None:
                if subject_state_low_mask is None:
                    raise ValueError(
                        "nucleus branch adapter requires subject_state_low_mask"
                    )
                residual = nucleus_adapter(pred, t, cond_pool)
                adapter_mask = nucleus_adapter_route_mask(
                    cfg.conditioning_mode,
                    subject_state_low_mask,
                    t,
                    cfg.nucleus_temporal_adapter_max_timestep,
                )
                routed_prediction = pred + residual
                pred = torch.where(
                    adapter_mask.to(device=pred.device)[:, None, None, None],
                    routed_prediction,
                    pred,
                )
            return pred

        def diffusion_loss(
            z: torch.Tensor,
            unet: nn.Module,
            t: torch.Tensor,
            cond_seq: torch.Tensor,
            cond_pool: torch.Tensor,
            prior: torch.Tensor | None,
            return_x0_with_grad: bool = False,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor | None,
            dict[str, torch.Tensor | None],
        ]:
            noise = torch.randn_like(z)
            x_t = diff.q_sample(z, t, noise)
            pred = branch_prediction(
                x_t,
                unet,
                t,
                cond_seq,
                cond_pool,
                prior,
            )
            if (
                state_early_ddim_centroid_weight > 0.0
                and unet is self.unet_low
            ):
                def record_base_low_gradient(gradient: torch.Tensor) -> None:
                    self._last_state_early_ddim_activation_gradient_rms[
                        "base_low"
                    ] = float(
                        gradient.detach().float().square().mean().sqrt()
                    )

                pred.register_hook(record_base_low_gradient)
            target = diff.training_target(z, t, noise, cfg.prediction_type)
            predicted_x0 = None
            if prior_proxy_mix > 0.0 or return_x0_with_grad:
                predicted_x0 = diff.model_output_to_x0_eps(
                    x_t,
                    t,
                    pred,
                    cfg.prediction_type,
                )[0]
                if not return_x0_with_grad:
                    predicted_x0 = predicted_x0.detach()
            return (
                foreground_mixed_mse(
                    pred,
                    target,
                    foreground_mask,
                    foreground_loss_mix,
                ),
                predicted_x0,
                {
                    "x_t": x_t,
                    "t": t,
                    "target": target,
                    "prior": prior,
                    "prediction": pred,
                },
            )

        # 独立时间步（式 4-36 求和）
        b = z_l.shape[0]
        if state_trajectory_population_size > 0:
            t_l = torch.randint(
                0, cfg.num_train_timesteps, (1,), device=z_l.device
            ).expand(b)
            t_m = torch.randint(
                0, cfg.num_train_timesteps, (1,), device=z_l.device
            ).expand(b)
        else:
            t_l = torch.randint(
                0, cfg.num_train_timesteps, (b,), device=z_l.device
            )
            t_m = torch.randint(
                0, cfg.num_train_timesteps, (b,), device=z_l.device
            )
        if state_population_size > 1 or state_trajectory_population_size > 0:
            shared_t_h = torch.randint(
                state_population_timestep_min,
                state_population_timestep_max + 1,
                (1,),
                device=z_l.device,
            )
            t_h = shared_t_h.expand(b)
        else:
            t_h = torch.randint(
                0,
                cfg.num_train_timesteps,
                (b,),
                device=z_l.device,
            )

        low_loss, predicted_z_l, low_rank_context = diffusion_loss(
            z_l,
            self.unet_low,
            t_l,
            c_sub_seq,
            c_sub_pool,
            None,
            return_x0_with_grad=cascade_prior_active,
        )
        if prior_proxy_mix > 0.0 or cascade_prior_active:
            assert predicted_z_l is not None
        teacher_prior_m = F.interpolate(
            z_l,
            size=z_m.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        prior_m = teacher_prior_m
        if predicted_z_l is not None:
            predicted_prior_m = F.interpolate(
                predicted_z_l,
                size=z_m.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            if cascade_prior_active:
                assert cascade_prior_source_mask is not None
                prior_m = select_single_step_cascade_prior(
                    teacher_prior_m,
                    predicted_prior_m,
                    cascade_prior_source_mask,
                    cascade_prior_gradient_mode,
                )

                def record_low_downstream_gradient(
                    gradient: torch.Tensor,
                ) -> None:
                    self._last_cascade_prior_activation_gradient_rms[
                        "low_from_mid_high"
                    ] = float(
                        gradient.detach().float().square().mean().sqrt()
                    )

                predicted_z_l.register_hook(record_low_downstream_gradient)
            else:
                prior_m = mix_detached_prior_proxy(
                    teacher_prior_m,
                    predicted_prior_m,
                    prior_proxy_mix,
                )
        mid_loss, predicted_z_m, mid_rank_context = diffusion_loss(
            z_m,
            self.unet_mid,
            t_m,
            c_st_seq,
            c_st_pool,
            prior_m,
            return_x0_with_grad=cascade_prior_active,
        )
        if state_prompt_causal_mid_weight > 0.0:
            if state_population_labels is None:
                raise ValueError(
                    "state prompt causal loss requires state labels"
                )
            if b != 2 * state_population_size:
                raise ValueError(
                    "state prompt causal batch size must equal twice the "
                    "per-state population size"
                )
            # This auxiliary path deliberately removes every image-derived
            # state cue available to the mid U-Net.  Both prompt arms share
            # one group-mean latent carrier, one group-mean low prior, one
            # timestep and paired noise.  The routed state text is the only
            # variable that can explain the untreated-to-positive target.
            (
                neutral_z_m,
                neutral_z_l,
                causal_noise,
                negative_mask,
                positive_mask,
            ) = state_prompt_causal_neutral_inputs(
                z_m,
                z_l,
                state_population_labels,
            )
            if (
                int(negative_mask.sum().item()) != state_population_size
                or int(positive_mask.sum().item()) != state_population_size
            ):
                raise ValueError(
                    "state prompt causal batch must contain equal state groups"
                )
            neutral_prior_m = F.interpolate(
                neutral_z_l,
                size=z_m.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).expand(b, -1, -1, -1)
            shared_t_mid = torch.randint(
                state_prompt_causal_timestep_min,
                state_prompt_causal_timestep_max + 1,
                (1,),
                device=z_m.device,
            )
            causal_t_mid = shared_t_mid.expand(b)
            causal_x_t_mid = diff.q_sample(
                neutral_z_m,
                causal_t_mid,
                causal_noise,
            )
            raw_mid_seq, raw_mid_pool = raw_routed_conds[1]
            causal_prediction_mid = branch_prediction(
                causal_x_t_mid,
                self.unet_mid,
                causal_t_mid,
                raw_mid_seq,
                raw_mid_pool,
                neutral_prior_m,
            )
            causal_x0_mid = diff.model_output_to_x0_eps(
                causal_x_t_mid,
                causal_t_mid,
                causal_prediction_mid,
                cfg.prediction_type,
            )[0]
            causal_mid_loss, causal_mid_metrics = (
                state_population_delta_loss(
                    causal_x0_mid,
                    z_m,
                    state_population_labels,
                    pool_size=state_prompt_causal_pool_size,
                )
            )
            losses["loss_state_prompt_causal_mid"] = (
                state_prompt_causal_mid_weight * causal_mid_loss
            )
            losses["metric_state_prompt_causal_mid_raw"] = (
                causal_mid_loss.detach()
            )
            losses["metric_state_prompt_causal_mid_cosine"] = (
                causal_mid_metrics["cosine"]
            )
            losses["metric_state_prompt_causal_mid_predicted_rms"] = (
                causal_mid_metrics["predicted_rms"]
            )
            losses["metric_state_prompt_causal_mid_target_rms"] = (
                causal_mid_metrics["target_rms"]
            )
        if prior_proxy_mix > 0.0 or cascade_prior_active:
            assert predicted_z_m is not None
        teacher_prior_h = F.interpolate(
            z_m,
            size=z_h.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        prior_h = teacher_prior_h
        if predicted_z_m is not None:
            predicted_prior_h = F.interpolate(
                predicted_z_m,
                size=z_h.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            if cascade_prior_active:
                assert cascade_prior_source_mask is not None
                prior_h = select_single_step_cascade_prior(
                    teacher_prior_h,
                    predicted_prior_h,
                    cascade_prior_source_mask,
                    cascade_prior_gradient_mode,
                )

                def record_mid_downstream_gradient(
                    gradient: torch.Tensor,
                ) -> None:
                    self._last_cascade_prior_activation_gradient_rms[
                        "mid_from_high"
                    ] = float(
                        gradient.detach().float().square().mean().sqrt()
                    )

                predicted_z_m.register_hook(record_mid_downstream_gradient)
            else:
                prior_h = mix_detached_prior_proxy(
                    teacher_prior_h,
                    predicted_prior_h,
                    prior_proxy_mix,
                )
        high_loss, predicted_z_h, high_rank_context = diffusion_loss(
            z_h,
            self.unet_high,
            t_h,
            c_bg_seq,
            c_bg_pool,
            prior_h,
            return_x0_with_grad=(
                state_population_delta_weight > 0.0
            ),
        )
        losses["loss_low"] = w_l * low_loss
        losses["loss_mid"] = w_m * mid_loss
        losses["loss_high"] = w_h * high_loss
        if cascade_prior_active:
            assert cascade_prior_source_mask is not None
            losses["metric_cascade_prior_predicted_fraction"] = (
                cascade_prior_source_mask.float().mean().detach()
            )
            losses["metric_cascade_prior_mid_predicted_rmse_clean"] = (
                (predicted_prior_m.detach().float() - teacher_prior_m.float())
                .square()
                .mean()
                .sqrt()
            )
            losses["metric_cascade_prior_high_predicted_rmse_clean"] = (
                (predicted_prior_h.detach().float() - teacher_prior_h.float())
                .square()
                .mean()
                .sqrt()
            )
            losses["metric_cascade_prior_e2e_enabled"] = z_l.new_tensor(
                1.0 if cascade_prior_gradient_mode == "e2e" else 0.0
            )
            if cascade_prior_capture_forward:
                tensors = {
                    "low_predicted_x0": predicted_z_l,
                    "mid_predicted_x0": predicted_z_m,
                    "mid_selected_prior": prior_m,
                    "high_selected_prior": prior_h,
                    "low_noise_prediction": low_rank_context["prediction"],
                    "mid_noise_prediction": mid_rank_context["prediction"],
                    "high_noise_prediction": high_rank_context["prediction"],
                }
                fingerprints: dict[str, Any] = {}
                for name, value in tensors.items():
                    assert value is not None
                    cpu = value.detach().float().cpu().contiguous()
                    raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
                    fingerprints[name] = {
                        "shape": list(cpu.shape),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "mean": float(cpu.mean()),
                        "rms": float(cpu.square().mean().sqrt()),
                    }
                self._last_cascade_prior_forward_fingerprints = fingerprints

        if state_early_ddim_centroid_weight > 0.0:
            if state_population_labels is None:
                raise ValueError(
                    "state early DDIM centroid loss requires state labels"
                )
            if cfg.conditioning_mode != SEMANTIC_STATE_ALLSCALE_MODE:
                raise ValueError(
                    "state early DDIM centroid loss requires "
                    "semantic_state_allscale conditioning"
                )
            classes = int(torch.unique(state_population_labels).numel())
            if b != classes * state_early_ddim_population_size:
                raise ValueError(
                    "state early DDIM batch size must equal class count "
                    "times population size"
                )
            samples_per_class = state_early_ddim_population_size
            expected_labels = torch.arange(
                classes,
                device=z_l.device,
                dtype=state_population_labels.dtype,
            ).repeat_interleave(samples_per_class)
            if not torch.equal(state_population_labels, expected_labels):
                raise ValueError(
                    "state early DDIM labels must be grouped in class order"
                )
            if state_early_ddim_generator is None:
                raise ValueError(
                    "state early DDIM centroid loss requires an isolated generator"
                )

            # Formal inference begins with FP32 raw standard-normal noise. Every
            # source-local state arm receives the same per-member noise so the
            # state text is the only changing factor inside a group.
            base_noise = torch.randn(
                z_l[:samples_per_class].shape,
                device=z_l.device,
                generator=state_early_ddim_generator,
            )
            early_x = torch.cat(
                [base_noise for _ in range(classes)], dim=0
            )
            predicted_x0_values, guided_predictions = (
                self.state_early_ddim_low_trajectory(
                    early_x,
                    raw_conds,
                    inference_steps=state_early_ddim_inference_steps,
                    trace_steps=state_early_ddim_trace_steps,
                    subject_cfg=state_early_ddim_subject_cfg,
                    subject_state_low_mask=subject_state_low_mask,
                    # The auxiliary objective teaches the denoiser to use the
                    # existing semantic embeddings.  It must not reduce its
                    # loss by moving the learnable null-subject token or text
                    # projection heads away from formal CFG semantics.
                    detach_conditions=True,
                )
            )
            step_losses: list[torch.Tensor] = []
            for step_index, (predicted_x0, guided_prediction) in enumerate(
                zip(predicted_x0_values, guided_predictions)
            ):
                def record_early_ddim_activation_gradient(
                    gradient: torch.Tensor,
                    *,
                    name: str = f"step{step_index + 1}",
                ) -> None:
                    self._last_state_early_ddim_activation_gradient_rms[
                        name
                    ] = float(
                        gradient.detach().float().square().mean().sqrt()
                    )

                guided_prediction.register_hook(
                    record_early_ddim_activation_gradient
                )
                step_loss, step_metrics = state_multiclass_centroid_loss(
                    predicted_x0,
                    z_l,
                    state_population_labels,
                    pool_size=state_early_ddim_pool_size,
                )
                step_losses.append(step_loss)
                losses[
                    f"metric_state_early_ddim_step{step_index + 1}_raw"
                ] = step_loss.detach()
                for metric_name, metric_value in step_metrics.items():
                    losses[
                        f"metric_state_early_ddim_step{step_index + 1}_"
                        f"{metric_name}"
                    ] = metric_value
            losses["loss_state_early_ddim_centroid_low"] = (
                state_early_ddim_centroid_weight
                * torch.stack(step_losses).mean()
            )

            if state_early_ddim_probe_gradients:
                named_low_parameters = tuple(
                    (name, parameter)
                    for name, parameter in self.unet_low.named_parameters()
                    if parameter.requires_grad
                    # E1 carries registered factor-delta attention adapters,
                    # but semantic_state_allscale does not route a factor-
                    # delta context through them.  They are therefore outside
                    # both objectives' active computation graph and must not
                    # be counted as zero gradients in this calibration.
                    and ".factor_delta_" not in name
                    and not name.startswith(
                        (
                            "down_state_delta_affines.",
                            "mid_state_delta_affines.",
                            "up_state_delta_affines.",
                            "down_target_state_response_affines.",
                            "mid_target_state_response_affines.",
                            "up_target_state_response_affines.",
                            "down_morphology_affines.",
                            "mid_morphology_affines.",
                            "up_morphology_affines.",
                            "down_morphology_spatial_affines.",
                            "mid_morphology_spatial_affines.",
                            "up_morphology_spatial_affines.",
                        )
                    )
                )
                if not named_low_parameters:
                    raise ValueError(
                        "state early DDIM gradient probe requires trainable low U-Net parameters"
                    )
                low_parameters = tuple(
                    parameter for _name, parameter in named_low_parameters
                )
                parameter_name_digest = hashlib.sha256(
                    "\n".join(
                        name for name, _parameter in named_low_parameters
                    ).encode("utf-8")
                ).hexdigest()

                def parameter_gradient_sums(
                    objective: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    gradients = torch.autograd.grad(
                        objective,
                        low_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    sum_sq = objective.new_zeros((), dtype=torch.float64)
                    count = objective.new_zeros((), dtype=torch.float64)
                    missing = [
                        name
                        for (name, _parameter), gradient in zip(
                            named_low_parameters, gradients
                        )
                        if gradient is None
                    ]
                    if missing:
                        raise RuntimeError(
                            "state early DDIM gradient probe found unused low "
                            f"U-Net parameters: {missing[:5]}"
                        )
                    for parameter, gradient in zip(low_parameters, gradients):
                        assert gradient is not None
                        measured_gradient = gradient.detach().float()
                        if dist.is_available() and dist.is_initialized():
                            # Reproduce the gradient vector that ordinary DDP
                            # would apply: average each parameter gradient
                            # across ranks before computing its norm.  Averaging
                            # per-rank squared norms would miss cancellation and
                            # would not bound the future optimizer update.
                            dist.all_reduce(
                                measured_gradient,
                                op=dist.ReduceOp.SUM,
                            )
                            measured_gradient.div_(dist.get_world_size())
                        sum_sq = (
                            sum_sq
                            + measured_gradient.double().square().sum()
                        )
                        count = count + parameter.numel()
                    return sum_sq, count

                base_sum_sq, base_count = parameter_gradient_sums(
                    losses["loss_low"]
                )
                auxiliary_sum_sq, auxiliary_count = parameter_gradient_sums(
                    torch.stack(step_losses).mean()
                )
                self._last_state_early_ddim_parameter_gradient_stats = {
                    "class_count": torch.as_tensor(
                        classes, device=z_l.device, dtype=torch.int64
                    ),
                    "base_sum_sq": base_sum_sq,
                    "base_count": base_count,
                    "auxiliary_sum_sq": auxiliary_sum_sq,
                    "auxiliary_count": auxiliary_count,
                    "parameter_tensor_count": torch.as_tensor(
                        len(named_low_parameters),
                        device=z_l.device,
                        dtype=torch.int64,
                    ),
                    "parameter_name_digest": parameter_name_digest,
                    "gradient_aggregation": (
                        "mean_parameter_gradient_across_ranks_then_rms"
                        if dist.is_available() and dist.is_initialized()
                        else "single_rank_parameter_gradient_rms"
                    ),
                }

        if state_trajectory_centroid_weight > 0.0:
            if state_population_labels is None:
                raise ValueError(
                    "state trajectory centroid loss requires state labels"
                )
            if cfg.conditioning_mode != SEMANTIC_STATE_ALLSCALE_MODE:
                raise ValueError(
                    "state trajectory centroid loss requires "
                    "semantic_state_allscale conditioning"
                )
            classes = int(torch.unique(state_population_labels).numel())
            if b != classes * state_trajectory_population_size:
                raise ValueError(
                    "state trajectory batch size must equal class count "
                    "times population size"
                )
            samples_per_class = state_trajectory_population_size
            expected_labels = torch.arange(
                classes, device=z_l.device, dtype=state_population_labels.dtype
            ).repeat_interleave(samples_per_class)
            if not torch.equal(state_population_labels, expected_labels):
                raise ValueError(
                    "state trajectory labels must be grouped in class order"
                )
            if state_trajectory_generator is None:
                raise ValueError(
                    "state trajectory centroid loss requires an isolated generator"
                )
            base_noise_l = torch.randn(
                z_l[:samples_per_class].shape,
                device=z_l.device,
                dtype=z_l.dtype,
                generator=state_trajectory_generator,
            )
            noise_l = torch.cat(
                [base_noise_l for _ in range(classes)], dim=0
            )
            trajectory_t = torch.full(
                (b,),
                state_trajectory_timestep,
                device=z_l.device,
                dtype=torch.long,
            )
            noisy_l = diff.q_sample(
                torch.zeros_like(z_l), trajectory_t, noise_l
            )
            raw_low_seq, raw_low_pool = raw_routed_conds[0]
            trajectory_prediction_l = branch_prediction(
                noisy_l,
                self.unet_low,
                trajectory_t,
                raw_low_seq,
                raw_low_pool,
                None,
            )
            trajectory_z_l = diff.model_output_to_x0_eps(
                noisy_l,
                trajectory_t,
                trajectory_prediction_l,
                cfg.prediction_type,
            )[0]
            base_noise_m = torch.randn(
                z_m[:samples_per_class].shape,
                device=z_m.device,
                dtype=z_m.dtype,
                generator=state_trajectory_generator,
            )
            noise_m = torch.cat(
                [base_noise_m for _ in range(classes)], dim=0
            )
            noisy_m = diff.q_sample(
                torch.zeros_like(z_m), trajectory_t, noise_m
            )
            trajectory_prior_m = F.interpolate(
                trajectory_z_l,
                size=z_m.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            raw_mid_seq, raw_mid_pool = raw_routed_conds[1]
            trajectory_prediction_m = branch_prediction(
                noisy_m,
                self.unet_mid,
                trajectory_t,
                raw_mid_seq,
                raw_mid_pool,
                trajectory_prior_m,
            )
            trajectory_z_m = diff.model_output_to_x0_eps(
                noisy_m,
                trajectory_t,
                trajectory_prediction_m,
                cfg.prediction_type,
            )[0]
            base_noise_h = torch.randn(
                z_h[:samples_per_class].shape,
                device=z_h.device,
                dtype=z_h.dtype,
                generator=state_trajectory_generator,
            )
            noise_h = torch.cat(
                [base_noise_h for _ in range(classes)], dim=0
            )
            noisy_h = diff.q_sample(
                torch.zeros_like(z_h), trajectory_t, noise_h
            )
            trajectory_prior_h = F.interpolate(
                trajectory_z_m,
                size=z_h.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            raw_high_seq, raw_high_pool = raw_routed_conds[2]
            trajectory_prediction_h = branch_prediction(
                noisy_h,
                self.unet_high,
                trajectory_t,
                raw_high_seq,
                raw_high_pool,
                trajectory_prior_h,
            )
            trajectory_z_h = diff.model_output_to_x0_eps(
                noisy_h,
                trajectory_t,
                trajectory_prediction_h,
                cfg.prediction_type,
            )[0]

            def record_trajectory_activation_gradient(
                branch_name: str,
            ):
                def hook(gradient: torch.Tensor) -> None:
                    self._last_state_trajectory_activation_gradient_rms[
                        branch_name
                    ] = float(
                        gradient.detach().float().square().mean().sqrt()
                    )

                return hook

            for branch_name, predicted in (
                ("low", trajectory_z_l),
                ("mid", trajectory_z_m),
                ("high", trajectory_z_h),
            ):
                predicted.register_hook(
                    record_trajectory_activation_gradient(branch_name)
                )
            for branch_name, predicted, target in (
                ("low", trajectory_z_l, z_l),
                ("mid", trajectory_z_m, z_m),
                ("high", trajectory_z_h, z_h),
            ):
                branch_loss, branch_metrics = state_multiclass_centroid_loss(
                    predicted,
                    target,
                    state_population_labels,
                    pool_size=state_trajectory_pool_size,
                )
                losses[f"loss_state_trajectory_centroid_{branch_name}"] = (
                    state_trajectory_centroid_weight * branch_loss / 3.0
                )
                losses[f"metric_state_trajectory_{branch_name}_raw"] = (
                    branch_loss.detach()
                )
                for metric_name, metric_value in branch_metrics.items():
                    losses[
                        f"metric_state_trajectory_{branch_name}_{metric_name}"
                    ] = metric_value

        if state_prompt_causal_high_centroid_weight > 0.0:
            if state_population_labels is None:
                raise ValueError(
                    "state prompt causal high centroid loss requires state labels"
                )
            if b != 2 * state_population_size:
                raise ValueError(
                    "state prompt causal high batch size must equal twice the "
                    "per-state population size"
                )
            (
                neutral_z_h,
                neutral_z_m,
                causal_noise_h,
                negative_mask,
                positive_mask,
            ) = state_prompt_causal_neutral_inputs(
                z_h,
                z_m,
                state_population_labels,
            )
            if (
                int(negative_mask.sum().item()) != state_population_size
                or int(positive_mask.sum().item()) != state_population_size
            ):
                raise ValueError(
                    "state prompt causal high batch must contain equal state groups"
                )
            neutral_prior_h = F.interpolate(
                neutral_z_m,
                size=z_h.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).expand(b, -1, -1, -1)
            shared_t_high = torch.randint(
                state_prompt_causal_high_timestep_min,
                state_prompt_causal_high_timestep_max + 1,
                (1,),
                device=z_h.device,
            )
            causal_t_high = shared_t_high.expand(b)
            causal_x_t_high = diff.q_sample(
                neutral_z_h,
                causal_t_high,
                causal_noise_h,
            )
            raw_high_seq, raw_high_pool = raw_routed_conds[2]
            causal_prediction_high = branch_prediction(
                causal_x_t_high,
                self.unet_high,
                causal_t_high,
                raw_high_seq,
                raw_high_pool,
                neutral_prior_h,
            )
            causal_x0_high = diff.model_output_to_x0_eps(
                causal_x_t_high,
                causal_t_high,
                causal_prediction_high,
                cfg.prediction_type,
            )[0]
            high_centroid_loss, high_centroid_metrics = (
                state_population_centroid_loss(
                    causal_x0_high,
                    z_h,
                    state_population_labels,
                    pool_size=state_prompt_causal_high_pool_size,
                )
            )
            losses["loss_state_prompt_causal_high_centroid"] = (
                state_prompt_causal_high_centroid_weight
                * high_centroid_loss
            )
            losses["metric_state_prompt_causal_high_centroid_raw"] = (
                high_centroid_loss.detach()
            )
            for metric_name, metric_value in high_centroid_metrics.items():
                losses[
                    f"metric_state_prompt_causal_high_{metric_name}"
                ] = metric_value

        if state_population_delta_weight > 0.0:
            if state_population_labels is None:
                raise ValueError(
                    "state population delta loss requires state labels"
                )
            if b != 2 * state_population_size:
                raise ValueError(
                    "state population batch size must equal twice the "
                    "per-state population size"
                )
            assert predicted_z_h is not None
            population_loss, population_metrics = (
                state_population_delta_loss(
                    predicted_z_h,
                    z_h,
                    state_population_labels,
                    pool_size=state_population_pool_size,
                )
            )
            losses["loss_state_population_delta"] = (
                state_population_delta_weight * population_loss
            )
            losses["metric_state_population_delta_raw"] = (
                population_loss.detach()
            )
            losses["metric_state_population_delta_cosine"] = (
                population_metrics["cosine"]
            )
            losses["metric_state_population_predicted_rms"] = (
                population_metrics["predicted_rms"]
            )
            losses["metric_state_population_target_rms"] = (
                population_metrics["target_rms"]
            )

        if (
            control_rank_weight > 0.0
            or factor_surrogate_weight > 0.0
            or morphology_mask_surrogate_weight > 0.0
        ):
            required_rank_tensors = (
                negative_subject_ids,
                negative_state_ids,
                negative_bg_ids,
                control_rank_enabled,
            )
            if any(value is None for value in required_rank_tensors):
                raise ValueError(
                    "control intervention requires negative subject/state/background "
                    "ids and an enabled mask"
                )
            assert negative_subject_ids is not None
            assert negative_state_ids is not None
            assert negative_bg_ids is not None
            assert control_rank_enabled is not None
            if control_rank_enabled.shape != (b, 3):
                raise ValueError(
                    "control_rank_enabled must have shape [batch, 3]"
                )
            factor_mask = control_rank_enabled[:, control_rank_factor].to(
                device=x512.device,
                dtype=torch.bool,
            )
            if bool(factor_mask.any()):
                alternative_ids = [
                    (
                        negative_subject_ids,
                        state_ids,
                        bg_ids,
                    ),
                    (
                        subject_ids,
                        negative_state_ids,
                        bg_ids,
                    ),
                    (
                        subject_ids,
                        state_ids,
                        negative_bg_ids,
                    ),
                ][control_rank_factor]

                def routed_rank_conditions(
                    conditions,
                ) -> tuple[
                    tuple[torch.Tensor, torch.Tensor],
                    tuple[torch.Tensor, torch.Tensor],
                    tuple[torch.Tensor, torch.Tensor],
                ]:
                    (subject_cond, state_cond, background_cond) = conditions
                    subject_cond = self._add_personalization(
                        *subject_cond,
                        use_personalization,
                    )
                    routed = self._route_conditions(
                        (
                            subject_cond,
                            state_cond,
                            background_cond,
                        ),
                        subject_state_low_mask=subject_state_low_mask,
                    )
                    return self._apply_nucleus_prompt_adapters(
                        routed,
                        subject_state_low_mask,
                    )

                positive_rank_conds = routed_rank_conditions(raw_conds)
                negative_rank_conds = routed_rank_conditions(
                    self.text_enc(*alternative_ids)
                )
                if cfg.conditioning_mode == FACTOR_DELTA_ADAPTER_MODE:
                    negative_rank_conds = self._apply_factor_delta_adapters(
                        negative_rank_conds,
                        positive_rank_conds,
                        control_rank_factor,
                    )
                factor_attention_delta_context = None
                factor_attention_delta_mask = None
                if (
                    cfg.conditioning_mode
                    == FACTOR_ATTENTION_DELTA_ADAPTER_MODE
                ):
                    factor_attention_delta_context = (
                        negative_rank_conds[control_rank_factor][0]
                        - positive_rank_conds[control_rank_factor][0]
                    )
                    factor_attention_delta_mask = factor_mask
                rank_specs = (
                    (
                        self.unet_low,
                        low_rank_context,
                        positive_rank_conds[0],
                        negative_rank_conds[0],
                    ),
                    (
                        self.unet_mid,
                        mid_rank_context,
                        positive_rank_conds[1],
                        negative_rank_conds[1],
                    ),
                    (
                        self.unet_high,
                        high_rank_context,
                        positive_rank_conds[2],
                        negative_rank_conds[2],
                    ),
                )
                (
                    rank_unet,
                    rank_context,
                    positive_condition,
                    negative_condition,
                ) = rank_specs[control_rank_factor]
                rank_x_t = rank_context["x_t"]
                rank_t = rank_context["t"]
                rank_target = rank_context["target"]
                assert isinstance(rank_x_t, torch.Tensor)
                assert isinstance(rank_t, torch.Tensor)
                assert isinstance(rank_target, torch.Tensor)
                if control_rank_weight > 0.0:
                    if control_rank_scope == "single_branch":
                        rank_indices = (control_rank_factor,)
                    else:
                        if factor_attention_delta_context is not None:
                            raise ValueError(
                                "routed_all rank is not defined for attention delta mode"
                            )
                        rank_indices = tuple(
                            index
                            for index in range(3)
                            if not (
                                torch.equal(
                                    positive_rank_conds[index][0],
                                    negative_rank_conds[index][0],
                                )
                                and torch.equal(
                                    positive_rank_conds[index][1],
                                    negative_rank_conds[index][1],
                                )
                            )
                        )
                        if not rank_indices:
                            raise ValueError(
                                "routed_all rank found no scale affected by the factor"
                            )
                    rank_losses = []
                    for rank_index in rank_indices:
                        (
                            scoped_unet,
                            scoped_context,
                            scoped_positive_condition,
                            scoped_negative_condition,
                        ) = rank_specs[rank_index]
                        scoped_x_t = scoped_context["x_t"]
                        scoped_t = scoped_context["t"]
                        scoped_target = scoped_context["target"]
                        assert isinstance(scoped_x_t, torch.Tensor)
                        assert isinstance(scoped_t, torch.Tensor)
                        assert isinstance(scoped_target, torch.Tensor)
                        positive_prediction = branch_prediction(
                            scoped_x_t,
                            scoped_unet,
                            scoped_t,
                            *scoped_positive_condition,
                            scoped_context["prior"],
                        )
                        negative_prediction = branch_prediction(
                            scoped_x_t,
                            scoped_unet,
                            scoped_t,
                            *scoped_negative_condition,
                            scoped_context["prior"],
                        )
                        positive_error = foreground_mixed_mse_per_sample(
                            positive_prediction,
                            scoped_target,
                            foreground_mask,
                            foreground_loss_mix,
                        )
                        negative_error = foreground_mixed_mse_per_sample(
                            negative_prediction,
                            scoped_target,
                            foreground_mask,
                            foreground_loss_mix,
                        )
                        rank_violation = F.relu(
                            (1.0 + control_rank_margin) * positive_error
                            - negative_error
                        )
                        rank_losses.append(
                            rank_violation[factor_mask].mean()
                        )
                    rank_loss = torch.stack(rank_losses).mean()
            else:
                rank_loss = x512.new_zeros(())
            if control_rank_weight > 0.0:
                losses["loss_control_rank"] = control_rank_weight * rank_loss

            if factor_surrogate_weight > 0.0:
                if factor_surrogate_bundle is None:
                    raise ValueError(
                        "factor surrogate weight requires a frozen bundle"
                    )
                if factor_surrogate_target_ids is None:
                    raise ValueError(
                        "factor surrogate weight requires target class ids"
                    )
                if factor_surrogate_target_ids.shape != (b, 3):
                    raise ValueError(
                        "factor_surrogate_target_ids must have shape [batch,3]"
                    )
                if bool(factor_mask.any()):
                    surrogate_specs = (
                        (
                            z_l,
                            self.unet_low,
                            negative_rank_conds[0],
                            None,
                            "subject",
                        ),
                        (
                            z_m,
                            self.unet_mid,
                            negative_rank_conds[1],
                            teacher_prior_m,
                            "state_dnn",
                        ),
                        (
                            z_h,
                            self.unet_high,
                            negative_rank_conds[2],
                            teacher_prior_h,
                            "background",
                        ),
                    )
                    (
                        surrogate_z,
                        surrogate_unet,
                        surrogate_condition,
                        surrogate_prior,
                        surrogate_name,
                    ) = surrogate_specs[control_rank_factor]
                    surrogate_t = torch.randint(
                        factor_surrogate_timestep_min,
                        factor_surrogate_timestep_max + 1,
                        (b,),
                        device=surrogate_z.device,
                    )
                    surrogate_noise = torch.randn_like(surrogate_z)
                    surrogate_x_t = diff.q_sample(
                        surrogate_z,
                        surrogate_t,
                        surrogate_noise,
                    )
                    surrogate_prediction = branch_prediction(
                        surrogate_x_t,
                        surrogate_unet,
                        surrogate_t,
                        *surrogate_condition,
                        surrogate_prior,
                        factor_delta_context=(
                            factor_attention_delta_context
                        ),
                        factor_delta_mask=factor_attention_delta_mask,
                    )
                    surrogate_x0 = diff.model_output_to_x0_eps(
                        surrogate_x_t,
                        surrogate_t,
                        surrogate_prediction,
                        cfg.prediction_type,
                    )[0][factor_mask]
                    target_ids = factor_surrogate_target_ids[
                        factor_mask,
                        control_rank_factor,
                    ]
                    if bool((target_ids < 0).any()):
                        raise ValueError(
                            "enabled factor surrogate target contains -1"
                        )
                    if surrogate_name == "subject":
                        if (
                            getattr(
                                factor_surrogate_bundle,
                                "subject_image_probe",
                                None,
                            )
                            is None
                        ):
                            surrogate_logits = (
                                factor_surrogate_bundle.subject_logits(
                                    surrogate_x0
                                )
                            )
                        else:
                            active_z_m = z_m[factor_mask]
                            active_prior_m = F.interpolate(
                                surrogate_x0,
                                size=active_z_m.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )
                            mid_t = torch.randint(
                                factor_surrogate_timestep_min,
                                factor_surrogate_timestep_max + 1,
                                (len(active_z_m),),
                                device=active_z_m.device,
                            )
                            mid_noise = torch.randn_like(active_z_m)
                            mid_x_t = diff.q_sample(
                                active_z_m,
                                mid_t,
                                mid_noise,
                            )
                            mid_condition = (
                                negative_rank_conds[1][0][factor_mask],
                                negative_rank_conds[1][1][factor_mask],
                            )
                            mid_prediction = branch_prediction(
                                mid_x_t,
                                self.unet_mid,
                                mid_t,
                                *mid_condition,
                                active_prior_m,
                            )
                            mid_x0 = diff.model_output_to_x0_eps(
                                mid_x_t,
                                mid_t,
                                mid_prediction,
                                cfg.prediction_type,
                            )[0]

                            active_z_h = z_h[factor_mask]
                            active_prior_h = F.interpolate(
                                mid_x0,
                                size=active_z_h.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )
                            high_t = torch.randint(
                                factor_surrogate_timestep_min,
                                factor_surrogate_timestep_max + 1,
                                (len(active_z_h),),
                                device=active_z_h.device,
                            )
                            high_noise = torch.randn_like(active_z_h)
                            high_x_t = diff.q_sample(
                                active_z_h,
                                high_t,
                                high_noise,
                            )
                            high_condition = (
                                negative_rank_conds[2][0][factor_mask],
                                negative_rank_conds[2][1][factor_mask],
                            )
                            high_prediction = branch_prediction(
                                high_x_t,
                                self.unet_high,
                                high_t,
                                *high_condition,
                                active_prior_h,
                            )
                            high_x0 = diff.model_output_to_x0_eps(
                                high_x_t,
                                high_t,
                                high_prediction,
                                cfg.prediction_type,
                            )[0]
                            decoded = self.vae_high.decode(
                                self._denormalize_latent(
                                    high_x0,
                                    "high",
                                )
                            )
                            surrogate_logits = (
                                factor_surrogate_bundle
                                .subject_image_logits(decoded)
                            )
                    elif surrogate_name == "state_dnn":
                        if (
                            getattr(
                                factor_surrogate_bundle,
                                "dnn_state_image_judge",
                                None,
                            )
                            is None
                        ):
                            surrogate_logits = (
                                factor_surrogate_bundle.dnn_state_logits(
                                    surrogate_x0
                                )
                            )
                        else:
                            active_z_h = z_h[factor_mask]
                            active_prior_h = F.interpolate(
                                surrogate_x0,
                                size=active_z_h.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )
                            high_t = torch.randint(
                                factor_surrogate_timestep_min,
                                factor_surrogate_timestep_max + 1,
                                (len(active_z_h),),
                                device=active_z_h.device,
                            )
                            high_noise = torch.randn_like(active_z_h)
                            high_x_t = diff.q_sample(
                                active_z_h,
                                high_t,
                                high_noise,
                            )
                            high_condition = (
                                negative_rank_conds[2][0][factor_mask],
                                negative_rank_conds[2][1][factor_mask],
                            )
                            high_prediction = branch_prediction(
                                high_x_t,
                                self.unet_high,
                                high_t,
                                *high_condition,
                                active_prior_h,
                            )
                            high_x0 = diff.model_output_to_x0_eps(
                                high_x_t,
                                high_t,
                                high_prediction,
                                cfg.prediction_type,
                            )[0]
                            decoded = self.vae_high.decode(
                                self._denormalize_latent(
                                    high_x0,
                                    "high",
                                )
                            )
                            surrogate_logits = (
                                factor_surrogate_bundle
                                .dnn_state_image_logits(decoded)
                            )
                    else:
                        decoded = self.vae_high.decode(
                            self._denormalize_latent(
                                surrogate_x0,
                                "high",
                            )
                        )
                        surrogate_logits = (
                            factor_surrogate_bundle.background_logits(decoded)
                        )
                    loss_logits = surrogate_logits.float()
                    loss_targets = target_ids
                    if (
                        surrogate_name in {"subject", "state_dnn"}
                        and factor_surrogate_state_population_size > 1
                    ):
                        grouped_logits, grouped_targets = (
                            population_mean_logits_and_targets(
                                loss_logits,
                                loss_targets,
                                factor_surrogate_state_population_size,
                            )
                        )
                        if (
                            surrogate_name == "subject"
                            and bool(
                                factor_surrogate_bundle
                                .has_subject_distribution_bank()
                            )
                        ):
                            factor_loss = (
                                factor_surrogate_bundle
                                .subject_distribution_loss(
                                    loss_logits,
                                    loss_targets,
                                    factor_surrogate_state_population_size,
                                )
                            )
                        elif (
                            surrogate_name == "state_dnn"
                            and bool(
                                factor_surrogate_bundle
                                .has_dnn_state_distribution_bank()
                            )
                        ):
                            factor_loss = (
                                factor_surrogate_bundle
                                .dnn_state_distribution_loss(
                                    loss_logits,
                                    loss_targets,
                                    factor_surrogate_state_population_size,
                                )
                            )
                        else:
                            factor_loss = F.cross_entropy(
                                grouped_logits,
                                grouped_targets,
                            )
                        loss_logits = grouped_logits
                        loss_targets = grouped_targets
                    else:
                        factor_loss = F.cross_entropy(
                            loss_logits,
                            loss_targets,
                        )
                    factor_accuracy = (
                        loss_logits.argmax(dim=1) == loss_targets
                    ).float().mean()
                else:
                    factor_loss = x512.new_zeros(())
                    factor_accuracy = x512.new_zeros(())
                losses["loss_factor_surrogate"] = (
                    factor_surrogate_weight * factor_loss
                )
                losses["metric_factor_surrogate_accuracy"] = (
                    factor_accuracy.detach()
                )

            if morphology_mask_surrogate_weight > 0.0:
                if morphology_mask_surrogate_bundle is None:
                    raise ValueError(
                        "morphology mask weight requires frozen surrogates"
                    )
                if morphology_mask_target is None:
                    raise ValueError(
                        "morphology mask weight requires target masks"
                    )
                if morphology_mask_source_ids is None:
                    raise ValueError(
                        "morphology mask weight requires source IDs"
                    )
                if cfg.conditioning_mode == MORPHOLOGY_FEATURE_AFFINE_MODE:
                    if morphology_class_ids is None:
                        raise ValueError(
                            "feature-affine geometry supervision requires "
                            "factual morphology class IDs"
                        )
                    if morphology_counterfactual_class_ids is None:
                        raise ValueError(
                            "feature-affine geometry supervision requires "
                            "counterfactual morphology class IDs"
                        )
                    if morphology_class_ids.shape != (b,) or (
                        morphology_counterfactual_class_ids.shape != (b,)
                    ):
                        raise ValueError(
                            "factual/counterfactual morphology class IDs must "
                            "have shape [batch]"
                        )
                if morphology_mask_target.shape[:2] != (b, 1):
                    raise ValueError(
                        "morphology mask targets must have shape [batch,1,H,W]"
                    )
                if morphology_mask_source_ids.shape != (b,):
                    raise ValueError(
                        "morphology mask source IDs must have shape [batch]"
                    )
                if bool(factor_mask.any()):
                    # Keep the ordinary diffusion-noise stream identical to
                    # the matched control. Auxiliary geometry sampling uses
                    # the current state but restores it before the next batch.
                    cpu_rng_state = torch.get_rng_state()
                    cuda_rng_state = (
                        torch.cuda.get_rng_state(z_l.device)
                        if z_l.device.type == "cuda"
                        else None
                    )
                    low_t = torch.randint(
                        morphology_mask_surrogate_timestep_min,
                        morphology_mask_surrogate_timestep_max + 1,
                        (b,),
                        device=z_l.device,
                    )
                    low_noise = torch.randn_like(z_l)
                    low_x_t = diff.q_sample(z_l, low_t, low_noise)
                    low_prediction = branch_prediction(
                        low_x_t,
                        self.unet_low,
                        low_t,
                        *negative_rank_conds[0],
                        None,
                        factor_delta_context=(
                            factor_attention_delta_context
                        ),
                        factor_delta_mask=factor_attention_delta_mask,
                        morphology_class_ids_override=(
                            morphology_counterfactual_class_ids
                            if cfg.conditioning_mode
                            == MORPHOLOGY_FEATURE_AFFINE_MODE
                            else None
                        ),
                        morphology_route_mask_override=factor_mask,
                        morphology_spatial_mask_override=(
                            morphology_mask_target
                            if cfg.conditioning_mode
                            == MORPHOLOGY_SPATIAL_AFFINE_MODE
                            else None
                        ),
                    )
                    low_x0 = diff.model_output_to_x0_eps(
                        low_x_t,
                        low_t,
                        low_prediction,
                        cfg.prediction_type,
                    )[0]

                    mid_t = torch.randint(
                        morphology_mask_surrogate_timestep_min,
                        morphology_mask_surrogate_timestep_max + 1,
                        (b,),
                        device=z_m.device,
                    )
                    mid_noise = torch.randn_like(z_m)
                    mid_x_t = diff.q_sample(z_m, mid_t, mid_noise)
                    low_prior = F.interpolate(
                        low_x0,
                        size=z_m.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    mid_prediction = branch_prediction(
                        mid_x_t,
                        self.unet_mid,
                        mid_t,
                        *negative_rank_conds[1],
                        low_prior,
                    )
                    mid_x0 = diff.model_output_to_x0_eps(
                        mid_x_t,
                        mid_t,
                        mid_prediction,
                        cfg.prediction_type,
                    )[0]

                    high_t = torch.randint(
                        morphology_mask_surrogate_timestep_min,
                        morphology_mask_surrogate_timestep_max + 1,
                        (b,),
                        device=z_h.device,
                    )
                    high_noise = torch.randn_like(z_h)
                    high_x_t = diff.q_sample(z_h, high_t, high_noise)
                    mid_prior = F.interpolate(
                        mid_x0,
                        size=z_h.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    high_prediction = branch_prediction(
                        high_x_t,
                        self.unet_high,
                        high_t,
                        *negative_rank_conds[2],
                        mid_prior,
                    )
                    high_x0 = diff.model_output_to_x0_eps(
                        high_x_t,
                        high_t,
                        high_prediction,
                        cfg.prediction_type,
                    )[0]
                    decoded = self.vae_high.decode(
                        self._denormalize_latent(high_x0, "high")
                    )[factor_mask]
                    active_source_ids = morphology_mask_source_ids[
                        factor_mask
                    ]
                    if bool((active_source_ids < 0).any()):
                        raise ValueError(
                            "enabled morphology mask sample has invalid source"
                        )
                    mask_logits = morphology_mask_surrogate_bundle(
                        decoded,
                        active_source_ids,
                    )
                    mask_targets = morphology_mask_target[factor_mask]
                    if mask_targets.shape[-2:] != mask_logits.shape[-2:]:
                        mask_targets = F.interpolate(
                            mask_targets.float(),
                            size=mask_logits.shape[-2:],
                            mode="nearest",
                        )
                    geometry_loss, geometry_parts = mask_surrogate_loss(
                        mask_logits,
                        mask_targets,
                    )
                    with torch.no_grad():
                        predicted_binary = mask_logits.sigmoid() >= 0.5
                        target_binary = mask_targets >= 0.5
                        intersection = (
                            predicted_binary & target_binary
                        ).sum(dim=(1, 2, 3)).float()
                        denominator = (
                            predicted_binary.sum(dim=(1, 2, 3)).float()
                            + target_binary.sum(dim=(1, 2, 3)).float()
                        )
                        geometry_dice = (
                            (2.0 * intersection + 1.0)
                            / (denominator + 1.0)
                        ).mean()
                    if morphology_pair_preservation_weight > 0.0:
                        if foreground_mask is None:
                            raise ValueError(
                                "morphology pair preservation requires factual "
                                "foreground masks"
                            )
                        with torch.no_grad():
                            anchor_low_prediction = branch_prediction(
                                low_x_t,
                                self.unet_low,
                                low_t,
                                *positive_rank_conds[0],
                                None,
                                morphology_class_ids_override=(
                                    morphology_class_ids
                                ),
                                morphology_route_mask_override=factor_mask,
                            )
                            anchor_low_x0 = diff.model_output_to_x0_eps(
                                low_x_t,
                                low_t,
                                anchor_low_prediction,
                                cfg.prediction_type,
                            )[0]
                            anchor_low_prior = F.interpolate(
                                anchor_low_x0,
                                size=z_m.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )
                            anchor_mid_prediction = branch_prediction(
                                mid_x_t,
                                self.unet_mid,
                                mid_t,
                                *positive_rank_conds[1],
                                anchor_low_prior,
                            )
                            anchor_mid_x0 = diff.model_output_to_x0_eps(
                                mid_x_t,
                                mid_t,
                                anchor_mid_prediction,
                                cfg.prediction_type,
                            )[0]
                            anchor_mid_prior = F.interpolate(
                                anchor_mid_x0,
                                size=z_h.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )
                            anchor_high_prediction = branch_prediction(
                                high_x_t,
                                self.unet_high,
                                high_t,
                                *positive_rank_conds[2],
                                anchor_mid_prior,
                            )
                            anchor_high_x0 = diff.model_output_to_x0_eps(
                                high_x_t,
                                high_t,
                                anchor_high_prediction,
                                cfg.prediction_type,
                            )[0]
                            anchor_decoded = self.vae_high.decode(
                                self._denormalize_latent(
                                    anchor_high_x0,
                                    "high",
                                )
                            )[factor_mask]
                        preservation_loss, preservation_parts = (
                            morphology_pair_preservation_loss(
                                anchor_decoded,
                                decoded,
                                foreground_mask[factor_mask],
                                morphology_mask_target[factor_mask],
                            )
                        )
                    else:
                        preservation_loss = x512.new_zeros(())
                        preservation_parts = {
                            "background_l1": x512.new_zeros(()),
                            "foreground_mean_l1": x512.new_zeros(()),
                            "foreground_std_l1": x512.new_zeros(()),
                        }
                    torch.set_rng_state(cpu_rng_state)
                    if cuda_rng_state is not None:
                        torch.cuda.set_rng_state(
                            cuda_rng_state,
                            z_l.device,
                        )
                else:
                    geometry_loss = x512.new_zeros(())
                    geometry_dice = x512.new_zeros(())
                    geometry_parts = {
                        "bce": x512.new_zeros(()),
                        "dice": x512.new_zeros(()),
                    }
                    preservation_loss = x512.new_zeros(())
                    preservation_parts = {
                        "background_l1": x512.new_zeros(()),
                        "foreground_mean_l1": x512.new_zeros(()),
                        "foreground_std_l1": x512.new_zeros(()),
                    }
                losses["loss_morphology_mask_surrogate"] = (
                    morphology_mask_surrogate_weight * geometry_loss
                )
                losses["metric_morphology_mask_dice"] = (
                    geometry_dice.detach()
                )
                losses["metric_morphology_mask_bce"] = geometry_parts[
                    "bce"
                ]
                losses["loss_morphology_pair_preservation"] = (
                    morphology_pair_preservation_weight * preservation_loss
                )
                losses["metric_morphology_pair_background_l1"] = (
                    preservation_parts["background_l1"].detach()
                )
                losses["metric_morphology_pair_foreground_mean_l1"] = (
                    preservation_parts["foreground_mean_l1"].detach()
                )
                losses["metric_morphology_pair_foreground_std_l1"] = (
                    preservation_parts["foreground_std_l1"].detach()
                )

        if self.factor_delta_adapters is not None:
            # Some DNN-state batches contain no enabled intervention. Keep the
            # frozen-backbone loss connected to all edit-only adapters so DDP
            # and backward remain valid without changing the objective.
            losses["loss_factor_delta_graph"] = sum(
                parameter.sum() * 0.0
                for parameter in self.factor_delta_adapters.parameters()
            )
        if cfg.conditioning_mode == FACTOR_ATTENTION_DELTA_ADAPTER_MODE:
            # Only one scale is routed per step. Connect every frozen-backbone
            # attention adapter to the graph so strict DDP remains valid while
            # preserving the objective and the direct path exactly.
            losses["loss_factor_attention_delta_graph"] = sum(
                parameter.sum() * 0.0
                for name, parameter in self.named_parameters()
                if "factor_delta_" in name
            )

        if counterfactual_pair_weight > 0.0:
            counterfactual_inputs = (
                counterfactual_anchor_x512,
                counterfactual_target_x512,
                counterfactual_anchor_subject_ids,
                counterfactual_anchor_state_ids,
                counterfactual_anchor_bg_ids,
                counterfactual_target_subject_ids,
                counterfactual_target_state_ids,
                counterfactual_target_bg_ids,
                counterfactual_teacher_unets,
            )
            if any(value is None for value in counterfactual_inputs):
                raise ValueError(
                    "counterfactual pair loss requires paired images, paired "
                    "prompt ids, and frozen teacher U-Nets"
                )
            assert counterfactual_anchor_x512 is not None
            assert counterfactual_target_x512 is not None
            assert counterfactual_anchor_subject_ids is not None
            assert counterfactual_anchor_state_ids is not None
            assert counterfactual_anchor_bg_ids is not None
            assert counterfactual_target_subject_ids is not None
            assert counterfactual_target_state_ids is not None
            assert counterfactual_target_bg_ids is not None
            assert counterfactual_teacher_unets is not None
            pair_losses = self.counterfactual_pair_losses(
                counterfactual_anchor_x512,
                counterfactual_target_x512,
                counterfactual_anchor_subject_ids,
                counterfactual_anchor_state_ids,
                counterfactual_anchor_bg_ids,
                counterfactual_target_subject_ids,
                counterfactual_target_state_ids,
                counterfactual_target_bg_ids,
                factor=counterfactual_factor,
                teacher_unets=counterfactual_teacher_unets,
                timestep_min=counterfactual_timestep_min,
                timestep_max=counterfactual_timestep_max,
            )
            losses["loss_counterfactual_direction"] = (
                counterfactual_pair_weight * pair_losses["direction"]
            )
            losses["loss_counterfactual_teacher_anchor"] = (
                counterfactual_pair_weight
                * counterfactual_teacher_anchor_mix
                * pair_losses["teacher_anchor"]
            )
            losses["loss_counterfactual_teacher_target_orthogonal"] = (
                counterfactual_pair_weight
                * counterfactual_target_orthogonal_mix
                * pair_losses["teacher_target_orthogonal"]
            )
            losses["metric_counterfactual_target_direction_rms"] = (
                pair_losses["target_direction_rms"]
            )

        if observation_rgb_pair_weight > 0.0:
            observation_inputs = (
                observation_rgb_anchor_x512,
                observation_rgb_target_x512,
                observation_rgb_anchor_subject_ids,
                observation_rgb_anchor_state_ids,
                observation_rgb_anchor_bg_ids,
                observation_rgb_target_subject_ids,
                observation_rgb_target_state_ids,
                observation_rgb_target_bg_ids,
            )
            if any(value is None for value in observation_inputs):
                raise ValueError(
                    "observation RGB pair loss requires paired images and "
                    "paired prompt ids"
                )
            assert observation_rgb_anchor_x512 is not None
            assert observation_rgb_target_x512 is not None
            assert observation_rgb_anchor_subject_ids is not None
            assert observation_rgb_anchor_state_ids is not None
            assert observation_rgb_anchor_bg_ids is not None
            assert observation_rgb_target_subject_ids is not None
            assert observation_rgb_target_state_ids is not None
            assert observation_rgb_target_bg_ids is not None
            observation_losses = self.observation_rgb_pair_losses(
                observation_rgb_anchor_x512,
                observation_rgb_target_x512,
                observation_rgb_anchor_subject_ids,
                observation_rgb_anchor_state_ids,
                observation_rgb_anchor_bg_ids,
                observation_rgb_target_subject_ids,
                observation_rgb_target_state_ids,
                observation_rgb_target_bg_ids,
                timestep_min=observation_rgb_pair_timestep_min,
                timestep_max=observation_rgb_pair_timestep_max,
            )
            losses["loss_observation_rgb_pair_direction"] = (
                observation_rgb_pair_weight
                * observation_losses["direction"]
            )
            losses["loss_observation_rgb_pair_center"] = (
                observation_rgb_pair_weight
                * observation_rgb_pair_center_weight
                * observation_losses["center"]
            )
            for key, value in observation_losses.items():
                if key in {"direction", "center"}:
                    continue
                losses[f"metric_observation_rgb_pair_{key}"] = value

        if cfg.freeze_vae:
            losses["loss_vae"] = x512.new_zeros(())
            losses["loss_kl"] = x512.new_zeros(())
        else:
            assert xr_l is not None and xr_m is not None and xr_h is not None
            losses["loss_vae"] = vae_weight * (
                vae_reconstruction_loss(x128, xr_l)
                + vae_reconstruction_loss(x256, xr_m)
                + vae_reconstruction_loss(x512, xr_h)
            )
            losses["loss_kl"] = cfg.vae_kl_weight * (
                vae_kl_loss(mu_l, lv_l)
                + vae_kl_loss(mu_m, lv_m)
                + vae_kl_loss(mu_h, lv_h)
            )

        losses["loss_contrast"] = weighted_image_text_contrastive_loss(
            x512, contrastive_pools, self.contrastive,
            cfg.contrastive_weight, cfg.contrastive_tau,
        )

        losses["loss_total"] = sum(
            value
            for key, value in losses.items()
            if key.startswith("loss_") and key != "loss_total"
        )
        return losses

    def personalization_losses(
        self,
        x512_concept: torch.Tensor,
        subject_ids_c: torch.Tensor,
        state_ids_c: torch.Tensor,
        bg_ids_c: torch.Tensor,
        x512_prior: torch.Tensor,
        subject_ids_p: torch.Tensor,
        state_ids_p: torch.Tensor,
        bg_ids_p: torch.Tensor,
        use_concept_embedding: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        个体化微调专用（4.3.4 / 式 4-33–4-34）：仅低分辨率分支噪声预测。
        - 概念批：主体条件叠加 personalization，拟合少量目标细胞样本；
        - 先验批：不叠加 personalization，保持通用细胞分布。
        """
        cfg = self.cfg
        diff = self.diffusion

        def low_diffusion_loss(
            x512: torch.Tensor,
            subj: torch.Tensor,
            st: torch.Tensor,
            bg: torch.Tensor,
            use_pers: bool,
        ) -> torch.Tensor:
            x128 = F.interpolate(x512, size=(128, 128), mode="bilinear", align_corners=False)
            mean, logvar = self.vae_low.encode(x128)
            z_l = mean if cfg.freeze_vae and cfg.frozen_vae_use_mean else self.vae_low.reparameterize(mean, logvar)
            z_l = self._normalize_latent(z_l, "low")
            (c_sub_seq, c_sub_pool), _, _ = self.text_enc(subj, st, bg)
            c_sub_seq, c_sub_pool = self._add_personalization(c_sub_seq, c_sub_pool, use_pers)
            b = z_l.shape[0]
            t = torch.randint(0, cfg.num_train_timesteps, (b,), device=z_l.device)
            noise = torch.randn_like(z_l)
            x_t = diff.q_sample(z_l, t, noise)
            pred = self.unet_low(x_t, t, c_sub_seq, c_sub_pool)
            target = diff.training_target(
                z_l,
                t,
                noise,
                cfg.prediction_type,
            )
            return F.mse_loss(pred, target)

        loss_pi = low_diffusion_loss(
            x512_concept,
            subject_ids_c,
            state_ids_c,
            bg_ids_c,
            use_pers=use_concept_embedding,
        )
        loss_prior = low_diffusion_loss(
            x512_prior,
            subject_ids_p,
            state_ids_p,
            bg_ids_p,
            use_pers=False,
        )
        w_pi = cfg.lambda_personalization
        w_pr = cfg.lambda_prior_preservation
        total = w_pi * loss_pi + w_pr * loss_prior
        return {
            "loss_personal": loss_pi,
            "loss_prior_preserv": loss_prior,
            "loss_total": total,
        }

    @torch.no_grad()
    def sample(
        self,
        subject_ids: torch.Tensor,
        state_ids: torch.Tensor,
        bg_ids: torch.Tensor,
        batch_size: int | None = None,
        cfg_scale: float | None = None,
        cfg_scale_subject: float | torch.Tensor | None = None,
        cfg_scale_state: float | torch.Tensor | None = None,
        cfg_scale_background: float | torch.Tensor | None = None,
        num_steps: int | None = None,
        use_personalization: bool = False,
        generator: torch.Generator | None = None,
        subject_state_low_mask: torch.Tensor | None = None,
        spatial_subject_mask: torch.Tensor | None = None,
        spatial_subject_route_mask: torch.Tensor | None = None,
        morphology_class_ids: torch.Tensor | None = None,
        anchor_subject_ids: torch.Tensor | None = None,
        anchor_state_ids: torch.Tensor | None = None,
        anchor_bg_ids: torch.Tensor | None = None,
        intervention_factor: int | torch.Tensor | None = None,
        rgb_observation_adapter_scale: float | torch.Tensor = 1.0,
        return_intermediates: bool = False,
        denoising_trace_callback: Callable[
            [str, int, torch.Tensor, torch.Tensor], None
        ] | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """由粗到细逐级生成 512×512 RGB（算法 4-1 推理阶段）。"""
        cfg = self.cfg
        device = subject_ids.device
        bs = batch_size or subject_ids.shape[0]
        zc = cfg.latent_channels
        cfg_scale = cfg.cfg_scale if cfg_scale is None else cfg_scale
        branch_cfg_scales = {
            "subject": (
                cfg_scale if cfg_scale_subject is None else cfg_scale_subject
            ),
            "state": cfg_scale if cfg_scale_state is None else cfg_scale_state,
            "background": (
                cfg_scale
                if cfg_scale_background is None
                else cfg_scale_background
            ),
        }
        invalid_cfg_scales = {}
        for name, value in branch_cfg_scales.items():
            if isinstance(value, torch.Tensor):
                if value.ndim not in {0, 1}:
                    invalid_cfg_scales[name] = tuple(value.shape)
                elif bool(torch.any(value < 0).item()):
                    invalid_cfg_scales[name] = "contains negative values"
            elif float(value) < 0.0:
                invalid_cfg_scales[name] = value
        if invalid_cfg_scales:
            raise ValueError(
                "branch CFG scales must be non-negative: "
                f"{invalid_cfg_scales}"
            )
        num_steps = cfg.num_inference_steps if num_steps is None else num_steps

        (
            (c_sub_seq, c_sub_pool),
            (c_st_seq, c_st_pool),
            (c_bg_seq, c_bg_pool),
        ) = self.text_enc(subject_ids, state_ids, bg_ids)
        c_sub_seq, c_sub_pool = self._add_personalization(c_sub_seq, c_sub_pool, use_personalization)
        full_raw_conditions = (
            (c_sub_seq, c_sub_pool),
            (c_st_seq, c_st_pool),
            (c_bg_seq, c_bg_pool),
        )

        # 训练时条件丢弃会把单个可学习 null token 广播到完整的文本序列长度。
        # 推理必须保持相同形状；若仅传入长度 1 的序列，cross-attention 的
        # unconditional 分支会退化为单一 K/V，并在较高 CFG 下放大采样噪声。
        ns_seq = self.null_subj_seq.expand(bs, c_sub_seq.shape[1], -1)
        nst_seq = self.null_state_seq.expand(bs, c_st_seq.shape[1], -1)
        nbg_seq = self.null_bg_seq.expand(bs, c_bg_seq.shape[1], -1)
        ns_pool = self.null_subj.expand(bs, -1)
        nst_pool = self.null_state.expand(bs, -1)
        nbg_pool = self.null_bg.expand(bs, -1)
        null_raw_conditions = (
            (ns_seq, ns_pool),
            (nst_seq, nst_pool),
            (nbg_seq, nbg_pool),
        )
        (
            (c_sub_seq, c_sub_pool),
            (c_st_seq, c_st_pool),
            (c_bg_seq, c_bg_pool),
        ) = self._route_conditions(
            full_raw_conditions,
            subject_state_low_mask=subject_state_low_mask,
        )
        (
            (c_sub_seq, c_sub_pool),
            (c_st_seq, c_st_pool),
            (c_bg_seq, c_bg_pool),
        ) = self._apply_nucleus_prompt_adapters(
            (
                (c_sub_seq, c_sub_pool),
                (c_st_seq, c_st_pool),
                (c_bg_seq, c_bg_pool),
            ),
            subject_state_low_mask,
        )
        factor_attention_delta_contexts: tuple[
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
        ] = (None, None, None)
        factor_attention_delta_masks: tuple[
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
        ] = (None, None, None)
        if cfg.conditioning_mode in {
            FACTOR_DELTA_ADAPTER_MODE,
            FACTOR_ATTENTION_DELTA_ADAPTER_MODE,
        }:
            anchor_inputs = (
                anchor_subject_ids,
                anchor_state_ids,
                anchor_bg_ids,
                intervention_factor,
            )
            if any(value is not None for value in anchor_inputs):
                if any(value is None for value in anchor_inputs):
                    raise ValueError(
                        "factor delta edit requires all anchor ids and "
                        "intervention_factor"
                    )
                assert anchor_subject_ids is not None
                assert anchor_state_ids is not None
                assert anchor_bg_ids is not None
                assert intervention_factor is not None
                anchor_conditions = self._route_conditions(
                    self.text_enc(
                        anchor_subject_ids,
                        anchor_state_ids,
                        anchor_bg_ids,
                    ),
                    subject_state_low_mask=subject_state_low_mask,
                )
                target_conditions = (
                    (c_sub_seq, c_sub_pool),
                    (c_st_seq, c_st_pool),
                    (c_bg_seq, c_bg_pool),
                )
                if cfg.conditioning_mode == FACTOR_DELTA_ADAPTER_MODE:
                    (
                        (c_sub_seq, c_sub_pool),
                        (c_st_seq, c_st_pool),
                        (c_bg_seq, c_bg_pool),
                    ) = self._apply_factor_delta_adapters(
                        target_conditions,
                        anchor_conditions,
                        intervention_factor,
                    )
                else:
                    route_masks = self._factor_route_masks(
                        intervention_factor,
                        bs,
                        c_sub_seq.device,
                    )
                    factor_attention_delta_contexts = tuple(
                        target_conditions[index][0]
                        - anchor_conditions[index][0]
                        for index in range(3)
                    )
                    factor_attention_delta_masks = route_masks
        (
            (ns_seq, ns_pool),
            (nst_seq, nst_pool),
            (nbg_seq, nbg_pool),
        ) = self._route_conditions(
            null_raw_conditions,
            subject_state_low_mask=subject_state_low_mask,
        )
        (
            (ns_seq, ns_pool),
            (nst_seq, nst_pool),
            (nbg_seq, nbg_pool),
        ) = self._apply_nucleus_prompt_adapters(
            (
                (ns_seq, ns_pool),
                (nst_seq, nst_pool),
                (nbg_seq, nbg_pool),
            ),
            subject_state_low_mask,
        )

        # semantic_multiscale 在一个尺度内同时接收多个语义因子。独立条件
        # dropout 已让模型见过“只缺一个因子”的组合；推理时据此计算按因子
        # 分开的 CFG，避免 subject CFG 同时放大同尺度的 background。
        factor_ablated_conditions: list[
            tuple[
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
            ]
        ] = []
        if cfg.conditioning_mode in {
            SEMANTIC_MULTISCALE_MODE,
            SEMANTIC_STATE_ALLSCALE_MODE,
            SEMANTIC_STATE_SEQ_LOW_MODE,
            SEMANTIC_STATE_FACTOR_RESIDUAL_POOL_MODE,
            SEMANTIC_STATE_FACTOR_PARTITION_ATTENTION_MODE,
            SEMANTIC_STATE_FACTOR_ISOLATED_COMBINED_MODE,
            SEMANTIC_STATE_SPECIFIC_RESIDUAL_MODE,
        }:
            for factor_index in range(3):
                raw_without_factor = tuple(
                    null_raw_conditions[index]
                    if index == factor_index
                    else full_raw_conditions[index]
                    for index in range(3)
                )
                factor_ablated_conditions.append(
                    self._route_conditions(
                        raw_without_factor,
                        subject_state_low_mask=subject_state_low_mask,
                    )
                )

        diff = self.diffusion

        def sample_branch(
            branch_name: str,
            shape: tuple[int, ...],
            unet: ScaleUNet,
            cond_seq: torch.Tensor,
            cond_pool: torch.Tensor,
            null_seq: torch.Tensor,
            null_pool: torch.Tensor,
            prior: torch.Tensor | None,
            nucleus_adapter: NucleusLowResidualAdapter | None = None,
            branch_cfg_scale: float | torch.Tensor = 1.0,
            factor_delta_context: torch.Tensor | None = None,
            factor_delta_mask: torch.Tensor | None = None,
            factor_guidance: tuple[
                tuple[
                    torch.Tensor,
                    torch.Tensor,
                    float | torch.Tensor,
                ],
                ...,
            ] = (),
        ) -> torch.Tensor:
            def denoise_fn(
                x: torch.Tensor,
                t: torch.Tensor,
                cond_pair: tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor | None,
                    torch.Tensor | None,
                ],
            ) -> torch.Tensor:
                if (
                    unet is self.unet_low
                    and (
                        self.spatial_subject_input_adapter is not None
                        or self.morphology_latent_bank is not None
                    )
                    and (
                        spatial_subject_mask is not None
                        or morphology_class_ids is not None
                    )
                ):
                    structure_batch = (
                        spatial_subject_mask.shape[0]
                        if spatial_subject_mask is not None
                        else morphology_class_ids.shape[0]
                    )
                    if structure_batch != x.shape[0]:
                        raise ValueError(
                            "spatial subject mask batch size does not match sampling batch"
                        )
                    route = spatial_subject_route_mask
                    if route is None:
                        route = torch.ones(
                            x.shape[0],
                            device=x.device,
                            dtype=torch.bool,
                        )
                    if route.ndim != 1 or route.shape[0] != x.shape[0]:
                        raise ValueError(
                            "spatial subject route mask must have shape [batch]"
                        )
                    if self.spatial_subject_input_adapter is not None:
                        assert spatial_subject_mask is not None
                        spatial_residual = self.spatial_subject_input_adapter(
                            spatial_subject_mask.to(device=x.device),
                            x.shape[-2:],
                        )
                    else:
                        assert self.morphology_latent_bank is not None
                        assert morphology_class_ids is not None
                        spatial_residual = self.morphology_latent_bank(
                            morphology_class_ids.to(device=x.device),
                            x.shape[-2:],
                        )
                    spatial_residual = spatial_residual.to(dtype=x.dtype)
                    x = torch.where(
                        route.to(device=x.device, dtype=torch.bool)[
                            :, None, None, None
                        ],
                        x + spatial_residual,
                        x,
                    )
                if prior is not None:
                    p = prior
                    if p.shape[-2:] != x.shape[-2:]:
                        p = F.interpolate(p, size=x.shape[-2:], mode="bilinear", align_corners=False)
                    x_in = torch.cat([x, p], dim=1)
                else:
                    x_in = x
                cseq, cpool, delta_context, delta_mask = cond_pair
                prediction = unet(
                    x_in,
                    t,
                    cseq,
                    cpool,
                    factor_delta_context=delta_context,
                    factor_delta_mask=delta_mask,
                    morphology_class_ids=(
                        morphology_class_ids
                        if unet is self.unet_low
                        and cfg.conditioning_mode
                        == MORPHOLOGY_FEATURE_AFFINE_MODE
                        else None
                    ),
                    morphology_route_mask=(
                        spatial_subject_route_mask
                        if unet is self.unet_low
                        and cfg.conditioning_mode
                        in {
                            MORPHOLOGY_FEATURE_AFFINE_MODE,
                            MORPHOLOGY_SPATIAL_AFFINE_MODE,
                        }
                        else None
                    ),
                    morphology_spatial_mask=(
                        spatial_subject_mask
                        if unet is self.unet_low
                        and cfg.conditioning_mode
                        == MORPHOLOGY_SPATIAL_AFFINE_MODE
                        else None
                    ),
                )
                if self.rgb_factor_output_adapters is not None:
                    prediction = self._apply_rgb_factor_output_adapters(
                        prediction,
                        t,
                        c_st_pool,
                        c_bg_pool,
                        branch_name,
                        observation_scale=rgb_observation_adapter_scale,
                    )
                if nucleus_adapter is not None:
                    if subject_state_low_mask is None:
                        raise ValueError(
                            "nucleus branch adapter requires subject_state_low_mask"
                        )
                    residual = nucleus_adapter(prediction, t, cpool)
                    adapter_mask = nucleus_adapter_route_mask(
                        cfg.conditioning_mode,
                        subject_state_low_mask,
                        t,
                        cfg.nucleus_temporal_adapter_max_timestep,
                    )
                    routed_prediction = prediction + residual
                    prediction = torch.where(
                        adapter_mask.to(device=prediction.device)[
                            :, None, None, None
                        ],
                        routed_prediction,
                        prediction,
                    )
                if factor_guidance:
                    full_prediction = prediction
                    without_factor_predictions: list[
                        tuple[torch.Tensor, float | torch.Tensor]
                    ] = []
                    for without_seq, without_pool, factor_scale in factor_guidance:
                        if cfg_scale_is_one(factor_scale):
                            continue
                        without_prediction = unet(
                            x_in,
                            t,
                            without_seq,
                            without_pool,
                            morphology_class_ids=(
                                morphology_class_ids
                                if unet is self.unet_low
                                and cfg.conditioning_mode
                                == MORPHOLOGY_FEATURE_AFFINE_MODE
                                else None
                            ),
                            morphology_route_mask=(
                                spatial_subject_route_mask
                                if unet is self.unet_low
                                and cfg.conditioning_mode
                                in {
                                    MORPHOLOGY_FEATURE_AFFINE_MODE,
                                    MORPHOLOGY_SPATIAL_AFFINE_MODE,
                                }
                                else None
                            ),
                            morphology_spatial_mask=(
                                spatial_subject_mask
                                if unet is self.unet_low
                                and cfg.conditioning_mode
                                == MORPHOLOGY_SPATIAL_AFFINE_MODE
                                else None
                            ),
                        )
                        without_factor_predictions.append(
                            (without_prediction, factor_scale)
                        )
                    prediction = combine_factor_cfg_predictions(
                        full_prediction,
                        tuple(without_factor_predictions),
                    )
                return prediction

            return diff.ddim_sample(
                denoise_fn,
                shape,
                (
                    cond_seq,
                    cond_pool,
                    factor_delta_context,
                    factor_delta_mask,
                ),
                1.0 if factor_guidance else branch_cfg_scale,
                num_steps,
                None
                if factor_guidance
                else (null_seq, null_pool, None, None),
                generator=generator,
                prediction_type=cfg.prediction_type,
                trace_callback=(
                    (
                        lambda step_index, timestep, predicted_x0: (
                            denoising_trace_callback(
                                branch_name,
                                step_index,
                                timestep,
                                predicted_x0,
                            )
                        )
                    )
                    if denoising_trace_callback is not None
                    else None
                ),
            )

        semantic_factor_guidance: dict[
            str,
            tuple[
                tuple[
                    torch.Tensor,
                    torch.Tensor,
                    float | torch.Tensor,
                ],
                ...,
            ],
        ] = {}
        if factor_ablated_conditions:
            low_guidance = [
                (
                    *factor_ablated_conditions[0][0],
                    branch_cfg_scales["subject"],
                )
            ]
            if cfg.conditioning_mode in {
                SEMANTIC_STATE_ALLSCALE_MODE,
                SEMANTIC_STATE_SEQ_LOW_MODE,
                SEMANTIC_STATE_FACTOR_RESIDUAL_POOL_MODE,
                SEMANTIC_STATE_FACTOR_PARTITION_ATTENTION_MODE,
                SEMANTIC_STATE_FACTOR_ISOLATED_COMBINED_MODE,
                SEMANTIC_STATE_SPECIFIC_RESIDUAL_MODE,
            }:
                low_guidance.append(
                    (
                        *factor_ablated_conditions[1][0],
                        branch_cfg_scales["state"],
                    )
                )
            low_guidance.append(
                (
                    *factor_ablated_conditions[2][0],
                    branch_cfg_scales["background"],
                )
            )
            semantic_factor_guidance = {
                "low": tuple(low_guidance),
                "mid": (
                    (
                        *factor_ablated_conditions[1][1],
                        branch_cfg_scales["state"],
                    ),
                    (
                        *factor_ablated_conditions[2][1],
                        branch_cfg_scales["background"],
                    ),
                ),
                "high": (
                    (
                        *factor_ablated_conditions[1][2],
                        branch_cfg_scales["state"],
                    ),
                    (
                        *factor_ablated_conditions[2][2],
                        branch_cfg_scales["background"],
                    ),
                ),
            }

        z_l = sample_branch(
            "low",
            (bs, zc, 16, 16),
            self.unet_low,
            c_sub_seq,
            c_sub_pool,
            ns_seq,
            ns_pool,
            None,
            nucleus_adapter=self.nucleus_low_adapter,
            branch_cfg_scale=branch_cfg_scales["subject"],
            factor_delta_context=factor_attention_delta_contexts[0],
            factor_delta_mask=factor_attention_delta_masks[0],
            factor_guidance=semantic_factor_guidance.get("low", ()),
        )
        prior_m = F.interpolate(z_l, size=(32, 32), mode="bilinear", align_corners=False)
        z_m = sample_branch(
            "mid",
            (bs, zc, 32, 32),
            self.unet_mid,
            c_st_seq,
            c_st_pool,
            nst_seq,
            nst_pool,
            prior_m,
            nucleus_adapter=self.nucleus_mid_adapter,
            branch_cfg_scale=branch_cfg_scales["state"],
            factor_delta_context=factor_attention_delta_contexts[1],
            factor_delta_mask=factor_attention_delta_masks[1],
            factor_guidance=semantic_factor_guidance.get("mid", ()),
        )
        prior_h = F.interpolate(z_m, size=(64, 64), mode="bilinear", align_corners=False)
        z_h = sample_branch(
            "high",
            (bs, zc, 64, 64),
            self.unet_high,
            c_bg_seq,
            c_bg_pool,
            nbg_seq,
            nbg_pool,
            prior_h,
            nucleus_adapter=self.nucleus_high_adapter,
            branch_cfg_scale=branch_cfg_scales["background"],
            factor_delta_context=factor_attention_delta_contexts[2],
            factor_delta_mask=factor_attention_delta_masks[2],
            factor_guidance=semantic_factor_guidance.get("high", ()),
        )

        rgb = self.vae_high.decode(self._denormalize_latent(z_h, "high"))
        rgb = rgb.clamp(-1, 1)
        if return_intermediates:
            return {
                "rgb": rgb,
                "low": z_l,
                "mid": z_m,
                "high": z_h,
            }
        return rgb

    def forward(
        self,
        *args,
        mode: str = "train",
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """DDP 兼容入口，根据 mode 分发到对应损失函数。"""
        if mode == "personalize":
            return self.personalization_losses(*args, **kwargs)
        return self.training_losses(*args, **kwargs)

    def get_optimizer_param_groups(self, lr: float, wd: float) -> list[dict[str, Any]]:
        """个体化微调时可只对 personalization 与 text_enc / unet_low 设不同 lr。"""
        return [
            {
                "params": [p for p in self.parameters() if p.requires_grad],
                "lr": lr,
                "weight_decay": wd,
            }
        ]
