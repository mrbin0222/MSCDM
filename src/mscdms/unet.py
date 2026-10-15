from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .reversible import ReversibleCondBlock


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=timesteps.device) / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class CrossAttention2d(nn.Module):
    """噪声特征为 Q，文本 token 序列为 K/V。"""

    def __init__(
        self,
        ch: int,
        context_dim: int,
        num_heads: int = 8,
        factor_delta_rank: int | None = None,
        factor_partition_count: int | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, ch)
        self.to_q = nn.Conv2d(ch, ch, 1)
        self.to_k = nn.Linear(context_dim, ch)
        self.to_v = nn.Linear(context_dim, ch)
        self.proj = nn.Conv2d(ch, ch, 1)
        if factor_partition_count is not None and factor_partition_count < 2:
            raise ValueError("factor_partition_count must be at least two")
        self.factor_partition_count = factor_partition_count
        # This scalar zero gate makes the candidate exactly equal to the
        # original joint-token attention at initialization. The partitioned
        # path reuses the existing Q/K/V projections and adds no expert weights.
        self.factor_partition_gate = (
            nn.Parameter(torch.zeros(()))
            if factor_partition_count is not None
            else None
        )
        if factor_delta_rank is not None and factor_delta_rank <= 0:
            raise ValueError("factor_delta_rank must be positive when enabled")
        self.factor_delta_down = (
            nn.Linear(context_dim, factor_delta_rank, bias=False)
            if factor_delta_rank is not None
            else None
        )
        self.factor_delta_to_k = (
            nn.Linear(factor_delta_rank, ch, bias=False)
            if factor_delta_rank is not None
            else None
        )
        self.factor_delta_to_v = (
            nn.Linear(factor_delta_rank, ch, bias=False)
            if factor_delta_rank is not None
            else None
        )
        if self.factor_delta_to_k is not None:
            nn.init.zeros_(self.factor_delta_to_k.weight)
            nn.init.zeros_(self.factor_delta_to_v.weight)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        factor_delta_context: torch.Tensor | None = None,
        factor_delta_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, _, h, w = x.shape
        hd = x.shape[1] // self.num_heads
        xn = self.norm(x)
        q = self.to_q(xn).view(b, self.num_heads, hd, h * w).transpose(2, 3)  # (B,H,HW,Dh)
        k_flat = self.to_k(context)
        v_flat = self.to_v(context)
        if factor_delta_context is not None:
            if self.factor_delta_down is None:
                raise ValueError(
                    "factor delta context was supplied to attention without "
                    "an initialized adapter"
                )
            if factor_delta_context.shape != context.shape:
                raise ValueError(
                    "factor delta context must have the same shape as context"
                )
            if factor_delta_mask is None:
                raise ValueError(
                    "factor delta context requires factor_delta_mask"
                )
            if factor_delta_mask.shape != (b,):
                raise ValueError(
                    "factor_delta_mask must have shape [batch]"
                )
            delta = factor_delta_context * factor_delta_mask.to(
                device=context.device,
                dtype=context.dtype,
            )[:, None, None]
            hidden = F.silu(self.factor_delta_down(delta))
            k_flat = k_flat + self.factor_delta_to_k(hidden)
            v_flat = v_flat + self.factor_delta_to_v(hidden)
        k = k_flat.view(
            b, context.shape[1], self.num_heads, hd
        ).transpose(1, 2)  # (B,H,L,Dh)
        v = v_flat.view(
            b, context.shape[1], self.num_heads, hd
        ).transpose(1, 2)  # (B,H,L,Dh)
        scale = 1.0 / math.sqrt(float(hd))
        logits = (q @ k.transpose(-2, -1)) * scale
        attn = torch.softmax(logits, dim=-1)  # (B,H,HW,L)
        out_heads = attn @ v
        if self.factor_partition_count is not None:
            if context.shape[1] < self.factor_partition_count:
                raise ValueError(
                    "factor-partitioned attention received fewer tokens than factors"
                )
            logits_by_factor = torch.tensor_split(
                logits, self.factor_partition_count, dim=-1
            )
            values_by_factor = torch.tensor_split(
                v, self.factor_partition_count, dim=-2
            )
            if any(chunk.shape[-1] == 0 for chunk in logits_by_factor):
                raise ValueError("factor-partitioned attention produced an empty segment")
            partitioned_heads = torch.stack(
                [
                    torch.softmax(factor_logits, dim=-1) @ factor_values
                    for factor_logits, factor_values in zip(
                        logits_by_factor, values_by_factor, strict=True
                    )
                ],
                dim=0,
            ).mean(dim=0)
            out_heads = out_heads + self.factor_partition_gate.to(
                dtype=out_heads.dtype
            ) * (partitioned_heads - out_heads)
        out = out_heads.transpose(2, 3).reshape(b, -1, h, w)
        return x + self.proj(out)


class SelfAttention2d(nn.Module):
    def __init__(self, ch: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, ch)
        self.to_qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        hd = c // self.num_heads
        qkv = self.to_qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(b, self.num_heads, hd, h * w).transpose(2, 3)
        k = k.view(b, self.num_heads, hd, h * w).transpose(2, 3)
        v = v.view(b, self.num_heads, hd, h * w).transpose(2, 3)
        scale = 1.0 / math.sqrt(float(hd))
        attn = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        out = (attn @ v).transpose(2, 3).reshape(b, c, h, w)
        return x + self.proj(out)


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, g = x.chunk(2, dim=-1)
        return x * F.gelu(g)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        hidden = dim * mult
        self.proj_in = nn.Linear(dim, hidden * 2)
        self.geglu = GEGLU()
        self.proj_out = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_out(self.geglu(self.proj_in(x)))


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        ch: int,
        context_dim: int,
        num_heads: int,
        factor_delta_rank: int | None = None,
        factor_partition_count: int | None = None,
    ):
        super().__init__()
        self.self_attn = SelfAttention2d(ch, num_heads=num_heads)
        self.cross_attn = CrossAttention2d(
            ch,
            context_dim=context_dim,
            num_heads=num_heads,
            factor_delta_rank=factor_delta_rank,
            factor_partition_count=factor_partition_count,
        )
        self.norm = nn.LayerNorm(ch)
        self.ff = FeedForward(ch)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        factor_delta_context: torch.Tensor | None = None,
        factor_delta_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.self_attn(x)
        x = self.cross_attn(
            x,
            context,
            factor_delta_context=factor_delta_context,
            factor_delta_mask=factor_delta_mask,
        )
        b, c, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2)
        seq = seq + self.ff(self.norm(seq))
        return seq.transpose(1, 2).reshape(b, c, h, w)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, tdim: int):
        super().__init__()
        self.in1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.in2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_emb = nn.Linear(tdim, out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.in1(x)))
        h = h + self.t_emb(t)[:, :, None, None]
        h = self.conv2(F.silu(self.in2(h)))
        return self.skip(x) + h


class MorphologyFeatureAffine(nn.Module):
    """Zero-initialized class-specific affine modulation of U-Net features."""

    def __init__(self, channels: int, num_classes: int = 3):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(num_classes, channels))
        self.beta = nn.Parameter(torch.zeros(num_classes, channels))

    def forward(
        self,
        features: torch.Tensor,
        class_ids: torch.Tensor,
        route_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if class_ids.ndim != 1 or class_ids.shape[0] != features.shape[0]:
            raise ValueError("morphology class IDs must have shape [batch]")
        if not bool(((class_ids >= 0) & (class_ids < 3)).all()):
            raise ValueError("morphology class IDs must be in [0, 2]")
        gamma = self.gamma[class_ids.long()].to(dtype=features.dtype)
        beta = self.beta[class_ids.long()].to(dtype=features.dtype)
        if route_mask is not None:
            if route_mask.ndim != 1 or route_mask.shape[0] != features.shape[0]:
                raise ValueError("morphology route mask must have shape [batch]")
            route = route_mask.to(
                device=features.device,
                dtype=features.dtype,
            )[:, None]
            gamma = gamma * route
            beta = beta * route
        return (
            features * (1.0 + gamma[:, :, None, None])
            + beta[:, :, None, None]
        )


class StateDeltaFeatureAffine(nn.Module):
    """Edit-only class-specific feature FiLM for state appearance changes."""

    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("state delta feature affine needs at least two classes")
        self.num_classes = int(num_classes)
        self.gamma = nn.Parameter(torch.zeros(num_classes, channels))
        self.beta = nn.Parameter(torch.zeros(num_classes, channels))

    def forward(
        self,
        features: torch.Tensor,
        target_class_ids: torch.Tensor,
        anchor_class_ids: torch.Tensor,
        route_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected = (features.shape[0],)
        if target_class_ids.shape != expected or anchor_class_ids.shape != expected:
            raise ValueError("state class IDs must have shape [batch]")
        valid = (
            (target_class_ids >= 0)
            & (target_class_ids < self.num_classes)
            & (anchor_class_ids >= 0)
            & (anchor_class_ids < self.num_classes)
        )
        if not bool(valid.all()):
            raise ValueError("state class IDs are outside the configured vocabulary")
        target = target_class_ids.long()
        anchor = anchor_class_ids.long()
        gamma = (self.gamma[target] - self.gamma[anchor]).to(
            device=features.device,
            dtype=features.dtype,
        )
        beta = (self.beta[target] - self.beta[anchor]).to(
            device=features.device,
            dtype=features.dtype,
        )
        if route_mask is not None:
            if route_mask.shape != expected:
                raise ValueError("state delta route mask must have shape [batch]")
            route = route_mask.to(
                device=features.device,
                dtype=features.dtype,
            )[:, None]
            gamma = gamma * route
            beta = beta * route
        return features * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]


class TargetStateResponseAffine(nn.Module):
    """Target-only sparse FiLM that never reads or modifies the anchor class.

    This is intentionally separate from ``StateDeltaFeatureAffine``.  Changing
    a row in the latter also changes every edit for which that class is the
    anchor.  A target-only response can repair one missing state appearance
    while leaving all other target routes, including the frozen RPE route,
    bitwise unchanged.
    """

    def __init__(self, channels: int, target_class: int) -> None:
        super().__init__()
        if target_class < 0:
            raise ValueError("target response class must be non-negative")
        self.target_class = int(target_class)
        self.gamma = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(
        self,
        features: torch.Tensor,
        target_class_ids: torch.Tensor,
        route_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target_class_ids.shape != (features.shape[0],):
            raise ValueError("target state class IDs must have shape [batch]")
        active = target_class_ids.eq(self.target_class)
        if route_mask is not None:
            if route_mask.shape != active.shape:
                raise ValueError("target response route mask must have shape [batch]")
            active = active & route_mask.bool()
        route = active.to(device=features.device, dtype=features.dtype)[:, None]
        gamma = self.gamma.to(dtype=features.dtype)[None] * route
        beta = self.beta.to(dtype=features.dtype)[None] * route
        return features * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]


class MorphologySpatialAffine(nn.Module):
    """Zero-initialized spatial FiLM decoded from a target morphology mask.

    Unlike the three-value marginal decoder, this module receives the actual
    target geometry at every U-Net resolution.  The final projection is zero
    initialized, so adding the module preserves the frozen checkpoint exactly.
    """

    def __init__(self, channels: int, hidden_channels: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, channels * 2, 1),
        )
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        target_mask: torch.Tensor,
        route_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target_mask.ndim != 4 or target_mask.shape[:2] != (
            features.shape[0],
            1,
        ):
            raise ValueError(
                "morphology spatial mask must have shape [batch,1,H,W]"
            )
        mask = F.interpolate(
            target_mask.to(device=features.device, dtype=features.dtype),
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        gamma, beta = self.encoder(mask).chunk(2, dim=1)
        if route_mask is not None:
            if route_mask.ndim != 1 or route_mask.shape[0] != features.shape[0]:
                raise ValueError("morphology route mask must have shape [batch]")
            route = route_mask.to(
                device=features.device,
                dtype=features.dtype,
            )[:, None, None, None]
            gamma = gamma * route
            beta = beta * route
        return features * (1.0 + gamma) + beta


class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.op = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.op = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.op(x)


class ScaleUNet(nn.Module):
    """
    单尺度噪声预测 ε_θ。in_channels = 加噪潜变量 (+ 可选 prior)；
    out_channels = 与潜变量同维的噪声目标（prior 仅作条件输入，不预测其噪声）。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_ch: int,
        cond_dim: int,
        tdim: int = 256,
        channel_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (32, 16, 8),
        num_heads: int = 8,
        context_dim: int | None = None,
        input_resolution: int = 64,
        use_reversible: bool = False,
        reversible_save_memory: bool = True,
        factor_delta_attention_rank: int | None = None,
        factor_partition_count: int | None = None,
        morphology_feature_affine: bool = False,
        morphology_spatial_affine: bool = False,
        state_delta_feature_classes: int | None = None,
        target_state_response_class: int | None = None,
        pooled_condition_to_time: bool = False,
    ):
        super().__init__()
        self.tdim = tdim
        self.num_res_blocks = num_res_blocks
        self.time_mlp = nn.Sequential(
            nn.Linear(tdim, tdim * 4),
            nn.SiLU(),
            nn.Linear(tdim * 4, tdim),
        )
        self.pooled_condition_to_time = bool(pooled_condition_to_time)
        self.condition_time_projection = (
            nn.Sequential(
                nn.Linear(cond_dim, tdim),
                nn.SiLU(),
                nn.Linear(tdim, tdim),
            )
            if self.pooled_condition_to_time
            else None
        )
        self.use_reversible = use_reversible
        context_dim = cond_dim if context_dim is None else context_dim
        chs = [base_ch * m for m in channel_mult]
        self.in_conv = nn.Conv2d(in_channels, chs[0], 3, padding=1)

        save_mem = reversible_save_memory

        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.down_revs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.down_morphology_affines = (
            nn.ModuleList() if morphology_feature_affine else None
        )
        self.down_morphology_spatial_affines = (
            nn.ModuleList() if morphology_spatial_affine else None
        )
        self.down_state_delta_affines = (
            nn.ModuleList() if state_delta_feature_classes is not None else None
        )
        self.down_target_state_response_affines = (
            nn.ModuleList() if target_state_response_class is not None else None
        )

        cur_ch = chs[0]
        cur_res = input_resolution
        for i, ch in enumerate(chs):
            lvl_blocks = nn.ModuleList()
            lvl_attns = nn.ModuleList()
            lvl_revs = nn.ModuleList()
            lvl_morphology_affines = nn.ModuleList()
            lvl_morphology_spatial_affines = nn.ModuleList()
            lvl_state_delta_affines = nn.ModuleList()
            lvl_target_state_response_affines = nn.ModuleList()
            for _ in range(num_res_blocks):
                lvl_blocks.append(ResBlock(cur_ch, ch, tdim))
                use_attn = cur_res in set(attention_resolutions)
                lvl_attns.append(
                    BasicTransformerBlock(
                        ch,
                        context_dim=context_dim,
                        num_heads=num_heads,
                        factor_delta_rank=factor_delta_attention_rank,
                        factor_partition_count=factor_partition_count,
                    )
                    if use_attn
                    else nn.Identity()
                )
                if use_reversible and ch % 2 == 0:
                    lvl_revs.append(ReversibleCondBlock(ch, cond_dim, save_memory=save_mem))
                else:
                    lvl_revs.append(nn.Identity())
                if morphology_feature_affine:
                    lvl_morphology_affines.append(MorphologyFeatureAffine(ch))
                if morphology_spatial_affine:
                    lvl_morphology_spatial_affines.append(
                        MorphologySpatialAffine(ch)
                    )
                if state_delta_feature_classes is not None:
                    lvl_state_delta_affines.append(
                        StateDeltaFeatureAffine(ch, state_delta_feature_classes)
                    )
                if target_state_response_class is not None:
                    lvl_target_state_response_affines.append(
                        TargetStateResponseAffine(ch, target_state_response_class)
                    )
                cur_ch = ch
            self.down_blocks.append(lvl_blocks)
            self.down_attns.append(lvl_attns)
            self.down_revs.append(lvl_revs)
            if self.down_morphology_affines is not None:
                self.down_morphology_affines.append(lvl_morphology_affines)
            if self.down_morphology_spatial_affines is not None:
                self.down_morphology_spatial_affines.append(
                    lvl_morphology_spatial_affines
                )
            if self.down_state_delta_affines is not None:
                self.down_state_delta_affines.append(lvl_state_delta_affines)
            if self.down_target_state_response_affines is not None:
                self.down_target_state_response_affines.append(
                    lvl_target_state_response_affines
                )
            if i < len(chs) - 1:
                self.downsamples.append(Downsample(cur_ch, chs[i + 1]))
                cur_ch = chs[i + 1]
                cur_res //= 2

        self.mid1 = ResBlock(cur_ch, cur_ch, tdim)
        self.mid_attn = BasicTransformerBlock(
            cur_ch,
            context_dim=context_dim,
            num_heads=num_heads,
            factor_delta_rank=factor_delta_attention_rank,
            factor_partition_count=factor_partition_count,
        )
        if use_reversible and cur_ch % 2 == 0:
            self.mid_rev = ReversibleCondBlock(cur_ch, cond_dim, save_memory=save_mem)
        else:
            self.mid_rev = nn.Identity()
        self.mid2 = ResBlock(cur_ch, cur_ch, tdim)
        self.mid_morphology_affines = (
            nn.ModuleList(
                [
                    MorphologyFeatureAffine(cur_ch),
                    MorphologyFeatureAffine(cur_ch),
                ]
            )
            if morphology_feature_affine
            else None
        )
        self.mid_morphology_spatial_affines = (
            nn.ModuleList(
                [
                    MorphologySpatialAffine(cur_ch),
                    MorphologySpatialAffine(cur_ch),
                ]
            )
            if morphology_spatial_affine
            else None
        )
        self.mid_state_delta_affines = (
            nn.ModuleList(
                [
                    StateDeltaFeatureAffine(cur_ch, state_delta_feature_classes),
                    StateDeltaFeatureAffine(cur_ch, state_delta_feature_classes),
                ]
            )
            if state_delta_feature_classes is not None
            else None
        )
        self.mid_target_state_response_affines = (
            nn.ModuleList(
                [
                    TargetStateResponseAffine(cur_ch, target_state_response_class),
                    TargetStateResponseAffine(cur_ch, target_state_response_class),
                ]
            )
            if target_state_response_class is not None
            else None
        )

        self.up_merges = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.up_revs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.up_morphology_affines = (
            nn.ModuleList() if morphology_feature_affine else None
        )
        self.up_morphology_spatial_affines = (
            nn.ModuleList() if morphology_spatial_affine else None
        )
        self.up_state_delta_affines = (
            nn.ModuleList() if state_delta_feature_classes is not None else None
        )
        self.up_target_state_response_affines = (
            nn.ModuleList() if target_state_response_class is not None else None
        )

        for i in reversed(range(len(chs))):
            ch = chs[i]
            lvl_merges = nn.ModuleList()
            lvl_blocks = nn.ModuleList()
            lvl_attns = nn.ModuleList()
            lvl_revs = nn.ModuleList()
            lvl_morphology_affines = nn.ModuleList()
            lvl_morphology_spatial_affines = nn.ModuleList()
            lvl_state_delta_affines = nn.ModuleList()
            lvl_target_state_response_affines = nn.ModuleList()
            for _ in range(num_res_blocks):
                lvl_merges.append(nn.Conv2d(cur_ch + ch, ch, 1))
                lvl_blocks.append(ResBlock(ch, ch, tdim))
                use_attn = cur_res in set(attention_resolutions)
                lvl_attns.append(
                    BasicTransformerBlock(
                        ch,
                        context_dim=context_dim,
                        num_heads=num_heads,
                        factor_delta_rank=factor_delta_attention_rank,
                        factor_partition_count=factor_partition_count,
                    )
                    if use_attn
                    else nn.Identity()
                )
                if use_reversible and ch % 2 == 0:
                    lvl_revs.append(ReversibleCondBlock(ch, cond_dim, save_memory=save_mem))
                else:
                    lvl_revs.append(nn.Identity())
                if morphology_feature_affine:
                    lvl_morphology_affines.append(MorphologyFeatureAffine(ch))
                if morphology_spatial_affine:
                    lvl_morphology_spatial_affines.append(
                        MorphologySpatialAffine(ch)
                    )
                if state_delta_feature_classes is not None:
                    lvl_state_delta_affines.append(
                        StateDeltaFeatureAffine(ch, state_delta_feature_classes)
                    )
                if target_state_response_class is not None:
                    lvl_target_state_response_affines.append(
                        TargetStateResponseAffine(ch, target_state_response_class)
                    )
                cur_ch = ch
            self.up_merges.append(lvl_merges)
            self.up_blocks.append(lvl_blocks)
            self.up_attns.append(lvl_attns)
            self.up_revs.append(lvl_revs)
            if self.up_morphology_affines is not None:
                self.up_morphology_affines.append(lvl_morphology_affines)
            if self.up_morphology_spatial_affines is not None:
                self.up_morphology_spatial_affines.append(
                    lvl_morphology_spatial_affines
                )
            if self.up_state_delta_affines is not None:
                self.up_state_delta_affines.append(lvl_state_delta_affines)
            if self.up_target_state_response_affines is not None:
                self.up_target_state_response_affines.append(
                    lvl_target_state_response_affines
                )
            if i > 0:
                self.upsamples.append(Upsample(cur_ch, chs[i - 1]))
                cur_ch = chs[i - 1]
                cur_res *= 2

        self.out_norm = nn.GroupNorm(8, chs[0])
        self.out_conv = nn.Conv2d(chs[0], out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_seq: torch.Tensor,
        cond_pooled: torch.Tensor,
        factor_delta_context: torch.Tensor | None = None,
        factor_delta_mask: torch.Tensor | None = None,
        morphology_class_ids: torch.Tensor | None = None,
        morphology_route_mask: torch.Tensor | None = None,
        morphology_spatial_mask: torch.Tensor | None = None,
        state_target_class_ids: torch.Tensor | None = None,
        state_anchor_class_ids: torch.Tensor | None = None,
        state_delta_route_mask: torch.Tensor | None = None,
        target_state_response_route_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_emb = self.time_mlp(timestep_embedding(t, self.tdim))
        if self.condition_time_projection is not None:
            if cond_pooled.ndim != 2 or cond_pooled.shape != (
                x.shape[0],
                self.condition_time_projection[0].in_features,
            ):
                raise ValueError(
                    "pooled condition has an invalid shape for time injection"
                )
            t_emb = t_emb + self.condition_time_projection(cond_pooled)
        h = self.in_conv(x)
        skips: list[torch.Tensor] = []

        for i in range(len(self.down_blocks)):
            for j in range(len(self.down_blocks[i])):
                h = self.down_blocks[i][j](h, t_emb)
                if (
                    self.down_morphology_affines is not None
                    and morphology_class_ids is not None
                ):
                    h = self.down_morphology_affines[i][j](
                        h,
                        morphology_class_ids,
                        morphology_route_mask,
                    )
                if (
                    self.down_morphology_spatial_affines is not None
                    and morphology_spatial_mask is not None
                ):
                    h = self.down_morphology_spatial_affines[i][j](
                        h,
                        morphology_spatial_mask,
                        morphology_route_mask,
                    )
                if (
                    self.down_state_delta_affines is not None
                    and state_target_class_ids is not None
                    and state_anchor_class_ids is not None
                ):
                    h = self.down_state_delta_affines[i][j](
                        h,
                        state_target_class_ids,
                        state_anchor_class_ids,
                        state_delta_route_mask,
                    )
                if (
                    self.down_target_state_response_affines is not None
                    and state_target_class_ids is not None
                ):
                    h = self.down_target_state_response_affines[i][j](
                        h,
                        state_target_class_ids,
                        target_state_response_route_mask,
                    )
                a = self.down_attns[i][j]
                h = (
                    a(
                        h,
                        cond_seq,
                        factor_delta_context=factor_delta_context,
                        factor_delta_mask=factor_delta_mask,
                    )
                    if isinstance(a, BasicTransformerBlock)
                    else a(h)
                )
                r = self.down_revs[i][j]
                h = r(h, cond_pooled) if isinstance(r, ReversibleCondBlock) else r(h)
                skips.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        h = self.mid1(h, t_emb)
        if (
            self.mid_morphology_affines is not None
            and morphology_class_ids is not None
        ):
            h = self.mid_morphology_affines[0](
                h,
                morphology_class_ids,
                morphology_route_mask,
            )
        if (
            self.mid_morphology_spatial_affines is not None
            and morphology_spatial_mask is not None
        ):
            h = self.mid_morphology_spatial_affines[0](
                h,
                morphology_spatial_mask,
                morphology_route_mask,
            )
        if (
            self.mid_state_delta_affines is not None
            and state_target_class_ids is not None
            and state_anchor_class_ids is not None
        ):
            h = self.mid_state_delta_affines[0](
                h,
                state_target_class_ids,
                state_anchor_class_ids,
                state_delta_route_mask,
            )
        if (
            self.mid_target_state_response_affines is not None
            and state_target_class_ids is not None
        ):
            h = self.mid_target_state_response_affines[0](
                h,
                state_target_class_ids,
                target_state_response_route_mask,
            )
        h = self.mid_attn(
            h,
            cond_seq,
            factor_delta_context=factor_delta_context,
            factor_delta_mask=factor_delta_mask,
        )
        r = self.mid_rev
        h = r(h, cond_pooled) if isinstance(r, ReversibleCondBlock) else r(h)
        h = self.mid2(h, t_emb)
        if (
            self.mid_morphology_affines is not None
            and morphology_class_ids is not None
        ):
            h = self.mid_morphology_affines[1](
                h,
                morphology_class_ids,
                morphology_route_mask,
            )
        if (
            self.mid_morphology_spatial_affines is not None
            and morphology_spatial_mask is not None
        ):
            h = self.mid_morphology_spatial_affines[1](
                h,
                morphology_spatial_mask,
                morphology_route_mask,
            )
        if (
            self.mid_state_delta_affines is not None
            and state_target_class_ids is not None
            and state_anchor_class_ids is not None
        ):
            h = self.mid_state_delta_affines[1](
                h,
                state_target_class_ids,
                state_anchor_class_ids,
                state_delta_route_mask,
            )
        if (
            self.mid_target_state_response_affines is not None
            and state_target_class_ids is not None
        ):
            h = self.mid_target_state_response_affines[1](
                h,
                state_target_class_ids,
                target_state_response_route_mask,
            )

        upsample_idx = 0
        for i in range(len(self.up_blocks)):
            for j in range(len(self.up_blocks[i])):
                h = self.up_merges[i][j](torch.cat([h, skips.pop()], dim=1))
                h = self.up_blocks[i][j](h, t_emb)
                if (
                    self.up_morphology_affines is not None
                    and morphology_class_ids is not None
                ):
                    h = self.up_morphology_affines[i][j](
                        h,
                        morphology_class_ids,
                        morphology_route_mask,
                    )
                if (
                    self.up_morphology_spatial_affines is not None
                    and morphology_spatial_mask is not None
                ):
                    h = self.up_morphology_spatial_affines[i][j](
                        h,
                        morphology_spatial_mask,
                        morphology_route_mask,
                    )
                if (
                    self.up_state_delta_affines is not None
                    and state_target_class_ids is not None
                    and state_anchor_class_ids is not None
                ):
                    h = self.up_state_delta_affines[i][j](
                        h,
                        state_target_class_ids,
                        state_anchor_class_ids,
                        state_delta_route_mask,
                    )
                if (
                    self.up_target_state_response_affines is not None
                    and state_target_class_ids is not None
                ):
                    h = self.up_target_state_response_affines[i][j](
                        h,
                        state_target_class_ids,
                        target_state_response_route_mask,
                    )
                a = self.up_attns[i][j]
                h = (
                    a(
                        h,
                        cond_seq,
                        factor_delta_context=factor_delta_context,
                        factor_delta_mask=factor_delta_mask,
                    )
                    if isinstance(a, BasicTransformerBlock)
                    else a(h)
                )
                r = self.up_revs[i][j]
                h = r(h, cond_pooled) if isinstance(r, ReversibleCondBlock) else r(h)
            if i < len(self.upsamples):
                h = self.upsamples[upsample_idx](h)
                upsample_idx += 1

        return self.out_conv(F.silu(self.out_norm(h)))
