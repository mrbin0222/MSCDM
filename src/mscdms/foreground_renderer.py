from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return groups


class ConditionedConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.affine = nn.Linear(cond_dim, out_channels * 2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.norm1(self.conv1(inputs))
        gamma, beta = self.affine(condition).chunk(2, dim=1)
        hidden = hidden * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        hidden = F.silu(hidden)
        return F.silu(self.norm2(self.conv2(hidden)))


class ForegroundRendererUNet(nn.Module):
    """Render foreground RGB from an explicit support mask and factor IDs."""

    def __init__(
        self,
        *,
        num_sources: int,
        num_states: int,
        base_channels: int = 16,
        condition_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_sources = int(num_sources)
        self.num_states = int(num_states)
        self.base_channels = int(base_channels)
        self.condition_dim = int(condition_dim)
        self.source_embedding = nn.Embedding(self.num_sources, self.condition_dim)
        self.state_embedding = nn.Embedding(self.num_states, self.condition_dim)
        self.condition = nn.Sequential(
            nn.Linear(self.condition_dim * 2, self.condition_dim),
            nn.SiLU(),
            nn.Linear(self.condition_dim, self.condition_dim),
        )
        c = self.base_channels
        self.down1 = ConditionedConvBlock(4, c, self.condition_dim)
        self.down2 = ConditionedConvBlock(c, c * 2, self.condition_dim)
        self.down3 = ConditionedConvBlock(c * 2, c * 4, self.condition_dim)
        self.bottleneck = ConditionedConvBlock(c * 4, c * 8, self.condition_dim)
        self.up3 = ConditionedConvBlock(c * 12, c * 4, self.condition_dim)
        self.up2 = ConditionedConvBlock(c * 6, c * 2, self.condition_dim)
        self.up1 = ConditionedConvBlock(c * 3, c, self.condition_dim)
        self.output = nn.Conv2d(c, 3, 1)

    @property
    def config(self) -> dict[str, int]:
        return {
            "num_sources": self.num_sources,
            "num_states": self.num_states,
            "base_channels": self.base_channels,
            "condition_dim": self.condition_dim,
        }

    def forward(
        self,
        background: torch.Tensor,
        mask: torch.Tensor,
        source_ids: torch.Tensor,
        state_ids: torch.Tensor,
    ) -> torch.Tensor:
        if background.ndim != 4 or background.shape[1] != 3:
            raise ValueError("background must have shape [B,3,H,W]")
        if mask.shape != (len(background), 1, *background.shape[-2:]):
            raise ValueError("mask must have shape [B,1,H,W]")
        if source_ids.shape != (len(background),) or state_ids.shape != (
            len(background),
        ):
            raise ValueError("factor IDs must have shape [B]")
        condition = self.condition(
            torch.cat(
                [self.source_embedding(source_ids), self.state_embedding(state_ids)],
                dim=1,
            )
        )
        first = self.down1(torch.cat([background, mask], dim=1), condition)
        second = self.down2(F.avg_pool2d(first, 2), condition)
        third = self.down3(F.avg_pool2d(second, 2), condition)
        bottleneck = self.bottleneck(F.avg_pool2d(third, 2), condition)
        up_third = F.interpolate(
            bottleneck, size=third.shape[-2:], mode="bilinear", align_corners=False
        )
        up_third = self.up3(torch.cat([up_third, third], dim=1), condition)
        up_second = F.interpolate(
            up_third, size=second.shape[-2:], mode="bilinear", align_corners=False
        )
        up_second = self.up2(torch.cat([up_second, second], dim=1), condition)
        up_first = F.interpolate(
            up_second, size=first.shape[-2:], mode="bilinear", align_corners=False
        )
        up_first = self.up1(torch.cat([up_first, first], dim=1), condition)
        return self.output(up_first).tanh()


def compose_exact_background(
    background: torch.Tensor,
    foreground: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if background.shape != foreground.shape:
        raise ValueError("background and foreground shapes differ")
    if mask.shape != (len(background), 1, *background.shape[-2:]):
        raise ValueError("invalid composition mask")
    binary = (mask >= 0.5).to(dtype=background.dtype)
    return foreground * binary + background * (1.0 - binary)


def foreground_reconstruction_loss(
    foreground: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    binary = (mask >= 0.5).to(dtype=foreground.dtype)
    denominator = (binary.sum() * foreground.shape[1]).clamp_min(1.0)
    foreground_l1 = ((foreground - target).abs() * binary).sum() / denominator
    eroded = 1.0 - F.max_pool2d(1.0 - binary, 5, stride=1, padding=2)
    edge = (binary - eroded).clamp(0.0, 1.0)
    edge_denominator = (edge.sum() * foreground.shape[1]).clamp_min(1.0)
    edge_l1 = ((foreground - target).abs() * edge).sum() / edge_denominator
    loss = foreground_l1 + 0.5 * edge_l1
    return loss, {
        "foreground_l1": foreground_l1.detach(),
        "edge_l1": edge_l1.detach(),
    }


def load_foreground_renderer(
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[ForegroundRendererUNet, dict[str, Any]]:
    payload: dict[str, Any] = torch.load(
        str(checkpoint), map_location=map_location, weights_only=False
    )
    if payload.get("schema") != "exp4.foreground_renderer.stage26.checkpoint.v1":
        raise ValueError(f"unsupported foreground-renderer checkpoint: {checkpoint}")
    model = ForegroundRendererUNet(**payload["model_config"])
    model.load_state_dict(payload["model"], strict=True)
    return model, payload
