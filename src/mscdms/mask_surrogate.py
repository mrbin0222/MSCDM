from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class MaskSurrogateUNet(nn.Module):
    """Small differentiable image-to-instance-mask surrogate."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.down1 = ConvBlock(self.in_channels, self.base_channels)
        self.down2 = ConvBlock(self.base_channels, self.base_channels * 2)
        self.bottleneck = ConvBlock(self.base_channels * 2, self.base_channels * 4)
        self.up2 = ConvBlock(self.base_channels * 6, self.base_channels * 2)
        self.up1 = ConvBlock(self.base_channels * 3, self.base_channels)
        self.output = nn.Conv2d(self.base_channels, 1, 1)

    @property
    def config(self) -> dict[str, int]:
        return {
            "in_channels": self.in_channels,
            "base_channels": self.base_channels,
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError("mask surrogate input must have shape [B,3,H,W]")
        first = self.down1(inputs)
        second = self.down2(F.avg_pool2d(first, 2))
        bottleneck = self.bottleneck(F.avg_pool2d(second, 2))
        up_second = F.interpolate(
            bottleneck,
            size=second.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        up_second = self.up2(torch.cat([up_second, second], dim=1))
        up_first = F.interpolate(
            up_second,
            size=first.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.output(self.up1(torch.cat([up_first, first], dim=1)))


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError("mask logits and targets must have the same shape")
    probabilities = logits.float().sigmoid()
    targets = targets.float()
    numerator = 2.0 * (probabilities * targets).sum(dim=(1, 2, 3)) + 1.0
    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
        + 1.0
    )
    return (1.0 - numerator / denominator).mean()


def mask_surrogate_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    bce = F.binary_cross_entropy_with_logits(logits.float(), targets.float())
    dice = dice_loss(logits, targets)
    return bce + dice, {"bce": bce.detach(), "dice": dice.detach()}


class FrozenMaskSurrogate(nn.Module):
    def __init__(self, checkpoint: str | Path) -> None:
        super().__init__()
        payload: dict[str, Any] = torch.load(
            str(checkpoint),
            map_location="cpu",
            weights_only=False,
        )
        if payload.get("schema") != "exp4.mask_surrogate.checkpoint.v1":
            raise ValueError(f"unsupported mask-surrogate checkpoint: {checkpoint}")
        self.network = MaskSurrogateUNet(**payload["model_config"])
        self.network.load_state_dict(payload["model"], strict=True)
        self.source_name = str(payload["source_name"])
        self.image_size = int(payload["image_size"])
        self.requires_grad_(False)
        self.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return self.network(images)


class FrozenSourceMaskSurrogates(nn.Module):
    """Route BBBC021/DynamicNuclearNet images to frozen source judges."""

    SOURCE_TO_ID = {"bbbc021": 0, "dynamicnuclearnet": 1}

    def __init__(
        self,
        *,
        bbbc021_checkpoint: str | Path,
        dynamicnuclearnet_checkpoint: str | Path,
    ) -> None:
        super().__init__()
        self.surrogates = nn.ModuleDict(
            {
                "bbbc021": FrozenMaskSurrogate(bbbc021_checkpoint),
                "dynamicnuclearnet": FrozenMaskSurrogate(
                    dynamicnuclearnet_checkpoint
                ),
            }
        )
        for source, surrogate in self.surrogates.items():
            if surrogate.source_name != source:
                raise ValueError(
                    f"mask-surrogate source mismatch: {source} != "
                    f"{surrogate.source_name}"
                )
        image_sizes = {module.image_size for module in self.surrogates.values()}
        if len(image_sizes) != 1:
            raise ValueError("source mask surrogates use different resolutions")
        self.image_size = int(next(iter(image_sizes)))
        self.requires_grad_(False)
        self.eval()

    def audit(self) -> dict[str, Any]:
        return {
            "sources": dict(self.SOURCE_TO_ID),
            "image_size": self.image_size,
            "all_frozen": all(
                not parameter.requires_grad for parameter in self.parameters()
            ),
        }

    def forward(
        self,
        images: torch.Tensor,
        source_ids: torch.Tensor,
    ) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("source mask-surrogate images must be BCHW RGB")
        if source_ids.ndim != 1 or len(source_ids) != len(images):
            raise ValueError("source IDs must have shape [batch]")
        invalid = sorted(
            set(int(value) for value in source_ids.detach().cpu().tolist())
            - set(self.SOURCE_TO_ID.values())
        )
        if invalid:
            raise ValueError(f"invalid mask-surrogate source IDs: {invalid}")
        logits = images.new_zeros(
            (len(images), 1, self.image_size, self.image_size)
        )
        for source, source_id in self.SOURCE_TO_ID.items():
            selected = source_ids == source_id
            if bool(selected.any()):
                source_logits = self.surrogates[source](images[selected])
                # Autocast may produce bf16 surrogate outputs while `images`
                # (and therefore the aggregation tensor) remain float32.
                # Normalize only at the aggregation boundary; the cast is
                # differentiable, so training gradients still reach images.
                logits[selected] = source_logits.to(dtype=logits.dtype)
        return logits
