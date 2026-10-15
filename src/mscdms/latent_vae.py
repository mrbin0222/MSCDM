from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ResBlock2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.conv(x)


class AttnBlock2d(nn.Module):
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


class MidBlock(nn.Module):
    def __init__(self, ch: int, num_heads: int = 8):
        super().__init__()
        self.rb1 = ResBlock2d(ch, ch)
        self.attn = AttnBlock2d(ch, num_heads=num_heads)
        self.rb2 = ResBlock2d(ch, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rb1(x)
        x = self.attn(x)
        x = self.rb2(x)
        return x


class LatentAutoEncoder(nn.Module):
    """
    单尺度图像 <-> 潜空间，下采样因子 8（与文档 128->16 等一致）。
    训练扩散时编码器产生 z；解码器仅推理阶段将最终潜变量还原为 RGB。
    """

    def __init__(
        self,
        in_ch: int = 3,
        z_ch: int = 4,
        base_ch: int = 128,
        ch_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_heads: int = 8,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.z_ch = z_ch
        self.gradient_checkpointing = gradient_checkpointing
        chs = [base_ch * m for m in ch_mult]

        self.enc_in = nn.Conv2d(in_ch, chs[0], 3, padding=1)
        self.enc_blocks = nn.ModuleList()
        self.enc_downs = nn.ModuleList()
        cur = chs[0]
        for i, ch in enumerate(chs):
            self.enc_blocks.append(ResBlock2d(cur, ch))
            cur = ch
            if i < 3:
                self.enc_downs.append(nn.Conv2d(cur, chs[i + 1], 4, 2, 1))
                cur = chs[i + 1]
        self.enc_mid = MidBlock(cur, num_heads=num_heads)
        self.enc_out = nn.Sequential(nn.GroupNorm(8, cur), nn.SiLU(), nn.Conv2d(cur, z_ch * 2, 3, padding=1))

        self.dec_in = nn.Conv2d(z_ch, cur, 3, padding=1)
        self.dec_mid = MidBlock(cur, num_heads=num_heads)
        self.dec_blocks = nn.ModuleList()
        self.dec_ups = nn.ModuleList()
        for i in reversed(range(len(chs))):
            ch = chs[i]
            self.dec_blocks.append(ResBlock2d(cur, ch))
            cur = ch
            if i > 0 and i <= 3:
                self.dec_ups.append(nn.ConvTranspose2d(cur, chs[i - 1], 4, 2, 1))
                cur = chs[i - 1]
        self.dec_out = nn.Sequential(nn.GroupNorm(8, cur), nn.SiLU(), nn.Conv2d(cur, in_ch, 3, padding=1))

    def _encode_impl(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_in(x)
        for i, rb in enumerate(self.enc_blocks):
            h = rb(h)
            if i < len(self.enc_downs):
                h = self.enc_downs[i](h)
        h = self.enc_mid(h)
        stats = self.enc_out(h)
        mean, logvar = stats.chunk(2, dim=1)
        return mean, logvar

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gradient_checkpointing and self.training:
            return checkpoint(self._encode_impl, x, use_reentrant=False)
        return self._encode_impl(x)

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def _decode_impl(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_in(z)
        h = self.dec_mid(h)
        up_idx = 0
        for i, rb in enumerate(self.dec_blocks):
            h = rb(h)
            if i < len(self.dec_ups):
                h = self.dec_ups[up_idx](h)
                up_idx += 1
        return self.dec_out(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(self._decode_impl, z, use_reentrant=False)
        return self._decode_impl(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        xr = self.decode(z)
        return z, xr, mean, logvar


def vae_reconstruction_loss(x: torch.Tensor, xr: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(xr, x)


def vae_kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(torch.exp(logvar) + mean.pow(2) - 1.0 - logvar)
