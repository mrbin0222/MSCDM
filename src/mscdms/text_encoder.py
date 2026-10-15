from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer


_EXP4_ROOT = Path(__file__).resolve().parents[1]


def resolve_local_pretrained_path(pretrained_name: str) -> str:
    """
    把配置中的相对路径解析到 exp4 根目录，并在模型缺失时立即报错。

    这里有意不提供 Hub 名称回退，避免服务器训练到一半才因网络失败。
    """
    raw = Path(pretrained_name).expanduser()
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, _EXP4_ROOT / raw]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate.resolve())
    checked = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"未找到本地预训练模型目录 {pretrained_name!r}；已检查: {checked}。"
        "请先运行 scripts/materialize_pretrained_models.py。"
    )


class TextBranchHead(nn.Module):
    """轻量分支头：把共享主干特征映射到项目维度。"""

    def __init__(self, clip_dim: int, dim: int):
        super().__init__()
        self.seq_proj = nn.Linear(clip_dim, dim) if clip_dim != dim else nn.Identity()
        self.pool_proj = nn.Linear(clip_dim, dim) if clip_dim != dim else nn.Identity()

    def forward(self, seq: torch.Tensor, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.seq_proj(seq), self.pool_proj(pooled)


class DeterministicResidualMLP(nn.Module):
    """Two-layer residual MLP with an exact identity initial function.

    The down projection is a deterministic coordinate selector and the up
    projection is exactly zero.  Construction therefore consumes no model RNG,
    while the first backward pass can immediately update the up projection.
    """

    def __init__(self, dim: int, rank: int):
        super().__init__()
        if dim <= 0 or rank <= 0 or rank > dim:
            raise ValueError(
                f"state residual dimensions require 0 < rank <= dim, got {rank}/{dim}"
            )
        self.dim = int(dim)
        self.rank = int(rank)
        self.norm = nn.LayerNorm(dim)
        self.down_weight = nn.Parameter(torch.zeros(rank, dim))
        self.up_weight = nn.Parameter(torch.zeros(dim, rank))
        with torch.no_grad():
            diagonal = torch.arange(rank)
            self.down_weight[diagonal, diagonal] = 1.0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = F.linear(self.norm(value), self.down_weight)
        residual = F.linear(F.silu(hidden), self.up_weight)
        return value + residual

    def has_exact_initial_state(self) -> bool:
        with torch.no_grad():
            expected_down = torch.zeros_like(self.down_weight)
            diagonal = torch.arange(self.rank, device=expected_down.device)
            expected_down[diagonal, diagonal] = 1.0
            return bool(
                torch.equal(self.down_weight, expected_down)
                and torch.count_nonzero(self.up_weight).item() == 0
                and torch.equal(self.norm.weight, torch.ones_like(self.norm.weight))
                and torch.equal(self.norm.bias, torch.zeros_like(self.norm.bias))
            )


class StateSpecificResidualRepresentation(nn.Module):
    """Independent state-token and state-pool residual representations."""

    def __init__(self, dim: int, rank: int = 64):
        super().__init__()
        self.sequence = DeterministicResidualMLP(dim, rank)
        self.pooled = DeterministicResidualMLP(dim, rank)

    def forward(
        self,
        sequence: torch.Tensor,
        pooled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequence(sequence), self.pooled(pooled)

    def has_exact_initial_state(self) -> bool:
        return (
            self.sequence.has_exact_initial_state()
            and self.pooled.has_exact_initial_state()
        )


class MultiScaleTextEncoder(nn.Module):
    """
    三类语义嵌入 c_sub, c_state, c_bg（式 4-2）。
    训练时配合 classifier-free guidance：随机丢弃某一路条件。
    """

    _tokenizers: dict[str, CLIPTokenizer] = {}

    def __init__(
        self,
        dim: int,
        max_len: int = 64,
        pretrained_name: str = "models/clip-vit-base-patch32",
        *,
        local_files_only: bool = True,
        freeze_backbone: bool = True,
        state_specific_residual_rank: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.pretrained_name = resolve_local_pretrained_path(pretrained_name)
        self.local_files_only = local_files_only
        self.freeze_backbone = freeze_backbone
        self.clip = CLIPTextModel.from_pretrained(
            self.pretrained_name,
            local_files_only=local_files_only,
        )
        if freeze_backbone:
            self.clip.requires_grad_(False)
            self.clip.eval()
        clip_dim = self.clip.config.hidden_size
        self.subj = TextBranchHead(clip_dim, dim)
        self.state = TextBranchHead(clip_dim, dim)
        self.bg = TextBranchHead(clip_dim, dim)
        self.state_specific_residual = (
            StateSpecificResidualRepresentation(
                dim,
                rank=int(state_specific_residual_rank),
            )
            if state_specific_residual_rank is not None
            else None
        )

    @classmethod
    def _get_tokenizer(cls, pretrained_name: str, local_files_only: bool = True) -> CLIPTokenizer:
        resolved = resolve_local_pretrained_path(pretrained_name)
        if resolved not in cls._tokenizers:
            cls._tokenizers[resolved] = CLIPTokenizer.from_pretrained(
                resolved,
                local_files_only=local_files_only,
            )
        return cls._tokenizers[resolved]

    @staticmethod
    def texts_to_batch(
        subject: list[str],
        state: list[str],
        bg: list[str],
        device: torch.device,
        max_len: int,
        pretrained_name: str = "models/clip-vit-base-patch32",
        local_files_only: bool = True,
    ):
        tok = MultiScaleTextEncoder._get_tokenizer(pretrained_name, local_files_only)

        def pack(lines):
            enc = tok(
                lines,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            return enc.input_ids.to(device=device, dtype=torch.long)

        return pack(subject), pack(state), pack(bg)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            # model.train() 不得把冻结的 CLIP 主干切回训练态。
            self.clip.eval()
        return self

    def forward(
        self,
        subject_ids: torch.Tensor,
        state_ids: torch.Tensor,
        bg_ids: torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        # 共享 CLIP 主干：把三路 batch 合并后一次前向，再拆分给各自轻量头。
        b = subject_ids.shape[0]
        merged_ids = torch.cat([subject_ids, state_ids, bg_ids], dim=0)
        if self.freeze_backbone:
            with torch.no_grad():
                out = self.clip(input_ids=merged_ids)
        else:
            out = self.clip(input_ids=merged_ids)
        seq_all = out.last_hidden_state
        pool_all = out.pooler_output
        s_seq, st_seq, bg_seq = seq_all[:b], seq_all[b : 2 * b], seq_all[2 * b :]
        s_pool, st_pool, bg_pool = pool_all[:b], pool_all[b : 2 * b], pool_all[2 * b :]
        state_condition = self.state(st_seq, st_pool)
        if self.state_specific_residual is not None:
            state_condition = self.state_specific_residual(*state_condition)
        return self.subj(s_seq, s_pool), state_condition, self.bg(bg_seq, bg_pool)


class ContrastiveHead(nn.Module):
    """图像侧特征，用于式 4-23 InfoNCE。"""

    def __init__(self, in_ch: int = 3, dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        z = F.normalize(self.proj(h), dim=-1)
        return z


def contrastive_loss(
    z_img: torch.Tensor,
    z_txt: torch.Tensor,
    tau: float = 0.07,
) -> torch.Tensor:
    """对称 InfoNCE，z_txt 为拼接后的文本表征或全局 prompt 向量。"""
    logits = (z_img @ z_txt.T) / tau
    labels = torch.arange(z_img.shape[0], device=z_img.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
