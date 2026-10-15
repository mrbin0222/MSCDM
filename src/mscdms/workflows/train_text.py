"""Train text."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import logging
import math
import random
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from mscdms import MSCDM, MSCDMConfig
from mscdms.checkpoint import config_from_checkpoint, load_checkpoint
from mscdms.dataset import (
    CellMicroscopyDataset,
    CounterfactualPairDataset,
    discover_dataset_size,
)
from mscdms.distributed import cleanup_distributed, get_device, is_main_process, setup_distributed
from mscdms.reversible import ReversibleCondBlock
from mscdms.model import (
    build_cascade_prior_source_mask,
    build_isolated_condition_drop_masks,
    cascade_prior_probability,
)
from mscdms.sampling import (
    SourceBalancedSmallAreaSampler,
    SourceMorphologyBalancedSampler,
    SupportCellBalancedSampler,
)
from mscdms.text_encoder import MultiScaleTextEncoder

logger = logging.getLogger(__name__)
NUCLEUS_GATED_MODE = "subject_state_low_nucleus"
NUCLEUS_ADAPTER_MODE = "subject_state_low_nucleus_adapter"
NUCLEUS_MULTISCALE_ADAPTER_MODE = (
    "subject_state_low_nucleus_multiscale_adapter"
)
NUCLEUS_PROMPT_ADAPTER_MODE = (
    "subject_state_low_nucleus_prompt_adapters"
)
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
MORPHOLOGY_STRUCTURED_MODES = {
    MORPHOLOGY_LATENT_BANK_MODE,
    MORPHOLOGY_FEATURE_AFFINE_MODE,
}
MORPHOLOGY_SPATIAL_MASK_MODES = {
    SPATIAL_SUBJECT_INPUT_ADAPTER_MODE,
    MORPHOLOGY_SPATIAL_AFFINE_MODE,
}
NUCLEUS_GATED_MODES = {
    NUCLEUS_GATED_MODE,
    NUCLEUS_ADAPTER_MODE,
    NUCLEUS_MULTISCALE_ADAPTER_MODE,
    NUCLEUS_PROMPT_ADAPTER_MODE,
    NUCLEUS_DISTILL_PROMPT_ADAPTER_MODE,
    NUCLEUS_DISTILL_OUTPUT_ADAPTER_MODE,
    NUCLEUS_TEMPORAL_ADAPTER_MODE,
}


class DemoTensorDataset(Dataset):
    def __init__(self, n: int = 256, return_object_type: bool = False):
        self.n = n
        self.return_object_type = return_object_type

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        generator = torch.Generator().manual_seed(i)
        x = torch.randn(3, 512, 512, generator=generator) * 0.5
        result = (
            x,
            "MCF7 cell",
            "interphase state",
            "fluorescence microscopy, DAPI",
            "",
        )
        if self.return_object_type:
            return (*result, "nucleus" if i % 2 == 0 else "whole_cell")
        return result


class FactorPopulationBatchSampler:
    """DDP-safe same-target population batches from a factor manifest."""

    def __init__(
        self,
        records: list[dict],
        *,
        factor: str,
        population_size: int,
        num_replicas: int,
        rank: int,
        seed: int,
    ) -> None:
        if population_size <= 1:
            raise ValueError("population_size must exceed one")
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("invalid replica configuration")
        self.population_size = int(population_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.by_target: dict[int, list[int]] = {}
        for index, record in enumerate(records):
            payload = record.get("factor_surrogate")
            if not isinstance(payload, dict):
                continue
            enabled = payload.get("enabled", {})
            targets = payload.get("target_ids", {})
            if not bool(enabled.get(factor, False)):
                continue
            target = int(targets.get(factor, -1))
            if target < 0:
                raise ValueError("enabled population factor has no target")
            self.by_target.setdefault(target, []).append(index)
        if not self.by_target:
            raise ValueError(f"no population targets found for factor {factor}")
        if any(
            len(indices) < self.population_size
            for indices in self.by_target.values()
        ):
            raise ValueError("a population target has too few samples")
        total_groups = sum(
            len(indices) // self.population_size
            for indices in self.by_target.values()
        )
        self.groups_per_replica = (
            total_groups + self.num_replicas - 1
        ) // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.groups_per_replica

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        groups: list[list[int]] = []
        for target in sorted(self.by_target):
            indices = list(self.by_target[target])
            rng.shuffle(indices)
            usable = (
                len(indices) // self.population_size
            ) * self.population_size
            for start in range(0, usable, self.population_size):
                groups.append(indices[start : start + self.population_size])
        rng.shuffle(groups)
        total_size = self.groups_per_replica * self.num_replicas
        if len(groups) < total_size:
            groups.extend(groups[: total_size - len(groups)])
        elif len(groups) > total_size:
            groups = groups[:total_size]
        yield from groups[self.rank:total_size:self.num_replicas]


class StateContrastPopulationBatchSampler:
    """DDP-safe 16+16 state batches with fixed subject and background."""

    STATE_TO_ID = {
        "untreated_control": 0,
        "positive_control_sirna": 1,
    }

    def __init__(
        self,
        records: list[dict],
        *,
        population_size: int,
        num_replicas: int,
        rank: int,
        seed: int,
    ) -> None:
        if population_size <= 1:
            raise ValueError("population_size must exceed one")
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("invalid replica configuration")
        self.population_size = int(population_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.by_condition: dict[tuple[str, str], dict[int, list[int]]] = {}
        for index, record in enumerate(records):
            state_key = str(record.get("state_family_key") or "")
            if state_key not in self.STATE_TO_ID:
                raise ValueError(
                    f"unsupported state_family_key: {state_key!r}"
                )
            subject_key = str(record.get("subject_key") or "")
            background_key = str(record.get("background_key") or "")
            if not subject_key or not background_key:
                raise ValueError(
                    "state population records require subject/background keys"
                )
            key = (subject_key, background_key)
            state_id = self.STATE_TO_ID[state_key]
            states = self.by_condition.setdefault(key, {0: [], 1: []})
            states[state_id].append(index)
        if not self.by_condition:
            raise ValueError("no state population conditions found")
        for key, states in self.by_condition.items():
            if any(len(states[state_id]) < self.population_size for state_id in (0, 1)):
                raise ValueError(
                    f"condition {key} has too few samples for both states"
                )
        total_groups = sum(
            min(
                len(states[0]) // self.population_size,
                len(states[1]) // self.population_size,
            )
            for states in self.by_condition.values()
        )
        self.groups_per_replica = (
            total_groups + self.num_replicas - 1
        ) // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.groups_per_replica

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        groups: list[list[int]] = []
        for key in sorted(self.by_condition):
            states = self.by_condition[key]
            negative = list(states[0])
            positive = list(states[1])
            rng.shuffle(negative)
            rng.shuffle(positive)
            group_count = min(
                len(negative) // self.population_size,
                len(positive) // self.population_size,
            )
            for group_index in range(group_count):
                start = group_index * self.population_size
                stop = start + self.population_size
                groups.append(
                    negative[start:stop] + positive[start:stop]
                )
        rng.shuffle(groups)
        total_size = self.groups_per_replica * self.num_replicas
        if len(groups) < total_size:
            groups.extend(groups[: total_size - len(groups)])
        elif len(groups) > total_size:
            groups = groups[:total_size]
        yield from groups[self.rank:total_size:self.num_replicas]


class V31MulticlassStatePopulationBatchSampler:
    """DDP-safe source-local balanced batches for v3.1 two/three-state data."""

    STATE_TO_ID = {
        "RxRx1": {
            "untreated_control": 0,
            "positive_control_sirna": 1,
        },
        "BBBC021": {
            "dmso_vehicle_control": 0,
            "microtubule_destabilization": 1,
            "protein_synthesis_inhibition": 2,
        },
        "BBBC022": {
            "bbbc022_mock_control": 0,
            "microtubule_destabilization": 1,
            "protein_synthesis_inhibition": 2,
        },
    }

    def __init__(
        self,
        records: list[dict],
        *,
        population_size: int,
        num_replicas: int,
        rank: int,
        seed: int,
    ) -> None:
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("invalid replica configuration")
        self.population_size = int(population_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.by_condition: dict[tuple[str, str, str], dict[int, list[int]]] = {}
        self.class_count_by_condition: dict[tuple[str, str, str], int] = {}
        for index, record in enumerate(records):
            source = str(record.get("source_dataset") or "")
            state_key = str(record.get("state_family_key") or "")
            mapping = self.STATE_TO_ID.get(source)
            if mapping is None or state_key not in mapping:
                raise ValueError(
                    f"unsupported source/state pair: {source!r}/{state_key!r}"
                )
            subject_key = str(record.get("subject_key") or "")
            background_key = str(record.get("background_key") or "")
            if not subject_key or not background_key:
                raise ValueError(
                    "multiclass state records require subject/background keys"
                )
            key = (source, subject_key, background_key)
            class_count = len(mapping)
            states = self.by_condition.setdefault(
                key, {target: [] for target in range(class_count)}
            )
            self.class_count_by_condition[key] = class_count
            states[mapping[state_key]].append(index)
        if not self.by_condition:
            raise ValueError("no multiclass state population conditions found")
        self.groups_per_condition = min(
            min(
                len(states[target]) // self.population_size
                for target in range(self.class_count_by_condition[key])
            )
            for key, states in self.by_condition.items()
        )
        if self.groups_per_condition <= 0:
            raise ValueError("at least one condition cannot form a state group")
        self.groups_per_replica_by_class_count: dict[int, int] = {}
        for class_count in (2, 3):
            condition_count = 0
            for key, states in self.by_condition.items():
                if self.class_count_by_condition[key] != class_count:
                    continue
                if any(
                    len(states[target]) < self.population_size
                    for target in range(class_count)
                ):
                    raise ValueError(f"condition {key} has too few state samples")
                condition_count += 1
            total_groups = condition_count * self.groups_per_condition
            if total_groups:
                self.groups_per_replica_by_class_count[class_count] = (
                    total_groups + self.num_replicas - 1
                ) // self.num_replicas
        if set(self.groups_per_replica_by_class_count) != {2, 3}:
            raise ValueError("v3.1 batches require both two- and three-state groups")
        self.groups_per_replica = sum(
            self.groups_per_replica_by_class_count.values()
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.groups_per_replica

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        local_groups_by_class_count: dict[int, list[list[int]]] = {}
        for class_count in sorted(self.groups_per_replica_by_class_count):
            groups: list[list[int]] = []
            for key in sorted(self.by_condition):
                if self.class_count_by_condition[key] != class_count:
                    continue
                states = {
                    target: list(self.by_condition[key][target])
                    for target in range(class_count)
                }
                for indices in states.values():
                    rng.shuffle(indices)
                group_count = self.groups_per_condition
                for group_index in range(group_count):
                    start = group_index * self.population_size
                    stop = start + self.population_size
                    groups.append(
                        [
                            index
                            for target in range(class_count)
                            for index in states[target][start:stop]
                        ]
                    )
            rng.shuffle(groups)
            total_size = (
                self.groups_per_replica_by_class_count[class_count]
                * self.num_replicas
            )
            if len(groups) < total_size:
                groups.extend(groups[: total_size - len(groups)])
            elif len(groups) > total_size:
                groups = groups[:total_size]
            local_groups = groups[self.rank:total_size:self.num_replicas]
            rng.shuffle(local_groups)
            local_groups_by_class_count[class_count] = local_groups
        # Interleave two- and three-state batches. A two-batch smoke therefore
        # exercises both source-local class counts on every rank, while a full
        # epoch preserves exactly the same groups and counts.
        local_groups: list[list[int]] = []
        max_groups = max(map(len, local_groups_by_class_count.values()))
        for group_index in range(max_groups):
            for class_count in sorted(local_groups_by_class_count):
                groups = local_groups_by_class_count[class_count]
                if group_index < len(groups):
                    local_groups.append(groups[group_index])
        yield from local_groups


def make_v31_state_trajectory_generator(
    *,
    device: torch.device,
    base_seed: int,
    global_step: int,
    rank: int,
) -> torch.Generator:
    """Build a resume-stable RNG stream isolated from base diffusion noise."""
    if base_seed < 0 or global_step < 0 or rank < 0:
        raise ValueError("trajectory RNG seed, global step and rank must be non-negative")
    generator = torch.Generator(device=device)
    generator.manual_seed(base_seed + global_step * 1_000_003 + rank)
    return generator


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--manifest_name", default="manifest_hybrid.jsonl")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="仅用于本地训练链路冒烟，不用于正式实验")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32, help="每张 GPU 的 batch size")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--clip_model_dir", default="models/clip-vit-base-patch32")
    ap.add_argument("--no_vae_gradient_checkpointing", action="store_true")
    ap.add_argument("--normalize", choices=("percentile", "minmax", "none"), default="percentile")
    ap.add_argument("--val_every", type=int, default=1)
    ap.add_argument(
        "--checkpoint_selection_metric", choices=("total", "high"), default="total",
        help="Historical runs use total; P10 H0/U/C use the common high v-pred loss.",
    )
    ap.add_argument(
        "--val_fixed_seed", type=int, default=None,
        help="Reset validation RNG per rank and restore training RNG afterward.",
    )
    ap.add_argument(
        "--contrastive_weight", type=float, default=None,
        help="Override only the image-text contrastive loss weight; P10 C uses zero.",
    )
    ap.add_argument(
        "--p10_smoke_audit", action="store_true",
        help="Write per-rank gradient, frozen-parameter, validation-RNG and CUDA-peak audit for isolated P10 smoke runs.",
    )
    ap.add_argument("--max_train_batches", type=int, default=0)
    ap.add_argument("--max_val_batches", type=int, default=0)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--grad_clip_norm", type=float, default=1.0)
    ap.add_argument("--loss_spike_warn_factor", type=float, default=0.0)
    ap.add_argument("--cond_drop_prob", type=float, default=0.1)
    ap.add_argument(
        "--cond_drop_mode",
        choices=(
            "joint",
            "independent",
            "independent_matched",
            "stratified",
        ),
        default="joint",
        help=(
            "joint 沿用旧版三路同时丢弃；independent 独立丢弃 "
            "subject/state/background；independent_matched 和 stratified "
            "使用隔离随机流，分别对应0.325边际对照与八类固定配额。"
        ),
    )
    ap.add_argument(
        "--cond_drop_seed",
        type=int,
        default=20260815,
        help="matched/stratified 条件遮蔽专用随机种子；不推进基础扩散随机流。",
    )
    ap.add_argument(
        "--control_rank_weight",
        type=float,
        default=0.0,
        help=(
            "同图同噪声反事实条件排名损失权重；大于 0 时 manifest 必须含 "
            "control_negatives 与 control_rank_enabled。"
        ),
    )
    ap.add_argument(
        "--control_rank_margin",
        type=float,
        default=0.01,
        help="真实条件的逐样本扩散 MSE 应优于单属性错配条件的相对间隔。",
    )
    ap.add_argument(
        "--control_rank_scope",
        choices=("single_branch", "routed_all"),
        default="single_branch",
        help=(
            "single_branch 沿用按名义尺度计算排名损失；routed_all 在所有实际收到"
            "该因子的尺度上同时比较正确/错误提示。"
        ),
    )
    ap.add_argument(
        "--factor_surrogate_weight",
        type=float,
        default=0.0,
        help=(
            "冻结独立 factor surrogate 的单属性目标分类损失权重；"
            "0 为关闭。"
        ),
    )
    ap.add_argument("--factor_surrogate_subject_probe", default=None)
    ap.add_argument("--factor_surrogate_dnn_state_probe", default=None)
    ap.add_argument("--factor_surrogate_background_probe", default=None)
    ap.add_argument("--factor_surrogate_dino_model_dir", default=None)
    ap.add_argument(
        "--factor_surrogate_dnn_state_dino_checkpoint",
        default=None,
    )
    ap.add_argument(
        "--factor_surrogate_dnn_state_distribution_bank",
        default=None,
        help=(
            "train-only real DNN state-logit bank; when provided, population "
            "energy distance replaces hard state cross-entropy."
        ),
    )
    ap.add_argument(
        "--factor_surrogate_subject_image_probe",
        default=None,
        help=(
            "12-class decoded-image identity probe used for final-image "
            "subject supervision."
        ),
    )
    ap.add_argument(
        "--factor_surrogate_subject_distribution_bank",
        default=None,
        help=(
            "train-only real identity-logit bank; when provided, population "
            "energy distance replaces hard subject cross-entropy."
        ),
    )
    ap.add_argument(
        "--factor_surrogate_population_factor",
        choices=("subject", "state_dnn", "background"),
        default="state_dnn",
        help=(
            "Factor whose target IDs define same-target population batches. "
            "The default preserves the Stage-7/8/9 state behavior."
        ),
    )
    ap.add_argument(
        "--factor_surrogate_state_population_size",
        type=int,
        default=1,
    )
    ap.add_argument(
        "--factor_surrogate_fixed_factor",
        type=int,
        choices=(-1, 0, 1, 2),
        default=-1,
        help="-1 cycles subject/state/background; 0/1/2 fixes one factor.",
    )
    ap.add_argument(
        "--factor_surrogate_timestep_min",
        type=int,
        default=0,
    )
    ap.add_argument(
        "--factor_surrogate_timestep_max",
        type=int,
        default=249,
    )
    ap.add_argument(
        "--state_population_delta_weight",
        type=float,
        default=0.0,
        help=(
            "训练数据高尺度潜空间中，未处理到阳性处理的真实群体差异匹配权重；"
            "不使用任何冻结评价器。"
        ),
    )
    ap.add_argument(
        "--state_population_size",
        type=int,
        default=1,
        help="每种状态在同主体、同背景训练批次中的图像数；1 关闭群体组包。",
    )
    ap.add_argument(
        "--state_population_timestep_min",
        type=int,
        default=750,
    )
    ap.add_argument(
        "--state_population_timestep_max",
        type=int,
        default=999,
    )
    ap.add_argument(
        "--state_population_pool_size",
        type=int,
        default=8,
    )
    ap.add_argument(
        "--state_prompt_causal_mid_weight",
        type=float,
        default=0.0,
        help=(
            "中尺度状态文本因果差异损失权重；固定同一状态中性 mid 潜变量、"
            "同一 low 先验和同一噪声，只切换未处理/阳性处理文本。"
        ),
    )
    ap.add_argument(
        "--state_prompt_causal_timestep_min",
        type=int,
        default=750,
    )
    ap.add_argument(
        "--state_prompt_causal_timestep_max",
        type=int,
        default=999,
    )
    ap.add_argument(
        "--state_prompt_causal_pool_size",
        type=int,
        default=8,
    )
    ap.add_argument(
        "--state_prompt_causal_high_centroid_weight",
        type=float,
        default=0.0,
        help=(
            "高尺度两状态群体中心校准权重；固定状态中性 high carrier、"
            "状态中性 mid prior 和配对噪声，仅靠状态文本分别匹配两种真实中心。"
        ),
    )
    ap.add_argument(
        "--state_prompt_causal_high_timestep_min",
        type=int,
        default=750,
    )
    ap.add_argument(
        "--state_prompt_causal_high_timestep_max",
        type=int,
        default=999,
    )
    ap.add_argument(
        "--state_prompt_causal_high_pool_size",
        type=int,
        default=8,
    )
    ap.add_argument(
        "--state_trajectory_centroid_weight",
        type=float,
        default=0.0,
        help=(
            "v3.1多状态对称绝对中心损失权重；在state进入low/mid/high时，"
            "分别匹配每个来源内每个目标状态的真实训练潜变量中心。"
        ),
    )
    ap.add_argument(
        "--state_trajectory_population_size",
        type=int,
        default=0,
        help=(
            "v3.1每来源、主体、背景、状态的群体样本数；0关闭专用成组采样。"
        ),
    )
    ap.add_argument(
        "--state_trajectory_pool_size",
        type=int,
        default=8,
    )
    ap.add_argument(
        "--state_trajectory_timestep",
        type=int,
        default=999,
        help="纯噪声轨迹辅助路径的起始扩散时间步。",
    )
    ap.add_argument(
        "--state_trajectory_noise_seed",
        type=int,
        default=20260812,
        help=(
            "纯噪声轨迹辅助路径的独立随机种子；不推进基础扩散训练随机流，"
            "用于保持权重0与非零候选的成对可比性。"
        ),
    )
    ap.add_argument(
        "--state_early_ddim_centroid_weight",
        type=float,
        default=0.0,
        help=(
            "v3.1高噪声low前两步DDIM绝对状态中心权重；严格复用25步推理的"
            "t=999到957轨迹和subject CFG=2。"
        ),
    )
    ap.add_argument(
        "--state_early_ddim_population_size",
        type=int,
        default=0,
        help="每个来源内状态的群体样本数；0关闭专用成组采样。",
    )
    ap.add_argument(
        "--state_early_ddim_pool_size",
        type=int,
        default=8,
    )
    ap.add_argument(
        "--state_early_ddim_inference_steps",
        type=int,
        default=25,
        help="冻结为25；用于取得与正式推理完全相同的整数时间表。",
    )
    ap.add_argument(
        "--state_early_ddim_trace_steps",
        type=int,
        default=2,
        help="冻结为2；只展开正式DDIM最早的两个高噪声步骤。",
    )
    ap.add_argument(
        "--state_early_ddim_subject_cfg",
        type=float,
        default=2.0,
        help="冻结为2；state/background CFG保持1。",
    )
    ap.add_argument(
        "--state_early_ddim_noise_seed",
        type=int,
        default=20260812,
        help="独立于普通扩散损失随机流的DDIM起始噪声种子。",
    )
    ap.add_argument(
        "--state_early_ddim_probe_gradients",
        action="store_true",
        help=(
            "仅用于lr=0的train-split梯度校准探针；记录同一low U-Net参数"
            "集合上的基础low损失和未加权DDIM辅助损失梯度。"
        ),
    )
    ap.add_argument(
        "--morphology_mask_surrogate_weight",
        type=float,
        default=0.0,
        help=(
            "反事实 morphology 显式 mask 几何损失权重；0 为关闭。"
        ),
    )
    ap.add_argument(
        "--morphology_mask_surrogate_bbbc021_checkpoint",
        default=None,
    )
    ap.add_argument(
        "--morphology_mask_surrogate_dynamicnuclearnet_checkpoint",
        default=None,
    )
    ap.add_argument(
        "--morphology_mask_surrogate_timestep_min",
        type=int,
        default=0,
    )
    ap.add_argument(
        "--morphology_mask_surrogate_timestep_max",
        type=int,
        default=249,
    )
    ap.add_argument(
        "--morphology_pair_preservation_weight",
        type=float,
        default=0.0,
        help=(
            "同噪声事实/反事实 morphology 双路径的非目标保持损失权重；"
            "背景像素和前景逐通道均值/标准差均由事实路径停止梯度锚定。"
        ),
    )
    ap.add_argument(
        "--counterfactual_subject_pairs",
        default=None,
        help="相对于 data_dir 的 train-only subject 真实反事实配对清单。",
    )
    ap.add_argument(
        "--counterfactual_state_pairs",
        default=None,
        help="相对于 data_dir 的 train-only state 真实反事实配对清单。",
    )
    ap.add_argument(
        "--counterfactual_pair_weight",
        type=float,
        default=0.0,
        help="共同噪声真实配对方向损失权重；0 为关闭。",
    )
    ap.add_argument(
        "--counterfactual_pair_batch_size",
        type=int,
        default=4,
        help="每张 GPU 的反事实配对 batch size。",
    )
    ap.add_argument(
        "--counterfactual_teacher_checkpoint",
        default=None,
        help="冻结 low/mid U-Net 源端输出锚定 checkpoint。",
    )
    ap.add_argument(
        "--counterfactual_teacher_anchor_mix",
        type=float,
        default=1.0,
        help="源端 teacher prediction anchor 相对方向损失的倍率。",
    )
    ap.add_argument(
        "--counterfactual_target_orthogonal_mix",
        type=float,
        default=0.0,
        help=(
            "目标端相对冻结 teacher 的非目标正交漂移惩罚倍率；"
            "0 保持 Stage-2a v1 行为。"
        ),
    )
    ap.add_argument(
        "--counterfactual_timestep_min",
        type=int,
        default=500,
    )
    ap.add_argument(
        "--counterfactual_timestep_max",
        type=int,
        default=999,
    )
    ap.add_argument(
        "--observation_rgb_pairs",
        default=None,
        help=(
            "相对于 data_dir 或绝对路径的 train-only 同细胞双RGB观察配对清单。"
        ),
    )
    ap.add_argument(
        "--observation_rgb_pair_weight",
        type=float,
        default=0.0,
        help=(
            "同细胞、同噪声 observation prompt-switch RGB latent 差分损失权重。"
        ),
    )
    ap.add_argument(
        "--observation_rgb_pair_center_weight",
        type=float,
        default=0.25,
        help="双观察预测中心相对冻结正式E1中心的保持倍率。",
    )
    ap.add_argument(
        "--observation_rgb_pair_batch_size",
        type=int,
        default=2,
        help="每张GPU的同细胞双RGB观察配对batch size。",
    )
    ap.add_argument(
        "--observation_rgb_pair_timestep_min",
        type=int,
        default=500,
    )
    ap.add_argument(
        "--observation_rgb_pair_timestep_max",
        type=int,
        default=999,
    )
    ap.add_argument(
        "--prediction_type",
        choices=("epsilon", "v"),
        default=None,
        help="扩散预测目标；新 high-noise 稳定性门禁推荐 v，默认沿用 checkpoint/配置",
    )
    ap.add_argument("--sample_every", type=int, default=10, help="每 N epoch 做一次定性采样；0 关闭")
    ap.add_argument("--num_val_samples", type=int, default=4)
    ap.add_argument("--val_vis_seed", type=int, default=None)
    ap.add_argument("--sample_cfg_scale", type=float, default=None)
    ap.add_argument("--sample_num_steps", type=int, default=None)
    ap.add_argument("--save", default="checkpoints/mscdm.pt")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--best_path", default=None)
    ap.add_argument("--resume", default=None, help="从完整 checkpoint_last/epoch_XXXX.pt 恢复")
    ap.add_argument(
        "--model_init_checkpoint",
        default=None,
        help="从完整模型 checkpoint 初始化权重和模型配置，但重置 optimizer、epoch 与 best validation",
    )
    ap.add_argument(
        "--vae_init_checkpoint",
        default=None,
        help="仅从旧 checkpoint 加载 vae_low/vae_mid/vae_high，其余模块重新初始化",
    )
    ap.add_argument(
        "--latent_calibration",
        default=None,
        help="三分支逐通道 latent calibration JSON；启用 affine normalization",
    )
    ap.add_argument("--freeze_vae", action="store_true", help="冻结三路 VAE 并使用确定性 posterior mean")
    ap.add_argument(
        "--train_limit_per_source",
        type=int,
        default=0,
        help="每个 source 最多保留 N 个 train 样本；用于小样本过拟合门禁，0 表示不限制",
    )
    ap.add_argument(
        "--train_sampling_strategy",
        choices=(
            "uniform",
            "small_area_source_v1",
            "source_morphology_balanced_v1",
            "support_cell_balanced_v1",
        ),
        default="uniform",
        help=(
            "训练样本策略；small_area_source_v1 保持来源总量并提高指定来源小目标抽样率；"
            "support_cell_balanced_v1 在不改变每轮长度的前提下均衡事实 "
            "source×subject×state×background 条件单元"
        ),
    )
    ap.add_argument("--hard_sample_source", default="idr0033")
    ap.add_argument("--hard_sample_area_quantile", type=float, default=0.25)
    ap.add_argument("--hard_sample_multiplier", type=float, default=2.0)
    ap.add_argument(
        "--foreground_loss_mix",
        type=float,
        default=0.0,
        help=(
            "扩散 MSE 中前景掩膜均值的混合比例；0 为原全图损失，"
            "例如 0.25 表示 75%% 全图 MSE + 25%% 前景 MSE"
        ),
    )
    ap.add_argument(
        "--conditioning_mode",
        choices=(
            "hierarchical",
            "uniform_all_factors",
            "cumulative",
            SEMANTIC_MULTISCALE_MODE,
            SEMANTIC_STATE_ALLSCALE_MODE,
            SEMANTIC_STATE_SEQ_LOW_MODE,
            SEMANTIC_STATE_FACTOR_RESIDUAL_POOL_MODE,
            SEMANTIC_STATE_FACTOR_PARTITION_ATTENTION_MODE,
            SEMANTIC_STATE_FACTOR_ISOLATED_COMBINED_MODE,
            SEMANTIC_STATE_SPECIFIC_RESIDUAL_MODE,
            "subject_state_low",
            NUCLEUS_GATED_MODE,
            NUCLEUS_ADAPTER_MODE,
            NUCLEUS_MULTISCALE_ADAPTER_MODE,
            NUCLEUS_PROMPT_ADAPTER_MODE,
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
        ),
        default=None,
        help=(
            "文本条件路由；hierarchical 为 v4 原路径，cumulative 让 subject/state "
            "累积进入后续尺度，semantic_multiscale 使用 low=subject+background、"
            "mid/high=state+background，semantic_state_allscale 进一步让 state 进入 low，"
            "semantic_state_seq_low 保留low state token但从low pooled中移除state，"
            "semantic_state_factor_residual_pool在原pooled均值上增加分因子零初始化残差，"
            "semantic_state_factor_partition_attention用零初始化标量门接入分因子token softmax，"
            "semantic_state_factor_isolated_combined同时启用上述两种分因子隔离路径，"
            "semantic_state_specific_residual为state token和pooled各增加一个函数保持的两层残差MLP，"
            "subject_state_low 将 subject+state 前移到 low，"
            "subject_state_low_nucleus 仅对 nucleus 启用该低层融合，"
            "subject_state_low_nucleus_adapter 使用 nucleus 低分支残差适配器，"
            "subject_state_low_nucleus_multiscale_adapter 对三个尺度均使用输出残差适配器，"
            "subject_state_low_nucleus_prompt_adapters 在 U-Net 前使用三路文本门控 token adapter。"
            "hierarchical_nucleus_distill_prompt_adapters 保持原分层路径并仅对 nucleus "
            "启用跨 VAE 图像空间蒸馏 adapter。"
            "hierarchical_nucleus_distill_output_adapters 保持原分层条件并仅对 nucleus "
            "的三尺度噪声预测增加蒸馏残差。"
            "subject_state_low_nucleus_temporal_multiscale_adapter 保持原始分层条件，"
            "仅在 nucleus 低噪声后期启用三尺度残差 adapter。"
            "hierarchical_factor_prompt_adapters 保持原始 hierarchical 路由，"
            "仅增加三路 factor-specific token adapter。"
            "hierarchical_factor_delta_adapters 保持普通生成原路径，只在"
            "显式反事实目标上注入 target-anchor 单因子差分。"
            "hierarchical_factor_attention_delta_adapters 保持普通生成原路径，"
            "只在显式编辑时向目标尺度 cross-attention K/V 注入差分。"
            "hierarchical_rgb_factor_output_adapters 冻结 hierarchical 基座，"
            "只给 morphology 与 observation 增加三尺度 RGB latent 输出残差。"
            "hierarchical_spatial_subject_input_adapter 在 low U-Net 输入前"
            "注入显式形态掩膜，未路由样本保持原路径。"
            "默认沿用 checkpoint/配置。"
        ),
    )
    ap.add_argument(
        "--trainable_scope",
        choices=(
            "all",
            "control_low_unet",
            "control_low_mid_unets",
            "control_mid_unet",
            "control_mid_high_unets",
            "control_high_unet",
            "nucleus_low_adapter",
            "nucleus_scale_adapters",
            "nucleus_prompt_adapters",
            "nucleus_distill_prompt_adapters",
            "nucleus_distill_output_adapters",
            "factor_prompt_adapters",
            "factor_delta_adapters",
            "factor_attention_delta_adapters",
            "rgb_factor_output_adapters",
            "rgb_observation_output_adapters",
            "spatial_subject_input_adapter",
            "morphology_latent_bank",
            "morphology_feature_affine",
            "morphology_spatial_affine",
        ),
        default="all",
        help=(
            "默认训练所有未冻结参数；control_low_unet 仅更新 subject，"
            "control_low_mid_unets 仅更新 subject/state 对应扩散分支；"
            "control_mid_unet 仅更新 state "
            "中尺度扩散分支；control_high_unet 仅更新最终高尺度扩散分支；"
            "adapter pilot 可仅训练专用适配器。"
        ),
    )
    ap.add_argument(
        "--dataset_object_type",
        choices=("nucleus", "whole_cell"),
        default=None,
        help="可选：将 train/val manifest 同时限定到一种 object_type。",
    )
    ap.add_argument(
        "--prior_proxy_mix",
        type=float,
        default=0.0,
        help=(
            "中、高分支训练先验中停止梯度的上一级单步 x0 预测占比；"
            "0 为原 teacher prior，0.5 为各占一半。推理路径不变。"
        ),
    )
    ap.add_argument(
        "--cascade_prior_schedule",
        choices=("off", "gpt_pro_v2_v1"),
        default="off",
        help=(
            "单步自预测级联先验的固定训练日程；gpt_pro_v2_v1按轮次使用"
            "0.25/0.50/0.75/1.00的模型预测先验概率。"
        ),
    )
    ap.add_argument(
        "--cascade_prior_gradient_mode",
        choices=("off", "stopgrad", "e2e"),
        default="off",
        help=(
            "stopgrad与e2e前向值相同，仅决定mid/high事实损失是否跨尺度反传。"
        ),
    )
    ap.add_argument(
        "--cascade_prior_mask_seed",
        type=int,
        default=20260816,
        help="隔离的级联先验来源mask随机种子。",
    )
    ap.add_argument(
        "--repeat_first_train_batch",
        action="store_true",
        help="仅用于实现冒烟：每轮重复首个rank-local batch。",
    )
    ap.add_argument("--train_ckpt_every", type=int, default=5)
    ap.add_argument("--train_ckpt_max_keep", type=int, default=3)
    ap.add_argument(
        "--p10_export_initial", default=None,
        help="Export one new, frozen step-0 full-model initialization, then exit before training.",
    )
    return ap.parse_args()


def losses_all_finite(losses: dict[str, torch.Tensor]) -> bool:
    return all(not torch.is_tensor(v) or torch.isfinite(v).all() for v in losses.values())


def validation_selection_loss(
    losses: dict[str, torch.Tensor], metric: str
) -> torch.Tensor:
    if metric not in {"total", "high"}:
        raise ValueError(f"unknown validation selection metric: {metric}")
    return losses["loss_high" if metric == "high" else "loss_total"]


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = (tensor.detach().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def pick_val_prompts(
    dataset: Dataset,
    seed: int,
    count: int,
) -> list[tuple[str, str, str, str, str]]:
    if len(dataset) == 0 or count <= 0:
        return []
    indices = random.Random(seed).sample(range(len(dataset)), min(count, len(dataset)))
    prompts = []
    for index in indices:
        sample = dataset[index]
        object_type = (
            str(sample[5])
            if bool(getattr(dataset, "return_object_type", False))
            else ""
        )
        prompts.append(
            (sample[1], sample[2], sample[3], str(sample[4]), object_type)
        )
    return prompts


def limit_manifest_dataset_per_source(
    dataset: CellMicroscopyDataset,
    limit: int,
    seed: int,
) -> dict[str, int]:
    """确定性地为每个数据源截取少量样本，供扩散链路过拟合门禁使用。"""
    if limit <= 0:
        return {}
    if not dataset.uses_manifest:
        raise ValueError("--train_limit_per_source 只支持 JSONL manifest 数据集")
    grouped: dict[str, list[dict]] = {}
    for record in dataset.records:
        source = str(record.get("source_name", "")).strip()
        if not source:
            raise ValueError("manifest 使用 --train_limit_per_source 时，每条记录必须含 source_name")
        grouped.setdefault(source, []).append(record)
    rng = random.Random(seed)
    selected: list[dict] = []
    counts: dict[str, int] = {}
    for source in sorted(grouped):
        candidates = grouped[source]
        chosen = rng.sample(candidates, min(limit, len(candidates)))
        selected.extend(chosen)
        counts[source] = len(chosen)
    selected.sort(key=lambda item: str(item.get("sample_id", item["image_path"])))
    dataset.records = selected
    return counts


def filter_manifest_dataset_object_type(
    dataset: CellMicroscopyDataset,
    object_type: str | None,
) -> int:
    if object_type is None:
        return len(dataset)
    if not dataset.uses_manifest:
        raise ValueError("--dataset_object_type 只支持 JSONL manifest 数据集")
    dataset.records = [
        record
        for record in dataset.records
        if str(record.get("object_type") or "") == object_type
    ]
    if not dataset.records:
        raise ValueError(f"manifest 中没有 object_type={object_type!r} 的样本")
    return len(dataset.records)


def apply_latent_calibration(cfg: MSCDMConfig, path: str | Path) -> dict:
    calibration_path = Path(path)
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    branches = payload.get("branches")
    if not isinstance(branches, dict):
        raise ValueError(f"{calibration_path} 缺少 branches")
    for branch in ("low", "mid", "high"):
        stats = branches.get(branch)
        if not isinstance(stats, dict):
            raise ValueError(f"{calibration_path} 缺少 branches.{branch}")
        mean = tuple(float(value) for value in stats.get("channel_mean", ()))
        std = tuple(float(value) for value in stats.get("channel_std", ()))
        if len(mean) != cfg.latent_channels or len(std) != cfg.latent_channels:
            raise ValueError(
                f"{branch} calibration 通道数不匹配："
                f"mean={len(mean)} std={len(std)} expected={cfg.latent_channels}"
            )
        if any(value <= 0 for value in std):
            raise ValueError(f"{branch} calibration std 必须全部大于 0")
        setattr(cfg, f"latent_mean_{branch}", mean)
        setattr(cfg, f"latent_std_{branch}", std)
    cfg.latent_normalization = True
    return payload


def configure_control_unet_trainable_scope(
    model: nn.Module,
    scope: str,
) -> tuple[str, ...] | None:
    scope_to_modules = {
        "control_low_unet": ("unet_low",),
        "control_low_mid_unets": ("unet_low", "unet_mid"),
        "control_mid_unet": ("unet_mid",),
        "control_mid_high_unets": ("unet_mid", "unet_high"),
        "control_high_unet": ("unet_high",),
    }
    module_names = scope_to_modules.get(scope)
    if module_names is None:
        return None
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module_name in module_names:
        module = getattr(model, module_name)
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return module_names


def enforce_control_unet_frozen_eval_mode(
    model: nn.Module,
    scope: str,
) -> None:
    if scope == "control_mid_unet":
        model.unet_low.eval()
        model.unet_high.eval()
    elif scope == "control_low_unet":
        model.unet_mid.eval()
        model.unet_high.eval()
    elif scope == "control_high_unet":
        model.unet_low.eval()
        model.unet_mid.eval()


def load_vae_initialization(model: MSCDM, checkpoint_path: str | Path) -> None:
    checkpoint = load_checkpoint(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("--vae_init_checkpoint 必须指向含 model 的 checkpoint")
    source_state = {
        key.removeprefix("module."): value
        for key, value in checkpoint["model"].items()
    }
    prefixes = ("vae_low.", "vae_mid.", "vae_high.")
    vae_state = {key: value for key, value in source_state.items() if key.startswith(prefixes)}
    expected = {key for key in model.state_dict() if key.startswith(prefixes)}
    missing = sorted(expected - set(vae_state))
    extra = sorted(set(vae_state) - expected)
    if missing or extra:
        raise ValueError(
            "VAE checkpoint 结构不匹配："
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    incompatible = model.load_state_dict(vae_state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"VAE 初始化含意外参数：{incompatible.unexpected_keys[:5]}")


def build_frozen_counterfactual_teacher_unets(
    model: MSCDM,
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, nn.Module]:
    """Load only original-v4 low/mid U-Nets for source-output anchoring."""
    checkpoint = load_checkpoint(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(
            "--counterfactual_teacher_checkpoint 必须指向含 model 的 checkpoint"
        )
    source_state = {
        key.removeprefix("module."): value
        for key, value in checkpoint["model"].items()
    }
    teachers: list[nn.Module] = []
    for branch_name, student in (
        ("unet_low", model.unet_low),
        ("unet_mid", model.unet_mid),
    ):
        prefix = f"{branch_name}."
        branch_state = {
            key.removeprefix(prefix): value
            for key, value in source_state.items()
            if key.startswith(prefix)
        }
        expected = set(student.state_dict())
        if set(branch_state) != expected:
            missing = sorted(expected - set(branch_state))
            extra = sorted(set(branch_state) - expected)
            raise ValueError(
                f"{branch_name} teacher checkpoint mismatch: "
                f"missing={missing[:5]} extra={extra[:5]}"
            )
        teacher = copy.deepcopy(student)
        teacher.load_state_dict(branch_state)
        teacher.requires_grad_(False)
        teacher.eval()
        teachers.append(teacher.to(device))
    return teachers[0], teachers[1]


def tokenize(
    batch,
    device: torch.device,
    cfg: MSCDMConfig,
    *,
    use_control_negatives: bool = False,
    use_morphology_mask_target: bool = False,
    use_morphology_class: bool = False,
    use_morphology_counterfactual_class: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    images = batch[0].to(device, non_blocking=True)
    subject_ids, state_ids, bg_ids = MultiScaleTextEncoder.texts_to_batch(
        list(batch[1]),
        list(batch[2]),
        list(batch[3]),
        device,
        max_len=cfg.max_text_len,
        pretrained_name=cfg.text_pretrained_name,
        local_files_only=cfg.text_local_files_only,
    )
    next_index = 5
    subject_state_low_mask = None
    if cfg.conditioning_mode in NUCLEUS_GATED_MODES:
        if len(batch) <= next_index:
            raise ValueError(
                "subject_state_low_nucleus requires object_type from the dataset"
            )
        object_types = [str(value) for value in batch[next_index]]
        invalid = sorted(set(object_types) - {"nucleus", "whole_cell"})
        if invalid:
            raise ValueError(f"invalid object_type values: {invalid}")
        subject_state_low_mask = torch.tensor(
            [value == "nucleus" for value in object_types],
            device=device,
            dtype=torch.bool,
        )
        next_index += 1
    negative_subject_ids = None
    negative_state_ids = None
    negative_bg_ids = None
    control_rank_enabled = None
    if use_control_negatives:
        if len(batch) <= next_index + 5:
            raise ValueError(
                "control rank requires three negative prompts and "
                "three enabled masks from the dataset"
            )
        (
            negative_subject_ids,
            negative_state_ids,
            negative_bg_ids,
        ) = MultiScaleTextEncoder.texts_to_batch(
            list(batch[next_index]),
            list(batch[next_index + 1]),
            list(batch[next_index + 2]),
            device,
            max_len=cfg.max_text_len,
            pretrained_name=cfg.text_pretrained_name,
            local_files_only=cfg.text_local_files_only,
        )
        control_rank_enabled = torch.stack(
            [
                torch.as_tensor(
                    batch[next_index + offset],
                    device=device,
                    dtype=torch.bool,
                )
                for offset in (3, 4, 5)
            ],
            dim=1,
        )
        next_index += 6
    morphology_mask_target = None
    morphology_mask_source_ids = None
    if use_morphology_mask_target:
        if len(batch) <= next_index + 1:
            raise ValueError(
                "morphology mask supervision requires target masks and "
                "source IDs from the dataset"
            )
        morphology_mask_target = batch[next_index].to(
            device,
            non_blocking=True,
        )
        morphology_mask_source_ids = torch.as_tensor(
            batch[next_index + 1],
            device=device,
            dtype=torch.long,
        )
        next_index += 2
    morphology_class_ids = None
    if use_morphology_class:
        if len(batch) <= next_index:
            raise ValueError(
                "morphology latent bank requires class IDs from the dataset"
            )
        morphology_class_ids = torch.as_tensor(
            batch[next_index],
            device=device,
            dtype=torch.long,
        )
        next_index += 1
    morphology_counterfactual_class_ids = None
    if use_morphology_counterfactual_class:
        if len(batch) <= next_index:
            raise ValueError(
                "morphology paired supervision requires counterfactual "
                "class IDs from the dataset"
            )
        morphology_counterfactual_class_ids = torch.as_tensor(
            batch[next_index],
            device=device,
            dtype=torch.long,
        )
        next_index += 1
    target_masks = (
        batch[next_index].to(device, non_blocking=True)
        if len(batch) > next_index
        else None
    )
    return (
        images,
        subject_ids,
        state_ids,
        bg_ids,
        negative_subject_ids,
        negative_state_ids,
        negative_bg_ids,
        control_rank_enabled,
        target_masks,
        subject_state_low_mask,
        morphology_mask_target,
        morphology_mask_source_ids,
        morphology_class_ids,
        morphology_counterfactual_class_ids,
    )


def tokenize_counterfactual_pair(
    batch,
    device: torch.device,
    cfg: MSCDMConfig,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    anchor_images = batch[0].to(device, non_blocking=True)
    target_images = batch[1].to(device, non_blocking=True)
    anchor_ids = MultiScaleTextEncoder.texts_to_batch(
        list(batch[2]),
        list(batch[3]),
        list(batch[4]),
        device,
        max_len=cfg.max_text_len,
        pretrained_name=cfg.text_pretrained_name,
        local_files_only=cfg.text_local_files_only,
    )
    target_ids = MultiScaleTextEncoder.texts_to_batch(
        list(batch[5]),
        list(batch[6]),
        list(batch[7]),
        device,
        max_len=cfg.max_text_len,
        pretrained_name=cfg.text_pretrained_name,
        local_files_only=cfg.text_local_files_only,
    )
    factor_values = torch.as_tensor(batch[8], dtype=torch.long)
    if factor_values.ndim != 1 or not bool(
        (factor_values == factor_values[0]).all()
    ):
        raise ValueError("each counterfactual pair batch must contain one factor")
    factor = int(factor_values[0].item())
    return (
        anchor_images,
        target_images,
        *anchor_ids,
        *target_ids,
        factor,
    )


def build_factor_surrogate_record_lookup(
    dataset: Dataset,
) -> dict[str, dict]:
    if not isinstance(dataset, CellMicroscopyDataset) or not dataset.uses_manifest:
        raise ValueError("factor surrogate training requires a manifest dataset")
    lookup: dict[str, dict] = {}
    for record in dataset.records:
        payload = record.get("factor_surrogate")
        if not isinstance(payload, dict):
            raise ValueError(
                f"{record.get('sample_id')}: missing factor_surrogate payload"
            )
        target_ids = payload.get("target_ids")
        enabled = payload.get("enabled")
        if not isinstance(target_ids, dict) or not isinstance(enabled, dict):
            raise ValueError(
                f"{record.get('sample_id')}: invalid factor_surrogate payload"
            )
        key = str(Path(record["_resolved_image_path"]).resolve())
        if key in lookup:
            raise ValueError(f"duplicate factor-surrogate image path: {key}")
        lookup[key] = payload
    return lookup


def factor_surrogate_targets_from_batch(
    batch,
    lookup: dict[str, dict],
    device: torch.device,
) -> torch.Tensor:
    paths = [str(Path(value).resolve()) for value in batch[4]]
    rows = []
    for path in paths:
        if path not in lookup:
            raise KeyError(f"factor-surrogate target not found for {path}")
        target_ids = lookup[path]["target_ids"]
        rows.append(
            [
                int(target_ids["subject"]),
                int(target_ids["state_dnn"]),
                int(target_ids["background"]),
            ]
        )
    return torch.tensor(rows, device=device, dtype=torch.long)


def build_state_population_record_lookup(
    dataset: Dataset,
) -> dict[str, tuple[int, str, str]]:
    if not isinstance(dataset, CellMicroscopyDataset) or not dataset.uses_manifest:
        raise ValueError("state population training requires a manifest dataset")
    lookup: dict[str, tuple[int, str, str]] = {}
    for record in dataset.records:
        state_key = str(record.get("state_family_key") or "")
        if state_key not in StateContrastPopulationBatchSampler.STATE_TO_ID:
            raise ValueError(
                f"unsupported state_family_key: {state_key!r}"
            )
        key = str(Path(record["_resolved_image_path"]).resolve())
        if key in lookup:
            raise ValueError(f"duplicate state-population image path: {key}")
        lookup[key] = (
            StateContrastPopulationBatchSampler.STATE_TO_ID[state_key],
            str(record.get("subject_key") or ""),
            str(record.get("background_key") or ""),
        )
    return lookup


def state_population_labels_from_batch(
    batch,
    lookup: dict[str, tuple[int, str, str]],
    device: torch.device,
    population_size: int,
) -> torch.Tensor:
    rows = []
    for value in batch[4]:
        path = str(Path(value).resolve())
        if path not in lookup:
            raise KeyError(f"state-population target not found for {path}")
        rows.append(lookup[path])
    if len(rows) != 2 * population_size:
        raise ValueError("state population batch has unexpected size")
    if len({row[1] for row in rows}) != 1:
        raise ValueError("state population batch mixes subject keys")
    if len({row[2] for row in rows}) != 1:
        raise ValueError("state population batch mixes background keys")
    labels = [row[0] for row in rows]
    if labels.count(0) != population_size or labels.count(1) != population_size:
        raise ValueError("state population batch is not balanced")
    return torch.tensor(labels, device=device, dtype=torch.long)


def build_v31_multiclass_state_record_lookup(
    dataset: Dataset,
) -> dict[str, tuple[int, str, str, str]]:
    if not isinstance(dataset, CellMicroscopyDataset) or not dataset.uses_manifest:
        raise ValueError("v3.1 multiclass state training requires manifest data")
    lookup: dict[str, tuple[int, str, str, str]] = {}
    for record in dataset.records:
        source = str(record.get("source_dataset") or "")
        state_key = str(record.get("state_family_key") or "")
        mapping = V31MulticlassStatePopulationBatchSampler.STATE_TO_ID.get(source)
        if mapping is None or state_key not in mapping:
            raise ValueError(
                f"unsupported source/state pair: {source!r}/{state_key!r}"
            )
        key = str(Path(record["_resolved_image_path"]).resolve())
        if key in lookup:
            raise ValueError(f"duplicate multiclass-state image path: {key}")
        lookup[key] = (
            mapping[state_key],
            source,
            str(record.get("subject_key") or ""),
            str(record.get("background_key") or ""),
        )
    return lookup


def v31_multiclass_state_labels_from_batch(
    batch,
    lookup: dict[str, tuple[int, str, str, str]],
    device: torch.device,
    population_size: int,
) -> torch.Tensor:
    rows = []
    for value in batch[4]:
        path = str(Path(value).resolve())
        if path not in lookup:
            raise KeyError(f"multiclass-state target not found for {path}")
        rows.append(lookup[path])
    if not rows or population_size <= 0:
        raise ValueError("multiclass-state batch is empty or population size invalid")
    if len({row[1] for row in rows}) != 1:
        raise ValueError("multiclass-state batch mixes source datasets")
    if len({row[2] for row in rows}) != 1:
        raise ValueError("multiclass-state batch mixes subject keys")
    if len({row[3] for row in rows}) != 1:
        raise ValueError("multiclass-state batch mixes background keys")
    labels = [row[0] for row in rows]
    classes = sorted(set(labels))
    if classes not in ([0, 1], [0, 1, 2]):
        raise ValueError("multiclass-state labels must be two or three contiguous classes")
    if len(rows) != len(classes) * population_size:
        raise ValueError("multiclass-state batch has unexpected size")
    if any(labels.count(value) != population_size for value in classes):
        raise ValueError("multiclass-state batch is not balanced")
    return torch.tensor(labels, device=device, dtype=torch.long)


def v31_multiclass_state_group_from_batch(
    batch,
    lookup: dict[str, tuple[int, str, str, str]],
) -> dict[str, str]:
    rows = []
    for value in batch[4]:
        path = str(Path(value).resolve())
        if path not in lookup:
            raise KeyError(f"multiclass-state target not found for {path}")
        rows.append(lookup[path])
    groups = {(row[1], row[2], row[3]) for row in rows}
    if len(groups) != 1:
        raise ValueError("multiclass-state batch mixes source/subject/background")
    source, subject, background = next(iter(groups))
    return {
        "source_dataset": source,
        "subject_key": subject,
        "background_key": background,
    }


@torch.no_grad()
def run_qualitative(
    epoch_display: int,
    model: MSCDM,
    device: torch.device,
    cfg: MSCDMConfig,
    prompts: list[tuple[str, str, str, str, str]],
    output_root: Path,
    cfg_scale: float | None,
    num_steps: int | None,
    sample_seed: int,
) -> None:
    if not prompts:
        return
    subject_ids, state_ids, bg_ids = MultiScaleTextEncoder.texts_to_batch(
        [p[0] for p in prompts],
        [p[1] for p in prompts],
        [p[2] for p in prompts],
        device,
        max_len=cfg.max_text_len,
        pretrained_name=cfg.text_pretrained_name,
        local_files_only=cfg.text_local_files_only,
    )
    generator = torch.Generator(device=device).manual_seed(sample_seed)
    subject_state_low_mask = None
    if cfg.conditioning_mode in NUCLEUS_GATED_MODES:
        object_types = [prompt[4] for prompt in prompts]
        if any(value not in {"nucleus", "whole_cell"} for value in object_types):
            raise ValueError(
                "gated qualitative prompts require nucleus/whole_cell object_type"
            )
        subject_state_low_mask = torch.tensor(
            [value == "nucleus" for value in object_types],
            device=device,
            dtype=torch.bool,
        )
    with torch.amp.autocast("cuda", enabled=False):
        rgb = model.sample(
            subject_ids,
            state_ids,
            bg_ids,
            cfg_scale=cfg_scale,
            num_steps=num_steps,
            generator=generator,
            subject_state_low_mask=subject_state_low_mask,
        )
    epoch_dir = output_root / "samples" / f"epoch_{epoch_display:04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    prompt_records = []
    for index, (image, prompt) in enumerate(zip(rgb, prompts)):
        out_path = epoch_dir / f"sample_{index:02d}.png"
        tensor_to_pil(image).save(out_path)
        prompt_records.append(
            {
                "index": index,
                "subject": prompt[0],
                "state": prompt[1],
                "background": prompt[2],
                "reference_image": prompt[3],
                "object_type": prompt[4],
                "generated_image": str(out_path),
            }
        )
    write_json(epoch_dir / "prompts.json", prompt_records)
    write_json(
        epoch_dir / "sampling.json",
        {
            "cfg_scale": cfg.cfg_scale if cfg_scale is None else cfg_scale,
            "num_steps": cfg.num_inference_steps if num_steps is None else num_steps,
            "seed": sample_seed,
        },
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    initialization_modes = [
        bool(args.resume),
        bool(args.model_init_checkpoint),
        bool(args.vae_init_checkpoint),
    ]
    if sum(initialization_modes) > 1:
        raise ValueError(
            "--resume、--model_init_checkpoint 与 --vae_init_checkpoint 只能选择一个"
        )
    if (args.resume or args.model_init_checkpoint) and args.latent_calibration:
        raise ValueError(
            "完整 checkpoint 已恢复模型配置，不应再次传 --latent_calibration"
        )
    if not 0.0 <= args.cond_drop_prob <= 1.0:
        raise ValueError("--cond_drop_prob 必须位于 [0, 1]")
    if args.cond_drop_seed < 0:
        raise ValueError("--cond_drop_seed 不能为负数")
    if args.cond_drop_mode in {"independent_matched", "stratified"}:
        if not math.isclose(args.cond_drop_prob, 0.325, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "matched/stratified v1 固定 cond_drop_prob=0.325，禁止扫描"
            )
        if args.batch_size != 32:
            raise ValueError("matched/stratified v1 固定每卡 batch_size=32")
    if args.control_rank_weight < 0.0:
        raise ValueError("--control_rank_weight 不能为负数")
    if args.control_rank_margin < 0.0:
        raise ValueError("--control_rank_margin 不能为负数")
    if args.control_rank_scope == "routed_all" and (
        args.factor_surrogate_weight > 0.0
        or args.morphology_mask_surrogate_weight > 0.0
    ):
        raise ValueError(
            "routed_all control rank 不能与 factor/morphology surrogate 同时启用"
        )
    if args.factor_surrogate_weight < 0.0:
        raise ValueError("--factor_surrogate_weight 不能为负数")
    if args.state_population_delta_weight < 0.0:
        raise ValueError("--state_population_delta_weight 不能为负数")
    if args.state_prompt_causal_mid_weight < 0.0:
        raise ValueError("--state_prompt_causal_mid_weight 不能为负数")
    if args.state_prompt_causal_high_centroid_weight < 0.0:
        raise ValueError(
            "--state_prompt_causal_high_centroid_weight 不能为负数"
        )
    if args.state_trajectory_centroid_weight < 0.0:
        raise ValueError("--state_trajectory_centroid_weight 不能为负数")
    if args.state_trajectory_population_size < 0:
        raise ValueError("--state_trajectory_population_size 不能为负数")
    if args.state_trajectory_pool_size <= 0:
        raise ValueError("--state_trajectory_pool_size 必须为正数")
    if args.state_trajectory_timestep < 0:
        raise ValueError("--state_trajectory_timestep 不能为负数")
    if args.state_trajectory_noise_seed < 0:
        raise ValueError("--state_trajectory_noise_seed 不能为负数")
    if args.state_trajectory_centroid_weight > 0.0 and (
        args.state_trajectory_population_size <= 0
    ):
        raise ValueError(
            "非零轨迹中心权重要求 --state_trajectory_population_size > 0"
        )
    if args.state_trajectory_population_size > 0 and args.batch_size not in {
        2 * args.state_trajectory_population_size,
        3 * args.state_trajectory_population_size,
    }:
        raise ValueError(
            "轨迹中心组包的 --batch_size 必须等于2倍或3倍population size"
        )
    if args.state_early_ddim_centroid_weight < 0.0:
        raise ValueError("--state_early_ddim_centroid_weight 不能为负数")
    if args.state_early_ddim_population_size < 0:
        raise ValueError("--state_early_ddim_population_size 不能为负数")
    if args.state_early_ddim_pool_size <= 0:
        raise ValueError("--state_early_ddim_pool_size 必须为正数")
    if args.state_early_ddim_inference_steps != 25:
        raise ValueError("高噪声DDIM目标固定使用25步推理时间表")
    if args.state_early_ddim_trace_steps != 2:
        raise ValueError("高噪声DDIM目标固定只展开前两步")
    if args.state_early_ddim_subject_cfg != 2.0:
        raise ValueError("高噪声DDIM目标固定使用subject CFG=2")
    if args.state_early_ddim_noise_seed < 0:
        raise ValueError("--state_early_ddim_noise_seed 不能为负数")
    if args.state_early_ddim_probe_gradients and (
        args.state_early_ddim_centroid_weight != 1.0
        or args.lr != 0.0
        or args.precision != "bf16"
    ):
        raise ValueError(
            "高噪声DDIM梯度探针固定使用raw weight=1、lr=0和bf16"
        )
    if (
        args.state_early_ddim_centroid_weight > 0.0
        and args.state_early_ddim_population_size <= 0
    ):
        raise ValueError(
            "非零高噪声DDIM权重要求 --state_early_ddim_population_size > 0"
        )
    if (
        args.state_early_ddim_population_size > 0
        and args.batch_size not in {
            2 * args.state_early_ddim_population_size,
            3 * args.state_early_ddim_population_size,
        }
    ):
        raise ValueError(
            "高噪声DDIM组包的 --batch_size 必须等于2倍或3倍population size"
        )
    if (
        args.state_trajectory_population_size > 0
        and args.state_early_ddim_population_size > 0
    ):
        raise ValueError("旧轨迹中心与高噪声DDIM目标不能同时启用")
    if args.state_population_size <= 0:
        raise ValueError("--state_population_size 必须为正数")
    if (
        (
            args.state_population_delta_weight > 0.0
            or args.state_prompt_causal_mid_weight > 0.0
            or args.state_prompt_causal_high_centroid_weight > 0.0
        )
        and args.state_population_size <= 1
    ):
        raise ValueError("群体状态差异损失要求 --state_population_size > 1")
    if args.state_population_size > 1 and (
        args.batch_size != 2 * args.state_population_size
    ):
        raise ValueError(
            "群体状态训练要求 --batch_size 等于 2 × --state_population_size"
        )
    if args.state_population_pool_size <= 0:
        raise ValueError("--state_population_pool_size 必须为正数")
    if (
        args.state_population_timestep_min < 0
        or args.state_population_timestep_max
        < args.state_population_timestep_min
    ):
        raise ValueError("群体状态 timestep 区间无效")
    if args.state_prompt_causal_pool_size <= 0:
        raise ValueError("--state_prompt_causal_pool_size 必须为正数")
    if (
        args.state_prompt_causal_timestep_min < 0
        or args.state_prompt_causal_timestep_max
        < args.state_prompt_causal_timestep_min
    ):
        raise ValueError("状态文本因果 timestep 区间无效")
    if args.state_prompt_causal_high_pool_size <= 0:
        raise ValueError("--state_prompt_causal_high_pool_size 必须为正数")
    if (
        args.state_prompt_causal_high_timestep_min < 0
        or args.state_prompt_causal_high_timestep_max
        < args.state_prompt_causal_high_timestep_min
    ):
        raise ValueError("高尺度群体中心 timestep 区间无效")
    if args.state_population_size > 1 and args.factor_surrogate_state_population_size > 1:
        raise ValueError("两种 population batch sampler 不能同时启用")
    if args.state_trajectory_population_size > 0 and (
        args.state_population_size > 1
        or args.factor_surrogate_state_population_size > 1
    ):
        raise ValueError("v3.1轨迹中心组包不能与旧population sampler同时启用")
    if args.state_early_ddim_population_size > 0 and (
        args.state_population_size > 1
        or args.factor_surrogate_state_population_size > 1
    ):
        raise ValueError("v3.1高噪声DDIM组包不能与旧population sampler同时启用")
    if args.state_trajectory_centroid_weight > 0.0 and args.trainable_scope != "all":
        raise ValueError("轨迹中心训练必须更新完整三尺度主模型")
    if (
        args.state_early_ddim_population_size > 0
        and args.trainable_scope != "all"
    ):
        raise ValueError("高噪声DDIM训练必须保留完整三尺度普通扩散更新")
    if (
        args.state_population_delta_weight > 0.0
        and args.trainable_scope != "control_high_unet"
    ):
        raise ValueError(
            "群体状态差异 pilot 只允许 --trainable_scope control_high_unet"
        )
    if (
        args.state_prompt_causal_mid_weight > 0.0
        and args.trainable_scope != "control_mid_unet"
    ):
        raise ValueError(
            "中尺度状态文本因果 pilot 只允许 --trainable_scope control_mid_unet"
        )
    if (
        args.state_prompt_causal_high_centroid_weight > 0.0
        and args.trainable_scope != "control_high_unet"
    ):
        raise ValueError(
            "高尺度状态群体中心 pilot 只允许 --trainable_scope control_high_unet"
        )
    if args.morphology_mask_surrogate_weight < 0.0:
        raise ValueError("--morphology_mask_surrogate_weight 不能为负数")
    if args.morphology_pair_preservation_weight < 0.0:
        raise ValueError("--morphology_pair_preservation_weight 不能为负数")
    if args.factor_surrogate_state_population_size <= 0:
        raise ValueError("factor surrogate state population size 必须为正数")
    population_factor_to_id = {
        "subject": 0,
        "state_dnn": 1,
        "background": 2,
    }
    if (
        args.factor_surrogate_state_population_size > 1
        and args.factor_surrogate_fixed_factor
        != population_factor_to_id[args.factor_surrogate_population_factor]
    ):
        raise ValueError(
            "population surrogate factor and --factor_surrogate_fixed_factor "
            "do not match"
        )
    if (
        args.factor_surrogate_timestep_min < 0
        or args.factor_surrogate_timestep_max
        < args.factor_surrogate_timestep_min
    ):
        raise ValueError("factor surrogate timestep 区间无效")
    if args.factor_surrogate_weight > 0.0:
        required_surrogates = (
            args.factor_surrogate_subject_probe,
            args.factor_surrogate_dnn_state_probe,
            args.factor_surrogate_background_probe,
            args.factor_surrogate_dino_model_dir,
        )
        if any(not value for value in required_surrogates):
            raise ValueError(
                "启用 factor surrogate 时必须提供三项 probe 和 DINO 模型目录"
            )
        if (
            args.factor_surrogate_state_population_size > 1
            and args.factor_surrogate_population_factor == "state_dnn"
            and not args.factor_surrogate_dnn_state_dino_checkpoint
        ):
            raise ValueError(
                "群体 state surrogate 必须提供适配后的 DINO state checkpoint"
            )
        if (
            args.factor_surrogate_dnn_state_distribution_bank
            and args.factor_surrogate_state_population_size <= 1
        ):
            raise ValueError(
                "DNN state distribution bank requires population size > 1"
            )
        if (
            args.factor_surrogate_subject_distribution_bank
            and args.factor_surrogate_state_population_size <= 1
        ):
            raise ValueError(
                "subject distribution bank requires population size > 1"
            )
        if (
            args.factor_surrogate_subject_distribution_bank
            and not args.factor_surrogate_subject_image_probe
        ):
            raise ValueError(
                "subject distribution bank requires subject image probe"
            )
        if (
            args.factor_surrogate_subject_distribution_bank
            and args.factor_surrogate_population_factor != "subject"
        ):
            raise ValueError(
                "subject distribution bank requires population factor subject"
            )
    elif args.factor_surrogate_dnn_state_distribution_bank:
        raise ValueError(
            "DNN state distribution bank requires factor surrogate weight > 0"
        )
    elif args.factor_surrogate_subject_distribution_bank:
        raise ValueError(
            "subject distribution bank requires factor surrogate weight > 0"
        )
    if args.morphology_mask_surrogate_weight > 0.0:
        required_mask_surrogates = (
            args.morphology_mask_surrogate_bbbc021_checkpoint,
            args.morphology_mask_surrogate_dynamicnuclearnet_checkpoint,
        )
        if any(not value for value in required_mask_surrogates):
            raise ValueError(
                "启用 morphology mask surrogate 时必须提供两个来源 checkpoint"
            )
        if args.factor_surrogate_fixed_factor != 0:
            raise ValueError(
                "morphology mask surrogate 必须固定 subject factor 0"
            )
        if args.morphology_mask_surrogate_timestep_min < 0 or (
            args.morphology_mask_surrogate_timestep_max
            < args.morphology_mask_surrogate_timestep_min
        ):
            raise ValueError("morphology mask surrogate timestep 区间无效")
    if args.morphology_pair_preservation_weight > 0.0:
        if args.morphology_mask_surrogate_weight <= 0.0:
            raise ValueError(
                "morphology paired preservation requires mask geometry "
                "supervision"
            )
        if args.trainable_scope not in {
            "control_low_unet",
            "morphology_feature_affine",
            "morphology_spatial_affine",
        }:
            raise ValueError(
                "morphology paired preservation requires "
                "the full low subject generator or a morphology feature/"
                "spatial affine scope"
            )
    if args.counterfactual_pair_weight < 0.0:
        raise ValueError("--counterfactual_pair_weight 不能为负数")
    if args.counterfactual_pair_batch_size <= 0:
        raise ValueError("--counterfactual_pair_batch_size 必须大于 0")
    if args.counterfactual_teacher_anchor_mix < 0.0:
        raise ValueError("--counterfactual_teacher_anchor_mix 不能为负数")
    if args.counterfactual_target_orthogonal_mix < 0.0:
        raise ValueError(
            "--counterfactual_target_orthogonal_mix 不能为负数"
        )
    if (
        args.counterfactual_timestep_min < 0
        or args.counterfactual_timestep_max < args.counterfactual_timestep_min
    ):
        raise ValueError("反事实 timestep 区间无效")
    if args.counterfactual_pair_weight > 0.0:
        required_counterfactual = (
            args.counterfactual_subject_pairs,
            args.counterfactual_state_pairs,
            args.counterfactual_teacher_checkpoint,
        )
        if any(not value for value in required_counterfactual):
            raise ValueError(
                "启用反事实配对损失时必须提供 subject/state 配对清单和 "
                "teacher checkpoint"
            )
        if args.trainable_scope != "control_low_mid_unets":
            raise ValueError(
                "Stage-2a 反事实配对 pilot 必须使用 "
                "--trainable_scope control_low_mid_unets"
            )
        if args.control_rank_weight > 0.0:
            raise ValueError("反事实配对损失与旧 control rank 不应同时启用")
    if args.observation_rgb_pair_weight < 0.0:
        raise ValueError("--observation_rgb_pair_weight 不能为负数")
    if args.observation_rgb_pair_center_weight < 0.0:
        raise ValueError(
            "--observation_rgb_pair_center_weight 不能为负数"
        )
    if args.observation_rgb_pair_batch_size <= 0:
        raise ValueError(
            "--observation_rgb_pair_batch_size 必须大于 0"
        )
    if (
        args.observation_rgb_pair_timestep_min < 0
        or args.observation_rgb_pair_timestep_max
        < args.observation_rgb_pair_timestep_min
    ):
        raise ValueError("observation RGB pair timestep 区间无效")
    if args.observation_rgb_pair_weight > 0.0:
        if not args.observation_rgb_pairs:
            raise ValueError(
                "启用 observation RGB pair loss 时必须提供配对清单"
            )
        if args.trainable_scope != "rgb_observation_output_adapters":
            raise ValueError(
                "P17 observation配对训练必须只更新RGB observation adapters"
            )
        if args.conditioning_mode != RGB_FACTOR_OUTPUT_ADAPTER_MODE:
            raise ValueError(
                "P17 observation配对训练要求RGB factor output adapter模式"
            )
        if args.cond_drop_prob != 0.0:
            raise ValueError(
                "P17 observation配对训练冻结cond_drop_prob为0"
            )
    if args.demo and (
        args.control_rank_weight > 0.0
        or args.factor_surrogate_weight > 0.0
        or args.morphology_mask_surrogate_weight > 0.0
        or args.state_population_size > 1
        or args.state_trajectory_population_size > 0
        or args.state_early_ddim_population_size > 0
        or args.cascade_prior_gradient_mode != "off"
    ):
        raise ValueError("--demo 不提供反事实条件，不能启用控制损失")
    if args.demo and args.counterfactual_pair_weight > 0.0:
        raise ValueError("--demo 不提供真实反事实配对")
    if args.demo and args.observation_rgb_pair_weight > 0.0:
        raise ValueError("--demo 不提供同细胞双RGB观察配对")
    if args.train_limit_per_source < 0:
        raise ValueError("--train_limit_per_source 不能为负数")
    if not 0.0 < args.hard_sample_area_quantile < 1.0:
        raise ValueError("--hard_sample_area_quantile 必须位于 (0, 1)")
    if args.hard_sample_multiplier <= 1.0:
        raise ValueError("--hard_sample_multiplier 必须大于 1")
    if not 0.0 <= args.foreground_loss_mix < 1.0:
        raise ValueError("--foreground_loss_mix 必须位于 [0, 1)")
    if not 0.0 <= args.prior_proxy_mix <= 1.0:
        raise ValueError("--prior_proxy_mix 必须位于 [0, 1]")
    cascade_schedule_active = args.cascade_prior_schedule != "off"
    cascade_gradient_active = args.cascade_prior_gradient_mode != "off"
    if cascade_schedule_active != cascade_gradient_active:
        raise ValueError(
            "--cascade_prior_schedule 与 --cascade_prior_gradient_mode 必须同时启用或关闭"
        )
    if args.cascade_prior_mask_seed < 0:
        raise ValueError("--cascade_prior_mask_seed 不能为负数")
    if cascade_gradient_active and args.prior_proxy_mix != 0.0:
        raise ValueError("级联先验实验不能同时启用 --prior_proxy_mix")
    if args.repeat_first_train_batch and args.max_train_batches <= 0:
        raise ValueError("--repeat_first_train_batch 要求正数 --max_train_batches")
    if args.demo and args.foreground_loss_mix > 0:
        raise ValueError("--demo 不提供目标掩膜，不能启用 --foreground_loss_mix")
    if args.sample_num_steps is not None and args.sample_num_steps <= 0:
        raise ValueError("--sample_num_steps 必须大于 0")
    rank, local_rank, world_size = setup_distributed()
    device = get_device(local_rank)
    use_ddp = world_size > 1
    use_cuda = device.type == "cuda"

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed + rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    resume_state = load_checkpoint(args.resume, map_location="cpu") if args.resume else None
    if resume_state is not None:
        if resume_state.get("resume_eligible") is False:
            raise ValueError("partial-epoch emergency checkpoint is not safe to resume")
        if resume_state.get("selection_metric", "total") != args.checkpoint_selection_metric:
            raise ValueError("cannot change checkpoint selection metric when resuming")
        if resume_state.get("val_fixed_seed") != args.val_fixed_seed:
            raise ValueError("cannot change fixed validation seed when resuming")
    model_init_state = (
        load_checkpoint(args.model_init_checkpoint, map_location="cpu")
        if args.model_init_checkpoint
        else None
    )
    config_state = resume_state if resume_state is not None else model_init_state
    cfg = config_from_checkpoint(config_state) if config_state is not None else MSCDMConfig()
    if (
        args.resume
        and args.conditioning_mode is not None
        and args.conditioning_mode != cfg.conditioning_mode
    ):
        raise ValueError("恢复 optimizer 状态时不能更改 conditioning_mode")
    if args.conditioning_mode is not None:
        cfg.conditioning_mode = args.conditioning_mode
    if (
        cascade_gradient_active
        and cfg.conditioning_mode != SEMANTIC_STATE_ALLSCALE_MODE
    ):
        raise ValueError(
            "单步自预测级联先验训练必须使用 semantic_state_allscale"
        )
    if (
        (
            args.state_trajectory_centroid_weight > 0.0
            or args.state_early_ddim_population_size > 0
        )
        and cfg.conditioning_mode != SEMANTIC_STATE_ALLSCALE_MODE
    ):
        raise ValueError(
            "状态轨迹训练必须使用 --conditioning_mode semantic_state_allscale"
        )
    if args.state_early_ddim_probe_gradients and cfg.dropout != 0.0:
        raise ValueError("高噪声DDIM梯度探针要求模型dropout=0")
    if (
        args.trainable_scope == "nucleus_low_adapter"
        and cfg.conditioning_mode != NUCLEUS_ADAPTER_MODE
    ):
        raise ValueError(
            "--trainable_scope nucleus_low_adapter requires "
            "--conditioning_mode subject_state_low_nucleus_adapter"
        )
    if (
        args.trainable_scope == "nucleus_scale_adapters"
        and cfg.conditioning_mode != NUCLEUS_MULTISCALE_ADAPTER_MODE
    ):
        raise ValueError(
            "--trainable_scope nucleus_scale_adapters requires "
            "--conditioning_mode "
            "subject_state_low_nucleus_multiscale_adapter"
        )
    if (
        args.trainable_scope == "nucleus_prompt_adapters"
        and cfg.conditioning_mode != NUCLEUS_PROMPT_ADAPTER_MODE
    ):
        raise ValueError(
            "--trainable_scope nucleus_prompt_adapters requires "
            "--conditioning_mode subject_state_low_nucleus_prompt_adapters"
        )
    if (
        args.trainable_scope == "nucleus_distill_prompt_adapters"
        and cfg.conditioning_mode != NUCLEUS_DISTILL_PROMPT_ADAPTER_MODE
    ):
        raise ValueError(
            "--trainable_scope nucleus_distill_prompt_adapters requires "
            "--conditioning_mode hierarchical_nucleus_distill_prompt_adapters"
        )
    if (
        args.trainable_scope == "nucleus_distill_output_adapters"
        and cfg.conditioning_mode != NUCLEUS_DISTILL_OUTPUT_ADAPTER_MODE
    ):
        raise ValueError(
            "--trainable_scope nucleus_distill_output_adapters requires "
            "--conditioning_mode hierarchical_nucleus_distill_output_adapters"
        )
    if (
        args.trainable_scope == "factor_prompt_adapters"
        and cfg.conditioning_mode != FACTOR_PROMPT_ADAPTER_MODE
    ):
        raise ValueError(
            "--trainable_scope factor_prompt_adapters requires "
            "--conditioning_mode hierarchical_factor_prompt_adapters"
        )
    if (
        args.trainable_scope == "factor_delta_adapters"
        and cfg.conditioning_mode != FACTOR_DELTA_ADAPTER_MODE
    ):
        raise ValueError(
            "--trainable_scope factor_delta_adapters requires "
            "--conditioning_mode hierarchical_factor_delta_adapters"
        )
    if (
        args.trainable_scope == "factor_attention_delta_adapters"
        and cfg.conditioning_mode != FACTOR_ATTENTION_DELTA_ADAPTER_MODE
    ):
        raise ValueError(
            "--trainable_scope factor_attention_delta_adapters requires "
            "--conditioning_mode "
            "hierarchical_factor_attention_delta_adapters"
        )
    if (
        args.trainable_scope in {
            "rgb_factor_output_adapters",
            "rgb_observation_output_adapters",
        }
        and cfg.conditioning_mode != RGB_FACTOR_OUTPUT_ADAPTER_MODE
    ):
        raise ValueError(
            "RGB output adapter trainable scopes require "
            "--conditioning_mode hierarchical_rgb_factor_output_adapters"
        )
    if (
        args.trainable_scope == "spatial_subject_input_adapter"
        and cfg.conditioning_mode != SPATIAL_SUBJECT_INPUT_ADAPTER_MODE
    ):
        raise ValueError(
            "--trainable_scope spatial_subject_input_adapter requires "
            "--conditioning_mode hierarchical_spatial_subject_input_adapter"
        )
    if (
        args.trainable_scope == "morphology_latent_bank"
        and cfg.conditioning_mode != MORPHOLOGY_LATENT_BANK_MODE
    ):
        raise ValueError(
            "--trainable_scope morphology_latent_bank requires "
            "--conditioning_mode hierarchical_morphology_latent_bank"
        )
    if (
        args.trainable_scope == "morphology_feature_affine"
        and cfg.conditioning_mode != MORPHOLOGY_FEATURE_AFFINE_MODE
    ):
        raise ValueError(
            "--trainable_scope morphology_feature_affine requires "
            "--conditioning_mode hierarchical_morphology_feature_affine"
        )
    if (
        args.trainable_scope == "morphology_spatial_affine"
        and cfg.conditioning_mode != MORPHOLOGY_SPATIAL_AFFINE_MODE
    ):
        raise ValueError(
            "--trainable_scope morphology_spatial_affine requires "
            "--conditioning_mode hierarchical_morphology_spatial_affine"
        )
    if (
        config_state is not None
        and args.prediction_type is not None
        and args.prediction_type != cfg.prediction_type
    ):
        raise ValueError(
            "完整模型 checkpoint 初始化不能更改 prediction_type"
        )
    if args.prediction_type is not None:
        cfg.prediction_type = args.prediction_type
    if args.contrastive_weight is not None:
        if args.contrastive_weight < 0.0:
            raise ValueError("--contrastive_weight must be nonnegative")
        if (
            args.resume
            and not math.isclose(
                args.contrastive_weight, cfg.contrastive_weight,
                rel_tol=0.0, abs_tol=1e-12,
            )
        ):
            raise ValueError("cannot change contrastive weight when resuming optimizer state")
        cfg.contrastive_weight = args.contrastive_weight
    calibration_payload = (
        apply_latent_calibration(cfg, args.latent_calibration)
        if args.latent_calibration
        else None
    )
    if args.resume and args.freeze_vae and not cfg.freeze_vae:
        raise ValueError("不能在恢复旧优化器状态时临时冻结 VAE；请使用 --vae_init_checkpoint 开新实验")
    if args.freeze_vae:
        cfg.freeze_vae = True
    cfg.text_pretrained_name = args.clip_model_dir
    cfg.text_local_files_only = True
    cfg.freeze_clip_backbone = True
    if args.no_vae_gradient_checkpointing:
        cfg.vae_gradient_checkpointing = False
    if args.counterfactual_timestep_max >= cfg.num_train_timesteps:
        raise ValueError(
            "--counterfactual_timestep_max 超出扩散训练时间步范围"
        )
    if (
        args.observation_rgb_pair_timestep_max
        >= cfg.num_train_timesteps
    ):
        raise ValueError(
            "--observation_rgb_pair_timestep_max 超出扩散训练时间步范围"
        )
    if args.factor_surrogate_timestep_max >= cfg.num_train_timesteps:
        raise ValueError(
            "--factor_surrogate_timestep_max 超出扩散训练时间步范围"
        )
    if args.state_population_timestep_max >= cfg.num_train_timesteps:
        raise ValueError(
            "--state_population_timestep_max 超出扩散训练时间步范围"
        )
    if args.state_trajectory_timestep >= cfg.num_train_timesteps:
        raise ValueError(
            "--state_trajectory_timestep 超出扩散训练时间步范围"
        )
    if (
        args.morphology_mask_surrogate_timestep_max
        >= cfg.num_train_timesteps
    ):
        raise ValueError(
            "--morphology_mask_surrogate_timestep_max 超出扩散训练时间步范围"
        )
    if args.tiny:
        cfg.base_channels = (8, 8, 8)
        cfg.channel_mult = (1, 2)
        cfg.num_res_blocks = 1
        cfg.num_heads = 1
        cfg.vae_base_ch = 8
        cfg.vae_ch_mult = (1, 1, 2, 2)
        cfg.use_reversible = False
        cfg.vae_gradient_checkpointing = False
        cfg.num_inference_steps = 2

    model = MSCDM(cfg)
    if args.vae_init_checkpoint:
        load_vae_initialization(model, args.vae_init_checkpoint)
        if is_main_process(rank):
            print("loaded VAE initialization <-", Path(args.vae_init_checkpoint).resolve())
    if model_init_state is not None:
        if not isinstance(model_init_state, dict) or "model" not in model_init_state:
            raise ValueError("--model_init_checkpoint 必须指向含 model 的 checkpoint")
        if cfg.conditioning_mode in {
            NUCLEUS_ADAPTER_MODE,
            NUCLEUS_MULTISCALE_ADAPTER_MODE,
            NUCLEUS_PROMPT_ADAPTER_MODE,
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
            SEMANTIC_STATE_FACTOR_RESIDUAL_POOL_MODE,
            SEMANTIC_STATE_FACTOR_PARTITION_ATTENTION_MODE,
            SEMANTIC_STATE_FACTOR_ISOLATED_COMBINED_MODE,
            SEMANTIC_STATE_SPECIFIC_RESIDUAL_MODE,
        }:
            incompatible = model.load_state_dict(
                model_init_state["model"],
                strict=False,
            )
            residual_expected_missing = {
                "factor_residual_pool.projections.low_subject.weight",
                "factor_residual_pool.projections.low_state.weight",
                "factor_residual_pool.projections.low_background.weight",
                "factor_residual_pool.projections.mid_state.weight",
                "factor_residual_pool.projections.mid_background.weight",
                "factor_residual_pool.projections.high_state.weight",
                "factor_residual_pool.projections.high_background.weight",
            }
            partition_expected_missing = {
                key
                for key in model.state_dict()
                if key.endswith(".factor_partition_gate")
            }
            state_specific_expected_missing = {
                key
                for key in model.state_dict()
                if key.startswith("text_enc.state_specific_residual.")
            }
            if cfg.conditioning_mode == SEMANTIC_STATE_FACTOR_RESIDUAL_POOL_MODE:
                expected_missing = residual_expected_missing
                invalid_missing = sorted(
                    set(incompatible.missing_keys) ^ expected_missing
                )
            elif (
                cfg.conditioning_mode
                == SEMANTIC_STATE_FACTOR_PARTITION_ATTENTION_MODE
            ):
                expected_missing = partition_expected_missing
                invalid_missing = sorted(
                    set(incompatible.missing_keys) ^ expected_missing
                )
            elif (
                cfg.conditioning_mode
                == SEMANTIC_STATE_FACTOR_ISOLATED_COMBINED_MODE
            ):
                expected_missing = (
                    residual_expected_missing | partition_expected_missing
                )
                invalid_missing = sorted(
                    set(incompatible.missing_keys) ^ expected_missing
                )
            elif (
                cfg.conditioning_mode
                == SEMANTIC_STATE_SPECIFIC_RESIDUAL_MODE
            ):
                expected_missing = state_specific_expected_missing
                invalid_missing = sorted(
                    set(incompatible.missing_keys) ^ expected_missing
                )
            else:
                invalid_missing = [
                    key
                    for key in incompatible.missing_keys
                    if not key.startswith(
                        (
                            "nucleus_low_adapter.",
                            "nucleus_mid_adapter.",
                            "nucleus_high_adapter.",
                            "nucleus_prompt_adapters.",
                            "nucleus_distill_prompt_adapters.",
                            "factor_prompt_adapters.",
                            "factor_delta_adapters.",
                            "rgb_factor_output_adapters.",
                            "spatial_subject_input_adapter.",
                            "morphology_latent_bank.",
                            "unet_low.down_morphology_affines.",
                            "unet_low.mid_morphology_affines.",
                            "unet_low.up_morphology_affines.",
                            "unet_low.down_morphology_spatial_affines.",
                            "unet_low.mid_morphology_spatial_affines.",
                            "unet_low.up_morphology_spatial_affines.",
                        )
                    )
                    and ".factor_delta_" not in key
                ]
            if invalid_missing or incompatible.unexpected_keys:
                raise ValueError(
                    "adapter 初始化 checkpoint 结构不匹配："
                    f"missing={invalid_missing[:5]} "
                    f"unexpected={incompatible.unexpected_keys[:5]}"
                )
            if cfg.conditioning_mode in {
                SEMANTIC_STATE_FACTOR_RESIDUAL_POOL_MODE,
                SEMANTIC_STATE_FACTOR_ISOLATED_COMBINED_MODE,
            }:
                residual_state = model.factor_residual_pool.state_dict()
                if set(residual_state) != {
                    key.removeprefix("factor_residual_pool.")
                    for key in residual_expected_missing
                } or any(
                    bool(torch.count_nonzero(value).item())
                    for value in residual_state.values()
                ):
                    raise ValueError(
                        "factor residual pooled initialization must contain "
                        "exactly seven all-zero projection tensors"
                    )
                if (
                    is_main_process(rank)
                    and cfg.conditioning_mode
                    == SEMANTIC_STATE_FACTOR_RESIDUAL_POOL_MODE
                ):
                    print(
                        "loaded function-preserving factor residual pooled "
                        "initialization: 7/7 projection tensors exactly zero"
                    )
            if (
                cfg.conditioning_mode
                in {
                    SEMANTIC_STATE_FACTOR_PARTITION_ATTENTION_MODE,
                    SEMANTIC_STATE_FACTOR_ISOLATED_COMBINED_MODE,
                }
            ):
                gates = {
                    key: value
                    for key, value in model.state_dict().items()
                    if key.endswith(".factor_partition_gate")
                }
                if len(gates) != 35 or any(
                    value.numel() != 1
                    or bool(torch.count_nonzero(value).item())
                    for value in gates.values()
                ):
                    raise ValueError(
                        "factor-partitioned attention initialization must contain "
                        "exactly 35 all-zero scalar gates"
                    )
                if (
                    is_main_process(rank)
                    and cfg.conditioning_mode
                    == SEMANTIC_STATE_FACTOR_PARTITION_ATTENTION_MODE
                ):
                    print(
                        "loaded function-preserving factor-partitioned attention "
                        "initialization: 35/35 scalar gates exactly zero"
                    )
            if (
                is_main_process(rank)
                and cfg.conditioning_mode
                == SEMANTIC_STATE_FACTOR_ISOLATED_COMBINED_MODE
            ):
                print(
                    "loaded function-preserving combined factor-isolated "
                    "initialization: 7/7 projection tensors and 35/35 scalar "
                    "gates exactly zero"
                )
            if cfg.conditioning_mode == SEMANTIC_STATE_SPECIFIC_RESIDUAL_MODE:
                state_specific = model.text_enc.state_specific_residual
                if (
                    state_specific is None
                    or len(state_specific_expected_missing) != 8
                    or not state_specific.has_exact_initial_state()
                ):
                    raise ValueError(
                        "state-specific residual initialization must contain "
                        "exactly eight deterministic identity/zero tensors"
                    )
                if is_main_process(rank):
                    print(
                        "loaded function-preserving state-specific residual "
                        "initialization: 8/8 tensors in exact deterministic state"
                    )
        else:
            model.load_state_dict(model_init_state["model"])
        if is_main_process(rank):
            print(
                "loaded full model initialization <-",
                Path(args.model_init_checkpoint).resolve(),
            )
    if args.p10_export_initial:
        if world_size != 1 or args.resume or args.model_init_checkpoint:
            raise ValueError("P10 step-0 export requires one process and VAE-only initialization")
        if not args.vae_init_checkpoint or not cfg.freeze_vae:
            raise ValueError("P10 step-0 export requires a frozen VAE checkpoint")
        path = Path(args.p10_export_initial).resolve()
        temporary = path.with_name(path.name + ".tmp")
        if path.exists() or temporary.exists():
            raise FileExistsError(path if path.exists() else temporary)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 2,
                "model": model.state_dict(),
                "cfg": asdict(cfg),
                "epoch": -1,
                "global_step": 0,
                "best_val": float("inf"),
                "initialization_only": True,
            },
            temporary,
        )
        temporary.replace(path)
        write_json(
            path.with_name(path.stem + "_meta.json"),
            {
                "status": "step0_exported_no_training",
                "checkpoint": str(path),
                "sha256": sha256_file(path),
                "seed": args.seed,
                "vae_init_checkpoint": str(Path(args.vae_init_checkpoint).resolve()),
                "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            },
        )
        cleanup_distributed()
        return
    configure_control_unet_trainable_scope(model, args.trainable_scope)
    probe_reversible_block_count = 0
    if args.state_early_ddim_probe_gradients:
        # The memory-saving custom autograd Function materializes reversible
        # block parameter gradients only inside its backward.  The calibration
        # probe needs torch.autograd.grad over the complete low U-Net parameter
        # set, so use the mathematically identical ordinary autograd path for
        # this lr=0 measurement only.  Formal training keeps save_memory=True.
        for module in model.unet_low.modules():
            if isinstance(module, ReversibleCondBlock):
                module.save_memory = False
                probe_reversible_block_count += 1
        if probe_reversible_block_count <= 0:
            raise ValueError(
                "高噪声DDIM梯度探针没有找到low分支可逆块"
            )
    if args.trainable_scope in {
        "nucleus_low_adapter",
        "nucleus_scale_adapters",
        "nucleus_prompt_adapters",
        "nucleus_distill_prompt_adapters",
        "nucleus_distill_output_adapters",
        "factor_prompt_adapters",
        "factor_delta_adapters",
        "factor_attention_delta_adapters",
        "rgb_factor_output_adapters",
        "rgb_observation_output_adapters",
        "spatial_subject_input_adapter",
        "morphology_latent_bank",
        "morphology_feature_affine",
        "morphology_spatial_affine",
    }:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if args.trainable_scope == "nucleus_low_adapter":
            adapter_names = ("nucleus_low_adapter",)
        elif args.trainable_scope in {
            "nucleus_scale_adapters",
            "nucleus_distill_output_adapters",
        }:
            adapter_names = (
                "nucleus_low_adapter",
                "nucleus_mid_adapter",
                "nucleus_high_adapter",
            )
        elif args.trainable_scope == "nucleus_prompt_adapters":
            adapter_names = ("nucleus_prompt_adapters",)
        elif args.trainable_scope == "factor_prompt_adapters":
            adapter_names = ("factor_prompt_adapters",)
        elif args.trainable_scope == "factor_delta_adapters":
            adapter_names = ("factor_delta_adapters",)
        elif args.trainable_scope == "factor_attention_delta_adapters":
            adapter_names = ()
        elif args.trainable_scope == "rgb_factor_output_adapters":
            adapter_names = ("rgb_factor_output_adapters",)
        elif args.trainable_scope == "rgb_observation_output_adapters":
            adapter_names = ()
        elif args.trainable_scope == "spatial_subject_input_adapter":
            adapter_names = ("spatial_subject_input_adapter",)
        elif args.trainable_scope == "morphology_latent_bank":
            adapter_names = ("morphology_latent_bank",)
        elif args.trainable_scope == "morphology_feature_affine":
            adapter_names = ()
        elif args.trainable_scope == "morphology_spatial_affine":
            adapter_names = ()
        else:
            adapter_names = ("nucleus_distill_prompt_adapters",)
        if args.trainable_scope == "factor_attention_delta_adapters":
            attention_parameters = [
                parameter
                for name, parameter in model.named_parameters()
                if ".factor_delta_" in name
            ]
            if not attention_parameters:
                raise ValueError(
                    "factor attention delta adapters are not initialized"
                )
            for parameter in attention_parameters:
                parameter.requires_grad_(True)
        if args.trainable_scope == "morphology_feature_affine":
            morphology_parameters = [
                parameter
                for name, parameter in model.named_parameters()
                if name.startswith(
                    (
                        "unet_low.down_morphology_affines.",
                        "unet_low.mid_morphology_affines.",
                        "unet_low.up_morphology_affines.",
                    )
                )
            ]
            if not morphology_parameters:
                raise ValueError(
                    "morphology feature affine decoder was not initialized"
                )
            for parameter in morphology_parameters:
                parameter.requires_grad_(True)
        if args.trainable_scope == "rgb_observation_output_adapters":
            observation_parameters = [
                parameter
                for name, parameter in model.named_parameters()
                if name.startswith("rgb_factor_output_adapters.")
                and "_observation." in name
            ]
            if not observation_parameters:
                raise ValueError(
                    "RGB observation output adapters are not initialized"
                )
            for parameter in observation_parameters:
                parameter.requires_grad_(True)
        if args.trainable_scope == "morphology_spatial_affine":
            morphology_parameters = [
                parameter
                for name, parameter in model.named_parameters()
                if name.startswith(
                    (
                        "unet_low.down_morphology_spatial_affines.",
                        "unet_low.mid_morphology_spatial_affines.",
                        "unet_low.up_morphology_spatial_affines.",
                    )
                )
            ]
            if not morphology_parameters:
                raise ValueError(
                    "morphology spatial affine decoder was not initialized"
                )
            for parameter in morphology_parameters:
                parameter.requires_grad_(True)
        for adapter_name in adapter_names:
            adapter = getattr(model, adapter_name)
            if adapter is None:
                raise ValueError(f"{adapter_name} is not initialized")
            for parameter in adapter.parameters():
                parameter.requires_grad_(True)
    model = model.to(device)
    if resume_state is not None:
        if "model" not in resume_state:
            raise ValueError("--resume 必须指向含 model/optimizer/scaler 的完整训练 checkpoint")
        model.load_state_dict(resume_state["model"])
    counterfactual_teacher_unets = None
    if args.counterfactual_pair_weight > 0.0:
        counterfactual_teacher_unets = (
            build_frozen_counterfactual_teacher_unets(
                model,
                args.counterfactual_teacher_checkpoint,
                device,
            )
        )
    factor_surrogate_bundle = None
    if args.factor_surrogate_weight > 0.0:
        from mscdms.factor_surrogate import FactorSurrogateBundle

        factor_surrogate_bundle = FactorSurrogateBundle(
            subject_probe=args.factor_surrogate_subject_probe,
            dnn_state_probe=args.factor_surrogate_dnn_state_probe,
            background_probe=args.factor_surrogate_background_probe,
            dino_model_dir=args.factor_surrogate_dino_model_dir,
            dnn_state_dino_checkpoint=(
                args.factor_surrogate_dnn_state_dino_checkpoint
            ),
            dnn_state_distribution_bank=(
                args.factor_surrogate_dnn_state_distribution_bank
            ),
            subject_image_probe=(
                args.factor_surrogate_subject_image_probe
            ),
            subject_distribution_bank=(
                args.factor_surrogate_subject_distribution_bank
            ),
        ).to(device)
        factor_surrogate_bundle.eval()
        if is_main_process(rank):
            print(
                "factor surrogate audit:",
                json.dumps(
                    factor_surrogate_bundle.audit(),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
    morphology_mask_surrogate_bundle = None
    if args.morphology_mask_surrogate_weight > 0.0:
        from mscdms.mask_surrogate import FrozenSourceMaskSurrogates

        morphology_mask_surrogate_bundle = FrozenSourceMaskSurrogates(
            bbbc021_checkpoint=(
                args.morphology_mask_surrogate_bbbc021_checkpoint
            ),
            dynamicnuclearnet_checkpoint=(
                args.morphology_mask_surrogate_dynamicnuclearnet_checkpoint
            ),
        ).to(device)
        morphology_mask_surrogate_bundle.eval()
        if is_main_process(rank):
            print(
                "morphology mask surrogate audit:",
                json.dumps(
                    morphology_mask_surrogate_bundle.audit(),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
    # The gradient calibration probe calls torch.autograd.grad twice on the
    # same parameters and then performs the ordinary backward pass.  DDP's
    # reducer hooks are incompatible with that diagnostic pattern.  The
    # probe uses lr=0, so every rank can keep an identical local model while
    # the requested gradient sufficient statistics are reduced explicitly.
    model_is_ddp = use_ddp and not args.state_early_ddim_probe_gradients
    if model_is_ddp:
        # P17 executes all three trainable observation adapters on every
        # iteration.  Avoid the extra autograd traversal and misleading DDP
        # warning for this fully used, adapter-only graph; retain the historic
        # conservative setting for heterogeneous legacy training routes.
        find_unused_parameters = (
            args.trainable_scope != "rgb_observation_output_adapters"
        )
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=find_unused_parameters,
        )
    raw_model = model.module if model_is_ddp else model

    optimizer = torch.optim.AdamW(raw_model.get_optimizer_param_groups(args.lr, args.wd))
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    amp_enabled = use_cuda and args.precision != "fp32"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and args.precision == "fp16")
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    if resume_state is not None:
        if "optimizer" not in resume_state or "scaler" not in resume_state:
            raise ValueError("该 checkpoint 只含推理权重，不能断点续训")
        optimizer.load_state_dict(resume_state["optimizer"])
        scaler.load_state_dict(resume_state["scaler"])

    if args.demo:
        train_dataset: Dataset = DemoTensorDataset(
            64,
            return_object_type=cfg.conditioning_mode in NUCLEUS_GATED_MODES,
        )
        val_dataset: Dataset | None = None
        total_records = len(train_dataset)
    else:
        root = Path(args.data_dir)
        total_records = discover_dataset_size(root, args.manifest_name)
        if total_records == 0:
            raise FileNotFoundError(f"{root} 下没有可训练样本")
        train_dataset = CellMicroscopyDataset(
            root,
            split="train",
            seed=args.seed,
            normalize_mode=args.normalize,
            manifest_name=args.manifest_name,
            return_target_mask=(
                args.foreground_loss_mix > 0
                or args.morphology_pair_preservation_weight > 0.0
                or cfg.conditioning_mode in MORPHOLOGY_SPATIAL_MASK_MODES
            ),
            return_morphology_class=(
                cfg.conditioning_mode in MORPHOLOGY_STRUCTURED_MODES
            ),
            return_object_type=cfg.conditioning_mode in NUCLEUS_GATED_MODES,
            return_control_negatives=(
                args.control_rank_weight > 0.0
                or args.factor_surrogate_weight > 0.0
                or args.morphology_mask_surrogate_weight > 0.0
            ),
            return_morphology_mask_target=(
                args.morphology_mask_surrogate_weight > 0.0
            ),
            return_morphology_counterfactual_class=(
                args.morphology_mask_surrogate_weight > 0.0
                and cfg.conditioning_mode
                == MORPHOLOGY_FEATURE_AFFINE_MODE
            ),
        )
        candidate_val = CellMicroscopyDataset(
            root,
            split="val",
            seed=args.seed,
            normalize_mode=args.normalize,
            manifest_name=args.manifest_name,
            return_target_mask=(
                args.foreground_loss_mix > 0
                or cfg.conditioning_mode in MORPHOLOGY_SPATIAL_MASK_MODES
            ),
            return_morphology_class=(
                cfg.conditioning_mode in MORPHOLOGY_STRUCTURED_MODES
            ),
            return_object_type=cfg.conditioning_mode in NUCLEUS_GATED_MODES,
            return_control_negatives=False,
            return_morphology_mask_target=False,
        )
        filtered_train_count = filter_manifest_dataset_object_type(
            train_dataset,
            args.dataset_object_type,
        )
        filtered_val_count = filter_manifest_dataset_object_type(
            candidate_val,
            args.dataset_object_type,
        )
        if args.dataset_object_type is not None and is_main_process(rank):
            print(
                f"object-type filter={args.dataset_object_type} "
                f"train={filtered_train_count} val={filtered_val_count}"
            )
        val_dataset = candidate_val if len(candidate_val) > 0 else None
        limited_counts = limit_manifest_dataset_per_source(
            train_dataset,
            args.train_limit_per_source,
            args.seed,
        )
        if limited_counts and is_main_process(rank):
            print("overfit-gate train source counts:", limited_counts)
    if len(train_dataset) == 0:
        raise ValueError("train split 为空")
    factor_surrogate_lookup: dict[str, dict] = {}
    if args.factor_surrogate_weight > 0.0:
        factor_surrogate_lookup = build_factor_surrogate_record_lookup(
            train_dataset
        )
    state_population_lookup: dict[str, tuple[int, str, str]] = {}
    if args.state_population_size > 1:
        state_population_lookup = build_state_population_record_lookup(
            train_dataset
        )
    grouped_state_population_size = (
        args.state_trajectory_population_size
        if args.state_trajectory_population_size > 0
        else args.state_early_ddim_population_size
    )
    trajectory_state_lookup: dict[str, tuple[int, str, str, str]] = {}
    if grouped_state_population_size > 0:
        trajectory_state_lookup = build_v31_multiclass_state_record_lookup(
            train_dataset
        )
    if is_main_process(rank):
        print(
            f"dataset total={total_records} train={len(train_dataset)} "
            f"val={len(val_dataset) if val_dataset is not None else 0}"
        )
        frozen_clip = all(not p.requires_grad for p in raw_model.text_enc.clip.parameters())
        print(f"CLIP frozen={frozen_clip}, local={raw_model.text_enc.pretrained_name}")
        print(
            f"VAE frozen={cfg.freeze_vae}, posterior_mean={cfg.frozen_vae_use_mean}, "
            f"latent_normalization={cfg.latent_normalization}, "
            f"prediction_type={cfg.prediction_type}"
        )
        if calibration_payload is not None:
            print(
                "latent calibration selected_n=",
                calibration_payload.get("selected_n", "unknown"),
            )

    hard_sampling_audit = None
    if args.train_sampling_strategy == "small_area_source_v1":
        if not isinstance(train_dataset, CellMicroscopyDataset) or not train_dataset.uses_manifest:
            raise ValueError("small_area_source_v1 只支持 manifest 数据集")
        train_sampler = SourceBalancedSmallAreaSampler(
            train_dataset.records,
            target_source=args.hard_sample_source,
            area_quantile=args.hard_sample_area_quantile,
            hard_multiplier=args.hard_sample_multiplier,
            num_replicas=world_size,
            rank=rank,
            seed=args.seed,
        )
        hard_sampling_audit = train_sampler.audit(epoch=0)
        if is_main_process(rank):
            print(
                "hard sampling audit:",
                json.dumps(hard_sampling_audit, ensure_ascii=False, sort_keys=True),
            )
    elif args.train_sampling_strategy == "source_morphology_balanced_v1":
        if not isinstance(train_dataset, CellMicroscopyDataset) or not train_dataset.uses_manifest:
            raise ValueError(
                "source_morphology_balanced_v1 only supports manifest data"
            )
        train_sampler = SourceMorphologyBalancedSampler(
            train_dataset.records,
            num_replicas=world_size,
            rank=rank,
            seed=args.seed,
        )
        hard_sampling_audit = train_sampler.audit(epoch=0)
        if is_main_process(rank):
            print(
                "morphology sampling audit:",
                json.dumps(hard_sampling_audit, ensure_ascii=False, sort_keys=True),
            )
    elif args.train_sampling_strategy == "support_cell_balanced_v1":
        if not isinstance(train_dataset, CellMicroscopyDataset) or not train_dataset.uses_manifest:
            raise ValueError(
                "support_cell_balanced_v1 only supports manifest data"
            )
        train_sampler = SupportCellBalancedSampler(
            train_dataset.records,
            num_replicas=world_size,
            rank=rank,
            seed=args.seed,
        )
        hard_sampling_audit = train_sampler.audit(epoch=0)
        if is_main_process(rank):
            print(
                "support-cell sampling audit:",
                json.dumps(hard_sampling_audit, ensure_ascii=False, sort_keys=True),
            )
    else:
        train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
            )
            if use_ddp
            else None
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        drop_last=len(train_dataset) >= args.batch_size,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
    )
    state_population_sampler = None
    state_population_loader = None
    if args.factor_surrogate_state_population_size > 1:
        if not isinstance(train_dataset, CellMicroscopyDataset):
            raise ValueError("population state batches require manifest data")
        state_population_sampler = FactorPopulationBatchSampler(
            train_dataset.records,
            factor=args.factor_surrogate_population_factor,
            population_size=args.factor_surrogate_state_population_size,
            num_replicas=world_size,
            rank=rank,
            seed=args.seed,
        )
        state_population_loader = DataLoader(
            train_dataset,
            batch_sampler=state_population_sampler,
            num_workers=args.num_workers,
            pin_memory=use_cuda,
            persistent_workers=args.num_workers > 0,
        )
        if is_main_process(rank):
            print(
                "state population batches:",
                json.dumps(
                    {
                        "population_size": (
                            args.factor_surrogate_state_population_size
                        ),
                        "groups_per_rank": len(state_population_sampler),
                        "target_classes": len(
                            state_population_sampler.by_target
                        ),
                    },
                    sort_keys=True,
                ),
            )
    real_state_population_sampler = None
    real_state_population_loader = None
    if args.state_population_size > 1:
        if not isinstance(train_dataset, CellMicroscopyDataset):
            raise ValueError("real state population batches require manifest data")
        real_state_population_sampler = StateContrastPopulationBatchSampler(
            train_dataset.records,
            population_size=args.state_population_size,
            num_replicas=world_size,
            rank=rank,
            seed=args.seed,
        )
        real_state_population_loader = DataLoader(
            train_dataset,
            batch_sampler=real_state_population_sampler,
            num_workers=args.num_workers,
            pin_memory=use_cuda,
            persistent_workers=args.num_workers > 0,
        )
        if is_main_process(rank):
            print(
                "real state population batches:",
                json.dumps(
                    {
                        "population_size_per_state": args.state_population_size,
                        "batch_size": 2 * args.state_population_size,
                        "groups_per_rank": len(real_state_population_sampler),
                        "subject_background_conditions": len(
                            real_state_population_sampler.by_condition
                        ),
                    },
                    sort_keys=True,
                ),
            )
    trajectory_state_sampler = None
    trajectory_state_loader = None
    if grouped_state_population_size > 0:
        if not isinstance(train_dataset, CellMicroscopyDataset):
            raise ValueError("v3.1 trajectory batches require manifest data")
        trajectory_state_sampler = V31MulticlassStatePopulationBatchSampler(
            train_dataset.records,
            population_size=grouped_state_population_size,
            num_replicas=world_size,
            rank=rank,
            seed=args.seed,
        )
        trajectory_state_loader = DataLoader(
            train_dataset,
            batch_sampler=trajectory_state_sampler,
            num_workers=args.num_workers,
            pin_memory=use_cuda,
            persistent_workers=args.num_workers > 0,
        )
        if is_main_process(rank):
            print(
                "v3.1 trajectory state batches:",
                json.dumps(
                    {
                        "population_size_per_state": (
                            grouped_state_population_size
                        ),
                        "groups_per_rank": len(trajectory_state_sampler),
                        "groups_per_rank_by_class_count": (
                            trajectory_state_sampler.groups_per_replica_by_class_count
                        ),
                        "groups_per_condition": (
                            trajectory_state_sampler.groups_per_condition
                        ),
                        "source_subject_background_conditions": len(
                            trajectory_state_sampler.by_condition
                        ),
                    },
                    sort_keys=True,
                ),
            )
    if (
        args.train_sampling_strategy == "source_morphology_balanced_v1"
        and val_dataset is not None
    ):
        if not isinstance(val_dataset, CellMicroscopyDataset):
            raise ValueError("balanced morphology validation requires manifest data")
        val_sampler = SourceMorphologyBalancedSampler(
            val_dataset.records,
            num_replicas=world_size,
            rank=rank,
            seed=args.seed + 1_000_000,
        )
    else:
        val_sampler = (
            DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
            )
            if use_ddp and val_dataset is not None
            else None
        )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=use_cuda,
            persistent_workers=args.num_workers > 0,
        )
        if val_dataset is not None
        else None
    )
    counterfactual_pair_loaders: dict[int, DataLoader] = {}
    counterfactual_pair_samplers: dict[int, DistributedSampler] = {}
    if args.counterfactual_pair_weight > 0.0:
        pair_specs = (
            (0, "subject", args.counterfactual_subject_pairs),
            (1, "state", args.counterfactual_state_pairs),
        )
        for factor_id, factor_name, manifest_name in pair_specs:
            pair_dataset = CounterfactualPairDataset(
                args.data_dir,
                str(manifest_name),
                expected_factor=factor_name,
                normalize_mode=args.normalize,
            )
            pair_sampler = (
                DistributedSampler(
                    pair_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    seed=args.seed + 100 + factor_id,
                )
                if use_ddp
                else None
            )
            if pair_sampler is not None:
                counterfactual_pair_samplers[factor_id] = pair_sampler
            counterfactual_pair_loaders[factor_id] = DataLoader(
                pair_dataset,
                batch_size=args.counterfactual_pair_batch_size,
                shuffle=pair_sampler is None,
                sampler=pair_sampler,
                num_workers=args.num_workers,
                drop_last=True,
                pin_memory=use_cuda,
                persistent_workers=args.num_workers > 0,
            )
            if is_main_process(rank):
                print(
                    f"counterfactual {factor_name} pairs={len(pair_dataset)} "
                    f"batch={args.counterfactual_pair_batch_size}"
                )
    observation_rgb_pair_loader = None
    observation_rgb_pair_sampler = None
    if args.observation_rgb_pair_weight > 0.0:
        observation_rgb_pair_dataset = CounterfactualPairDataset(
            args.data_dir,
            str(args.observation_rgb_pairs),
            expected_factor="background",
            normalize_mode=args.normalize,
        )
        observation_rgb_pair_sampler = (
            DistributedSampler(
                observation_rgb_pair_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed + 302,
            )
            if use_ddp
            else None
        )
        observation_rgb_pair_loader = DataLoader(
            observation_rgb_pair_dataset,
            batch_size=args.observation_rgb_pair_batch_size,
            shuffle=observation_rgb_pair_sampler is None,
            sampler=observation_rgb_pair_sampler,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=use_cuda,
            persistent_workers=args.num_workers > 0,
        )
        if is_main_process(rank):
            print(
                "observation RGB pairs="
                f"{len(observation_rgb_pair_dataset)} "
                f"batch={args.observation_rgb_pair_batch_size}"
            )

    default_save = "checkpoints/mscdm.pt"
    output_root = Path(args.output_dir).resolve() if args.output_dir else Path(args.save).resolve().parent
    best_path = output_root / "checkpoint_best.pt" if args.output_dir else Path(args.save).resolve()
    last_path = output_root / "checkpoint_last.pt" if args.output_dir else best_path.with_name(f"{best_path.stem}_last.pt")
    best_meta_path = output_root / "best_meta.json"
    metrics_path = output_root / "metrics.jsonl"
    full_ckpt_dir = output_root / "train_checkpoints"
    extra_best_path = Path(args.best_path).resolve() if args.best_path else None
    if is_main_process(rank):
        output_root.mkdir(parents=True, exist_ok=True)
        if args.output_dir and args.save != default_save:
            logger.warning("--output_dir 已指定，忽略 --save=%s", args.save)
        write_json(
            output_root / "run_config.json",
            {
                "model": asdict(cfg),
                "training": vars(args),
                "world_size": world_size,
                "state_early_ddim_probe": {
                    "rank_local_models_without_ddp_wrapper": (
                        args.state_early_ddim_probe_gradients
                    ),
                    "explicit_distributed_statistic_reduction": (
                        args.state_early_ddim_probe_gradients
                    ),
                    "gradient_measurement": (
                        "mean_each_parameter_gradient_across_ranks_then_rms"
                        if args.state_early_ddim_probe_gradients
                        else None
                    ),
                    "low_reversible_save_memory_disabled": (
                        args.state_early_ddim_probe_gradients
                    ),
                    "low_reversible_block_count": (
                        probe_reversible_block_count
                    ),
                },
                "hard_sampling_audit_epoch0": hard_sampling_audit,
            },
        )

    vis_seed = args.seed if args.val_vis_seed is None else args.val_vis_seed
    prompt_dataset = train_dataset if args.train_limit_per_source > 0 else val_dataset
    val_prompts = (
        pick_val_prompts(prompt_dataset, vis_seed, args.num_val_samples)
        if prompt_dataset is not None
        else []
    )
    start_epoch = int(resume_state.get("epoch", -1)) + 1 if resume_state else 0
    global_step = int(resume_state.get("global_step", 0)) if resume_state else 0
    best_val = float(resume_state.get("best_val", float("inf"))) if resume_state else float("inf")
    previous_loss: float | None = None
    stopped_reason: str | None = None
    last_completed_epoch = start_epoch - 1
    ckpt_queue: deque[Path] = deque(sorted(full_ckpt_dir.glob("epoch_*.pt")))

    def apply_warmup(epoch: int) -> None:
        factor = min(1.0, (epoch + 1) / args.warmup_epochs) if args.warmup_epochs > 0 else 1.0
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr * factor

    def checkpoint_payload(epoch: int, full: bool) -> dict:
        payload = {
            "format_version": 2,
            "model": raw_model.state_dict(),
            "cfg": asdict(cfg),
            "epoch": epoch,
            "global_step": global_step,
            "best_val": best_val,
            "selection_metric": args.checkpoint_selection_metric,
            "val_fixed_seed": args.val_fixed_seed,
            "resume_eligible": bool(full),
        }
        if full:
            payload.update(
                {
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "rng_state": {
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if use_cuda else None,
                        "numpy": np.random.get_state(),
                        "python": random.getstate(),
                    },
                }
            )
        return payload

    def save_rotating(epoch: int) -> None:
        if args.train_ckpt_every <= 0 or args.train_ckpt_max_keep <= 0:
            return
        if (epoch + 1) % args.train_ckpt_every != 0:
            return
        full_ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = full_ckpt_dir / f"epoch_{epoch + 1:04d}.pt"
        torch.save(checkpoint_payload(epoch, full=True), path)
        ckpt_queue.append(path)
        while len(ckpt_queue) > args.train_ckpt_max_keep:
            ckpt_queue.popleft().unlink(missing_ok=True)
        print("saved resumable ->", path)

    for epoch in range(start_epoch, args.epochs):
        # 各 epoch / rank 使用独立且可由 epoch 重建的随机流，恢复后不会让各卡噪声相同。
        epoch_seed = args.seed + epoch * world_size + rank
        random.seed(epoch_seed)
        np.random.seed(epoch_seed)
        torch.manual_seed(epoch_seed)
        if use_cuda:
            torch.cuda.manual_seed_all(epoch_seed)
        apply_warmup(epoch)
        model.train()
        # The mid-only experiment treats low/high as an immutable inference
        # path. Disable their stochastic training layers while retaining
        # gradients from frozen high to the mid prior.
        enforce_control_unet_frozen_eval_mode(
            raw_model,
            args.trainable_scope,
        )
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if state_population_sampler is not None:
            state_population_sampler.set_epoch(epoch)
        if real_state_population_sampler is not None:
            real_state_population_sampler.set_epoch(epoch)
        if trajectory_state_sampler is not None:
            trajectory_state_sampler.set_epoch(epoch)
        population_factor_id = {
            "subject": 0,
            "state_dnn": 1,
            "background": 2,
        }[args.factor_surrogate_population_factor]
        population_is_primary = (
            state_population_loader is not None
            and args.factor_surrogate_fixed_factor == population_factor_id
        )
        primary_train_loader = train_loader
        if population_is_primary:
            primary_train_loader = state_population_loader
        if real_state_population_loader is not None:
            primary_train_loader = real_state_population_loader
        if trajectory_state_loader is not None:
            primary_train_loader = trajectory_state_loader
        state_population_iterator = (
            None
            if population_is_primary or state_population_loader is None
            else iter(state_population_loader)
        )
        for factor_id, pair_sampler in counterfactual_pair_samplers.items():
            pair_sampler.set_epoch(epoch * 2 + factor_id)
        counterfactual_pair_iterators = {
            factor_id: iter(loader)
            for factor_id, loader in counterfactual_pair_loaders.items()
        }
        if observation_rgb_pair_sampler is not None:
            observation_rgb_pair_sampler.set_epoch(epoch)
        observation_rgb_pair_iterator = (
            iter(observation_rgb_pair_loader)
            if observation_rgb_pair_loader is not None
            else None
        )

        train_sum = 0.0
        smoke_grad_audit = None
        smoke_val_rng_preserved = None
        train_batches = 0
        train_component_sums: dict[str, float] = {}
        early_ddim_gradient_probe_records: list[dict[str, float | int]] = []
        cascade_prior_step_records: list[dict] = []
        progress = tqdm(
            primary_train_loader,
            desc=f"train {epoch + 1}/{args.epochs}",
            disable=not is_main_process(rank),
        )
        repeated_train_batch = None
        for batch_index, batch in enumerate(progress):
            if args.max_train_batches > 0 and batch_index >= args.max_train_batches:
                break
            if args.repeat_first_train_batch:
                if repeated_train_batch is None:
                    repeated_train_batch = batch
                else:
                    batch = repeated_train_batch
            factor_index = (
                args.factor_surrogate_fixed_factor
                if args.factor_surrogate_fixed_factor >= 0
                else global_step % 3
            )
            if (
                factor_index == population_factor_id
                and state_population_iterator is not None
            ):
                assert state_population_loader is not None
                try:
                    batch = next(state_population_iterator)
                except StopIteration:
                    state_population_iterator = iter(state_population_loader)
                    batch = next(state_population_iterator)
            (
                images,
                subject_ids,
                state_ids,
                bg_ids,
                negative_subject_ids,
                negative_state_ids,
                negative_bg_ids,
                control_rank_enabled,
                target_masks,
                subject_state_low_mask,
                morphology_mask_target,
                morphology_mask_source_ids,
                morphology_class_ids,
                morphology_counterfactual_class_ids,
            ) = tokenize(
                batch,
                device,
                cfg,
                use_control_negatives=(
                    args.control_rank_weight > 0.0
                    or args.factor_surrogate_weight > 0.0
                    or args.morphology_mask_surrogate_weight > 0.0
                ),
                use_morphology_mask_target=(
                    args.morphology_mask_surrogate_weight > 0.0
                ),
                use_morphology_class=(
                    cfg.conditioning_mode in MORPHOLOGY_STRUCTURED_MODES
                ),
                use_morphology_counterfactual_class=(
                    args.morphology_mask_surrogate_weight > 0.0
                    and cfg.conditioning_mode
                    == MORPHOLOGY_FEATURE_AFFINE_MODE
                ),
            )
            factor_surrogate_target_ids = None
            if args.factor_surrogate_weight > 0.0:
                factor_surrogate_target_ids = (
                    factor_surrogate_targets_from_batch(
                        batch,
                        factor_surrogate_lookup,
                        device,
                    )
                )
            state_population_labels = None
            state_early_ddim_group = None
            if (
                args.state_population_delta_weight > 0.0
                or args.state_prompt_causal_mid_weight > 0.0
                or args.state_prompt_causal_high_centroid_weight > 0.0
            ):
                state_population_labels = state_population_labels_from_batch(
                    batch,
                    state_population_lookup,
                    device,
                    args.state_population_size,
                )
            if grouped_state_population_size > 0:
                state_population_labels = (
                    v31_multiclass_state_labels_from_batch(
                        batch,
                        trajectory_state_lookup,
                        device,
                        grouped_state_population_size,
                    )
                )
                if args.state_early_ddim_probe_gradients:
                    state_early_ddim_group = (
                        v31_multiclass_state_group_from_batch(
                            batch,
                            trajectory_state_lookup,
                        )
                    )
            counterfactual_kwargs = {}
            if args.counterfactual_pair_weight > 0.0:
                pair_factor = global_step % 2
                pair_loader = counterfactual_pair_loaders[pair_factor]
                pair_iterator = counterfactual_pair_iterators[pair_factor]
                try:
                    pair_batch = next(pair_iterator)
                except StopIteration:
                    pair_iterator = iter(pair_loader)
                    counterfactual_pair_iterators[pair_factor] = pair_iterator
                    pair_batch = next(pair_iterator)
                (
                    pair_anchor_images,
                    pair_target_images,
                    pair_anchor_subject_ids,
                    pair_anchor_state_ids,
                    pair_anchor_bg_ids,
                    pair_target_subject_ids,
                    pair_target_state_ids,
                    pair_target_bg_ids,
                    verified_pair_factor,
                ) = tokenize_counterfactual_pair(
                    pair_batch,
                    device,
                    cfg,
                )
                if verified_pair_factor != pair_factor:
                    raise ValueError(
                        "counterfactual pair factor stream is inconsistent"
                    )
                counterfactual_kwargs = {
                    "counterfactual_anchor_x512": pair_anchor_images,
                    "counterfactual_target_x512": pair_target_images,
                    "counterfactual_anchor_subject_ids": (
                        pair_anchor_subject_ids
                    ),
                    "counterfactual_anchor_state_ids": pair_anchor_state_ids,
                    "counterfactual_anchor_bg_ids": pair_anchor_bg_ids,
                    "counterfactual_target_subject_ids": (
                        pair_target_subject_ids
                    ),
                    "counterfactual_target_state_ids": pair_target_state_ids,
                    "counterfactual_target_bg_ids": pair_target_bg_ids,
                    "counterfactual_factor": pair_factor,
                    "counterfactual_pair_weight": (
                        args.counterfactual_pair_weight
                    ),
                    "counterfactual_teacher_anchor_mix": (
                        args.counterfactual_teacher_anchor_mix
                    ),
                    "counterfactual_target_orthogonal_mix": (
                        args.counterfactual_target_orthogonal_mix
                    ),
                    "counterfactual_timestep_min": (
                        args.counterfactual_timestep_min
                    ),
                    "counterfactual_timestep_max": (
                        args.counterfactual_timestep_max
                    ),
                    "counterfactual_teacher_unets": (
                        counterfactual_teacher_unets
                    ),
                }
            observation_rgb_kwargs = {}
            if observation_rgb_pair_iterator is not None:
                assert observation_rgb_pair_loader is not None
                try:
                    observation_pair_batch = next(
                        observation_rgb_pair_iterator
                    )
                except StopIteration:
                    observation_rgb_pair_iterator = iter(
                        observation_rgb_pair_loader
                    )
                    observation_pair_batch = next(
                        observation_rgb_pair_iterator
                    )
                (
                    observation_anchor_images,
                    observation_target_images,
                    observation_anchor_subject_ids,
                    observation_anchor_state_ids,
                    observation_anchor_bg_ids,
                    observation_target_subject_ids,
                    observation_target_state_ids,
                    observation_target_bg_ids,
                    observation_factor,
                ) = tokenize_counterfactual_pair(
                    observation_pair_batch,
                    device,
                    cfg,
                )
                if observation_factor != 2:
                    raise ValueError(
                        "observation RGB pair stream must change background"
                    )
                observation_rgb_kwargs = {
                    "observation_rgb_anchor_x512": (
                        observation_anchor_images
                    ),
                    "observation_rgb_target_x512": (
                        observation_target_images
                    ),
                    "observation_rgb_anchor_subject_ids": (
                        observation_anchor_subject_ids
                    ),
                    "observation_rgb_anchor_state_ids": (
                        observation_anchor_state_ids
                    ),
                    "observation_rgb_anchor_bg_ids": (
                        observation_anchor_bg_ids
                    ),
                    "observation_rgb_target_subject_ids": (
                        observation_target_subject_ids
                    ),
                    "observation_rgb_target_state_ids": (
                        observation_target_state_ids
                    ),
                    "observation_rgb_target_bg_ids": (
                        observation_target_bg_ids
                    ),
                    "observation_rgb_pair_weight": (
                        args.observation_rgb_pair_weight
                    ),
                    "observation_rgb_pair_center_weight": (
                        args.observation_rgb_pair_center_weight
                    ),
                    "observation_rgb_pair_timestep_min": (
                        args.observation_rgb_pair_timestep_min
                    ),
                    "observation_rgb_pair_timestep_max": (
                        args.observation_rgb_pair_timestep_max
                    ),
                }
            optimizer.zero_grad(set_to_none=True)
            cond_drop_masks = None
            if args.cond_drop_mode in {"independent_matched", "stratified"}:
                cond_drop_masks = build_isolated_condition_drop_masks(
                    mode=args.cond_drop_mode,
                    batch_size=images.shape[0],
                    global_step=global_step,
                    rank=rank,
                    base_seed=args.cond_drop_seed,
                    device=device,
                    independent_drop_prob=args.cond_drop_prob,
                )
            cascade_prior_source_mask = None
            if cascade_gradient_active:
                cascade_prior_source_mask = build_cascade_prior_source_mask(
                    batch_size=images.shape[0],
                    probability=cascade_prior_probability(epoch + 1),
                    global_step=global_step,
                    rank=rank,
                    base_seed=args.cascade_prior_mask_seed,
                    device=device,
                )
            cascade_step_record = None
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                losses = model(
                    images,
                    subject_ids,
                    state_ids,
                    bg_ids,
                    cond_drop_prob=args.cond_drop_prob,
                    cond_drop_mode=args.cond_drop_mode,
                    cond_drop_masks=cond_drop_masks,
                    negative_subject_ids=negative_subject_ids,
                    negative_state_ids=negative_state_ids,
                    negative_bg_ids=negative_bg_ids,
                    control_rank_enabled=control_rank_enabled,
                    control_rank_factor=factor_index,
                    control_rank_weight=args.control_rank_weight,
                    control_rank_margin=args.control_rank_margin,
                    control_rank_scope=args.control_rank_scope,
                    factor_surrogate_bundle=factor_surrogate_bundle,
                    factor_surrogate_target_ids=(
                        factor_surrogate_target_ids
                    ),
                    factor_surrogate_weight=args.factor_surrogate_weight,
                    factor_surrogate_timestep_min=(
                        args.factor_surrogate_timestep_min
                    ),
                    factor_surrogate_timestep_max=(
                        args.factor_surrogate_timestep_max
                    ),
                    factor_surrogate_state_population_size=(
                        args.factor_surrogate_state_population_size
                    ),
                    state_population_labels=state_population_labels,
                    state_population_delta_weight=(
                        args.state_population_delta_weight
                    ),
                    state_population_size=args.state_population_size,
                    state_population_timestep_min=(
                        args.state_population_timestep_min
                    ),
                    state_population_timestep_max=(
                        args.state_population_timestep_max
                    ),
                    state_population_pool_size=(
                        args.state_population_pool_size
                    ),
                    state_prompt_causal_mid_weight=(
                        args.state_prompt_causal_mid_weight
                    ),
                    state_prompt_causal_timestep_min=(
                        args.state_prompt_causal_timestep_min
                    ),
                    state_prompt_causal_timestep_max=(
                        args.state_prompt_causal_timestep_max
                    ),
                    state_prompt_causal_pool_size=(
                        args.state_prompt_causal_pool_size
                    ),
                    state_prompt_causal_high_centroid_weight=(
                        args.state_prompt_causal_high_centroid_weight
                    ),
                    state_prompt_causal_high_timestep_min=(
                        args.state_prompt_causal_high_timestep_min
                    ),
                    state_prompt_causal_high_timestep_max=(
                        args.state_prompt_causal_high_timestep_max
                    ),
                    state_prompt_causal_high_pool_size=(
                        args.state_prompt_causal_high_pool_size
                    ),
                    state_trajectory_centroid_weight=(
                        args.state_trajectory_centroid_weight
                    ),
                    state_trajectory_population_size=(
                        args.state_trajectory_population_size
                    ),
                    state_trajectory_pool_size=(
                        args.state_trajectory_pool_size
                    ),
                    state_trajectory_timestep=(
                        args.state_trajectory_timestep
                    ),
                    state_trajectory_generator=(
                        make_v31_state_trajectory_generator(
                            device=device,
                            base_seed=args.state_trajectory_noise_seed,
                            global_step=global_step,
                            rank=rank,
                        )
                        if args.state_trajectory_centroid_weight > 0.0
                        else None
                    ),
                    state_early_ddim_centroid_weight=(
                        args.state_early_ddim_centroid_weight
                    ),
                    state_early_ddim_population_size=(
                        args.state_early_ddim_population_size
                    ),
                    state_early_ddim_pool_size=(
                        args.state_early_ddim_pool_size
                    ),
                    state_early_ddim_inference_steps=(
                        args.state_early_ddim_inference_steps
                    ),
                    state_early_ddim_trace_steps=(
                        args.state_early_ddim_trace_steps
                    ),
                    state_early_ddim_subject_cfg=(
                        args.state_early_ddim_subject_cfg
                    ),
                    state_early_ddim_generator=(
                        make_v31_state_trajectory_generator(
                            device=device,
                            base_seed=args.state_early_ddim_noise_seed,
                            global_step=global_step,
                            rank=rank,
                        )
                        if args.state_early_ddim_centroid_weight > 0.0
                        else None
                    ),
                    state_early_ddim_probe_gradients=(
                        args.state_early_ddim_probe_gradients
                    ),
                    morphology_mask_surrogate_bundle=(
                        morphology_mask_surrogate_bundle
                    ),
                    morphology_mask_target=morphology_mask_target,
                    morphology_mask_source_ids=(
                        morphology_mask_source_ids
                    ),
                    morphology_mask_surrogate_weight=(
                        args.morphology_mask_surrogate_weight
                    ),
                    morphology_mask_surrogate_timestep_min=(
                        args.morphology_mask_surrogate_timestep_min
                    ),
                    morphology_mask_surrogate_timestep_max=(
                        args.morphology_mask_surrogate_timestep_max
                    ),
                    foreground_mask=target_masks,
                    foreground_loss_mix=args.foreground_loss_mix,
                    prior_proxy_mix=args.prior_proxy_mix,
                    cascade_prior_source_mask=cascade_prior_source_mask,
                    cascade_prior_gradient_mode=(
                        args.cascade_prior_gradient_mode
                    ),
                    cascade_prior_capture_forward=(global_step == 0),
                    subject_state_low_mask=subject_state_low_mask,
                    spatial_subject_mask=(
                        target_masks
                        if cfg.conditioning_mode
                        in MORPHOLOGY_SPATIAL_MASK_MODES
                        else None
                    ),
                    morphology_class_ids=morphology_class_ids,
                    morphology_counterfactual_class_ids=(
                        morphology_counterfactual_class_ids
                    ),
                    morphology_pair_preservation_weight=(
                        args.morphology_pair_preservation_weight
                    ),
                    **counterfactual_kwargs,
                    **observation_rgb_kwargs,
                )
                loss = losses["loss_total"]
            if cascade_gradient_active and is_main_process(rank):
                cascade_step_record = {
                    "batch_index": batch_index,
                    "global_step_before_update": global_step,
                    "epoch": epoch + 1,
                    "schedule_probability": cascade_prior_probability(
                        epoch + 1
                    ),
                    "loss_low": float(losses["loss_low"].detach()),
                    "loss_mid": float(losses["loss_mid"].detach()),
                    "loss_high": float(losses["loss_high"].detach()),
                    "loss_total": float(loss.detach()),
                    "predicted_fraction": float(
                        losses[
                            "metric_cascade_prior_predicted_fraction"
                        ].detach()
                    ),
                    "mid_predicted_rmse_clean": float(
                        losses[
                            "metric_cascade_prior_mid_predicted_rmse_clean"
                        ].detach()
                    ),
                    "high_predicted_rmse_clean": float(
                        losses[
                            "metric_cascade_prior_high_predicted_rmse_clean"
                        ].detach()
                    ),
                }
                if global_step == 0:
                    cascade_step_record["forward_fingerprints"] = getattr(
                        raw_model,
                        "_last_cascade_prior_forward_fingerprints",
                        {},
                    )
            finite = torch.tensor(1.0 if losses_all_finite(losses) else 0.0, device=device)
            if use_ddp:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if finite.item() == 0:
                stopped_reason = "non_finite_loss"
                logger.error("non-finite loss at epoch=%s batch=%s step=%s", epoch + 1, batch_index, global_step + 1)
                break

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.p10_smoke_audit and batch_index == 0:
                smoke_groups = {}
                frozen_counts = {"vae": 0, "clip": 0, "other": 0}
                for name, parameter in raw_model.named_parameters():
                    if not parameter.requires_grad:
                        if parameter.grad is not None:
                            raise RuntimeError(f"frozen P10 parameter has gradient: {name}")
                        frozen_key = (
                            "vae" if name.startswith("vae_") else
                            "clip" if name.startswith("text_enc.clip.") else "other"
                        )
                        frozen_counts[frozen_key] += parameter.numel()
                        continue
                    group = name.split(".", 1)[0]
                    entry = smoke_groups.setdefault(group, {"parameters": 0, "gradient_parameters": 0, "grads": []})
                    entry["parameters"] += parameter.numel()
                    if parameter.grad is not None:
                        entry["gradient_parameters"] += parameter.numel()
                        entry["grads"].append(parameter.grad.detach())
                smoke_grad_audit = {"frozen_parameter_count": frozen_counts, "groups": {}}
                for group, entry in smoke_groups.items():
                    grads = entry.pop("grads")
                    finite = bool(torch.stack([torch.isfinite(grad).all() for grad in grads]).all()) if grads else True
                    norm = float(torch.linalg.vector_norm(torch.stack([
                        torch.linalg.vector_norm(grad.float()) for grad in grads
                    ])).item()) if grads else 0.0
                    smoke_grad_audit["groups"][group] = {
                        **entry, "finite": finite, "gradient_norm": norm,
                    }
                    if not finite:
                        raise RuntimeError(f"non-finite P10 gradient in {group}")
                for group in ("unet_low", "unet_mid", "unet_high"):
                    if smoke_grad_audit["groups"].get(group, {}).get("gradient_norm", 0.0) <= 0.0:
                        raise RuntimeError(f"missing P10 diffusion gradient: {group}")
                contrast_norm = smoke_grad_audit["groups"].get("contrastive", {}).get("gradient_norm", 0.0)
                if (cfg.contrastive_weight == 0.0 and contrast_norm != 0.0) or (
                    cfg.contrastive_weight > 0.0 and contrast_norm <= 0.0
                ):
                    raise RuntimeError("P10 contrastive gradient does not match configured loss weight")
            if args.state_trajectory_centroid_weight > 0.0:
                trajectory_gradients = getattr(
                    raw_model,
                    "_last_state_trajectory_activation_gradient_rms",
                    {},
                )
                for branch_name in ("low", "mid", "high"):
                    component_name = (
                        f"metric_state_trajectory_{branch_name}_"
                        "activation_gradient_rms"
                    )
                    train_component_sums[component_name] = (
                        train_component_sums.get(component_name, 0.0)
                        + float(trajectory_gradients.get(branch_name, 0.0))
                    )
            if args.state_early_ddim_centroid_weight > 0.0:
                early_gradients = getattr(
                    raw_model,
                    "_last_state_early_ddim_activation_gradient_rms",
                    {},
                )
                for component_name in ("base_low", "step1", "step2"):
                    metric_name = (
                        "metric_state_early_ddim_"
                        f"{component_name}_activation_gradient_rms"
                    )
                    train_component_sums[metric_name] = (
                        train_component_sums.get(metric_name, 0.0)
                        + float(early_gradients.get(component_name, 0.0))
                    )
            if cascade_gradient_active:
                cascade_gradients = getattr(
                    raw_model,
                    "_last_cascade_prior_activation_gradient_rms",
                    {},
                )
                for component_name in (
                    "low_from_mid_high",
                    "mid_from_high",
                ):
                    metric_name = (
                        "metric_cascade_prior_"
                        f"{component_name}_activation_gradient_rms"
                    )
                    train_component_sums[metric_name] = (
                        train_component_sums.get(metric_name, 0.0)
                        + float(cascade_gradients.get(component_name, 0.0))
                    )
                if cascade_step_record is not None:
                    cascade_step_record["cross_scale_activation_gradient_rms"] = {
                        name: float(cascade_gradients.get(name, 0.0))
                        for name in (
                            "low_from_mid_high",
                            "mid_from_high",
                        )
                    }
                    cascade_prior_step_records.append(cascade_step_record)
            if args.state_early_ddim_probe_gradients:
                parameter_stats = getattr(
                    raw_model,
                    "_last_state_early_ddim_parameter_gradient_stats",
                    {},
                )
                required_stats = {
                    "class_count",
                    "base_sum_sq",
                    "base_count",
                    "auxiliary_sum_sq",
                    "auxiliary_count",
                    "parameter_tensor_count",
                    "parameter_name_digest",
                    "gradient_aggregation",
                }
                if set(parameter_stats) != required_stats:
                    raise RuntimeError(
                        "高噪声DDIM参数梯度探针没有返回完整统计"
                    )
                probe_values = torch.stack(
                    [
                        parameter_stats["base_sum_sq"],
                        parameter_stats["auxiliary_sum_sq"],
                    ]
                ).to(device=device, dtype=torch.float64)
                probe_counts = torch.stack(
                    [
                        parameter_stats["base_count"],
                        parameter_stats["auxiliary_count"],
                    ]
                ).to(device=device, dtype=torch.float64)
                if use_ddp:
                    # model.training_losses has already averaged every
                    # parameter gradient across ranks before computing these
                    # sums.  Every rank must therefore hold the same global
                    # DDP-equivalent statistic; summing it again here would
                    # duplicate the single optimizer-update vector.
                    probe_vectors_by_rank = [
                        torch.zeros_like(probe_values)
                        for _ in range(world_size)
                    ]
                    dist.all_gather(probe_vectors_by_rank, probe_values)
                    if any(
                        not torch.allclose(
                            probe_values,
                            rank_values,
                            rtol=0.0,
                            atol=1e-12,
                        )
                        for rank_values in probe_vectors_by_rank
                    ):
                        raise RuntimeError(
                            "各rank的DDP等价参数梯度能量统计不一致"
                        )
                    counts_by_rank = [
                        torch.zeros_like(probe_counts) for _ in range(world_size)
                    ]
                    dist.all_gather(counts_by_rank, probe_counts)
                    if any(
                        not torch.equal(probe_counts, rank_counts)
                        for rank_counts in counts_by_rank
                    ):
                        raise RuntimeError(
                            "各rank的高噪声DDIM梯度参数计数不一致"
                        )
                class_count = int(parameter_stats["class_count"].item())
                class_count_tensor = torch.tensor(
                    [class_count], device=device, dtype=torch.int64
                )
                if use_ddp:
                    class_counts_by_rank = [
                        torch.zeros_like(class_count_tensor)
                        for _ in range(world_size)
                    ]
                    dist.all_gather(
                        class_counts_by_rank, class_count_tensor
                    )
                    if any(
                        int(rank_value.item()) != class_count
                        for rank_value in class_counts_by_rank
                    ):
                        raise RuntimeError(
                            "各rank的高噪声DDIM状态类别数不一致"
                        )
                group_records = [state_early_ddim_group]
                if use_ddp:
                    group_records = [None for _ in range(world_size)]
                    dist.all_gather_object(
                        group_records,
                        state_early_ddim_group,
                    )
                if class_count not in {2, 3}:
                    raise RuntimeError("参数梯度探针只接受二/三状态批次")
                base_sum_sq = float(probe_values[0].item())
                auxiliary_sum_sq = float(probe_values[1].item())
                base_count = int(probe_counts[0].item())
                auxiliary_count = int(probe_counts[1].item())
                parameter_tensor_count = int(
                    parameter_stats["parameter_tensor_count"].item()
                )
                parameter_name_digest = str(
                    parameter_stats["parameter_name_digest"]
                )
                gradient_aggregation = str(
                    parameter_stats["gradient_aggregation"]
                )
                expected_gradient_aggregation = (
                    "mean_parameter_gradient_across_ranks_then_rms"
                    if use_ddp
                    else "single_rank_parameter_gradient_rms"
                )
                if gradient_aggregation != expected_gradient_aggregation:
                    raise RuntimeError(
                        "高噪声DDIM参数梯度聚合口径不符合运行合同"
                    )
                parameter_contracts = [
                    {
                        "tensor_count": parameter_tensor_count,
                        "name_digest": parameter_name_digest,
                    }
                ]
                if use_ddp:
                    parameter_contracts = [None for _ in range(world_size)]
                    dist.all_gather_object(
                        parameter_contracts,
                        {
                            "tensor_count": parameter_tensor_count,
                            "name_digest": parameter_name_digest,
                        },
                    )
                    if any(
                        contract != parameter_contracts[0]
                        for contract in parameter_contracts
                    ):
                        raise RuntimeError(
                            "各rank的low U-Net梯度参数集合不一致"
                        )
                if (
                    not math.isfinite(base_sum_sq)
                    or not math.isfinite(auxiliary_sum_sq)
                    or base_sum_sq <= 0.0
                    or auxiliary_sum_sq <= 0.0
                    or base_count <= 0
                    or auxiliary_count <= 0
                    or base_count != auxiliary_count
                ):
                    raise RuntimeError(
                        "高噪声DDIM参数梯度统计必须有限、为正且参数计数一致"
                    )
                if is_main_process(rank):
                    early_ddim_gradient_probe_records.append(
                        {
                            "batch_index": batch_index,
                            "class_count": class_count,
                            "world_size": world_size,
                            "base_sum_sq": base_sum_sq,
                            "base_count": base_count,
                            "base_rms": (
                                base_sum_sq / max(base_count, 1)
                            ) ** 0.5,
                            "auxiliary_sum_sq": auxiliary_sum_sq,
                            "auxiliary_count": auxiliary_count,
                            "auxiliary_rms": (
                                auxiliary_sum_sq
                                / max(auxiliary_count, 1)
                            ) ** 0.5,
                            "parameter_tensor_count": parameter_tensor_count,
                            "parameter_name_digest": parameter_name_digest,
                            "gradient_aggregation": gradient_aggregation,
                            "rank_groups": group_records,
                        }
                    )
            if args.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(
                    (p for p in raw_model.parameters() if p.requires_grad),
                    args.grad_clip_norm,
                )
            if args.state_early_ddim_probe_gradients:
                # This route is a backward-only measurement.  Do not even
                # invoke AdamW with lr=0: avoiding optimizer state creation
                # makes the no-update contract explicit and keeps the return
                # checkpoint compact enough for identity auditing.
                pass
            else:
                scaler.step(optimizer)
                scaler.update()
            value = float(loss.detach())
            if (
                args.loss_spike_warn_factor > 0
                and previous_loss is not None
                and previous_loss > 0
                and value > previous_loss * args.loss_spike_warn_factor
            ):
                logger.warning("loss spike: current=%s previous=%s", value, previous_loss)
            previous_loss = value
            train_sum += value
            train_batches += 1
            for key, component in losses.items():
                if key.startswith((
                    "loss_state_trajectory_",
                    "metric_state_trajectory_",
                    "loss_state_early_ddim_",
                    "metric_state_early_ddim_",
                    "loss_observation_rgb_pair_",
                    "metric_observation_rgb_pair_",
                    "metric_condition_mask_",
                    "metric_cascade_prior_",
                )):
                    train_component_sums[key] = (
                        train_component_sums.get(key, 0.0)
                        + float(component.detach())
                    )
            global_step += 1
            progress.set_postfix(loss=f"{value:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        if stopped_reason:
            if is_main_process(rank):
                emergency = checkpoint_payload(epoch - 1, full=True)
                emergency["resume_eligible"] = False
                emergency["partial_epoch"] = epoch + 1
                emergency["stopped_reason"] = stopped_reason
                torch.save(emergency, output_root / "checkpoint_emergency.pt")
            break

        train_mean = train_sum / max(train_batches, 1)
        global_train_batches = train_batches
        if use_ddp:
            train_tensor = torch.tensor([train_sum, train_batches], device=device, dtype=torch.float64)
            dist.all_reduce(train_tensor, op=dist.ReduceOp.SUM)
            train_mean = float(train_tensor[0] / train_tensor[1].clamp(min=1))
            global_train_batches = int(train_tensor[1].item())
        train_component_means: dict[str, float] = {}
        component_names = sorted(train_component_sums)
        if component_names:
            component_tensor = torch.tensor(
                [train_component_sums[name] for name in component_names],
                device=device,
                dtype=torch.float64,
            )
            if use_ddp:
                dist.all_reduce(component_tensor, op=dist.ReduceOp.SUM)
            denominator = max(global_train_batches, 1)
            train_component_means = {
                name: float(component_tensor[index] / denominator)
                for index, name in enumerate(component_names)
            }
        last_completed_epoch = epoch
        val_mean: float | None = None
        val_selection_mean: float | None = None

        if val_loader is not None and (epoch + 1) % args.val_every == 0:
            if args.p10_smoke_audit:
                smoke_cpu_rng_before = torch.get_rng_state().clone()
                smoke_cuda_rng_before = torch.cuda.get_rng_state().clone() if use_cuda else None
            # Control-ranking variants can leave a large fragmented CUDA cache
            # after their extra branch forwards.  Release only unused cached
            # blocks before the validation VAE requests its large attention
            # workspace; live tensors and training numerics are unchanged.
            if use_cuda:
                torch.cuda.empty_cache()
            model.eval()
            val_sum = 0.0
            val_selection_sum = 0.0
            val_batches = 0
            val_bad = False
            validation_rng = (
                torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if use_cuda else [])
                if args.val_fixed_seed is not None else contextlib.nullcontext()
            )
            with torch.no_grad(), validation_rng:
                if args.val_fixed_seed is not None:
                    torch.manual_seed(args.val_fixed_seed + rank)
                    if use_cuda:
                        torch.cuda.manual_seed_all(args.val_fixed_seed + rank)
                for batch_index, batch in enumerate(val_loader):
                    if args.max_val_batches > 0 and batch_index >= args.max_val_batches:
                        break
                    (
                        images,
                        subject_ids,
                        state_ids,
                        bg_ids,
                        _,
                        _,
                        _,
                        _,
                        target_masks,
                        subject_state_low_mask,
                        _,
                        _,
                        morphology_class_ids,
                        _,
                    ) = tokenize(
                        batch,
                        device,
                        cfg,
                        use_morphology_class=(
                            cfg.conditioning_mode
                            in MORPHOLOGY_STRUCTURED_MODES
                        ),
                    )
                    with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                        losses = raw_model.training_losses(
                            images,
                            subject_ids,
                            state_ids,
                            bg_ids,
                            foreground_mask=target_masks,
                            foreground_loss_mix=args.foreground_loss_mix,
                            prior_proxy_mix=args.prior_proxy_mix,
                            subject_state_low_mask=subject_state_low_mask,
                            spatial_subject_mask=(
                                target_masks
                                if cfg.conditioning_mode
                                in MORPHOLOGY_SPATIAL_MASK_MODES
                                else None
                            ),
                            morphology_class_ids=morphology_class_ids,
                        )
                    if not losses_all_finite(losses):
                        val_bad = True
                        break
                    val_sum += float(losses["loss_total"].detach())
                    val_selection_sum += float(
                        validation_selection_loss(
                            losses, args.checkpoint_selection_metric
                        ).detach()
                    )
                    val_batches += 1
            if args.p10_smoke_audit:
                smoke_val_rng_preserved = bool(torch.equal(smoke_cpu_rng_before, torch.get_rng_state()))
                if use_cuda:
                    smoke_val_rng_preserved = smoke_val_rng_preserved and bool(
                        torch.equal(smoke_cuda_rng_before, torch.cuda.get_rng_state())
                    )
                if not smoke_val_rng_preserved:
                    raise RuntimeError("P10 validation changed training RNG state")
            totals = torch.tensor([val_sum, val_selection_sum, val_batches, float(val_bad)], device=device, dtype=torch.float64)
            if use_ddp:
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            if totals[3] == 0 and totals[2] > 0:
                val_mean = float(totals[0] / totals[2])
                val_selection_mean = float(totals[1] / totals[2])
                if is_main_process(rank) and val_selection_mean < best_val:
                    best_val = val_selection_mean
                    best_state = checkpoint_payload(epoch, full=False)
                    torch.save(best_state, best_path)
                    if extra_best_path is not None:
                        extra_best_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(best_state, extra_best_path)
                    write_json(
                        best_meta_path,
                        {
                            "epoch": epoch + 1,
                            "val_loss": best_val,
                            "selection_metric": args.checkpoint_selection_metric,
                            "val_total_loss": val_mean,
                            "val_fixed_seed": args.val_fixed_seed,
                            "global_step": global_step,
                            "path": str(best_path),
                            "iso_utc": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    print("saved best inference checkpoint ->", best_path)
            elif is_main_process(rank):
                logger.error("validation returned non-finite loss or zero batches")

        if is_main_process(rank):
            append_jsonl(
                metrics_path,
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "train_loss": train_mean,
                    "train_components": train_component_means or None,
                    "state_early_ddim_gradient_probe": (
                        early_ddim_gradient_probe_records or None
                    ),
                    "cascade_prior_step_records": (
                        cascade_prior_step_records or None
                    ),
                    "val_loss": val_mean,
                    "val_selection_loss": val_selection_mean,
                    "selection_metric": args.checkpoint_selection_metric,
                    "lr": optimizer.param_groups[0]["lr"],
                    "iso_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            print(f"epoch={epoch + 1} train_loss={train_mean:.4f} val_loss={val_mean}")
            if train_component_means:
                print(
                    "train state-control auxiliary components:",
                    json.dumps(train_component_means, sort_keys=True),
                )
        if use_ddp:
            dist.barrier()
        if args.p10_smoke_audit:
            append_jsonl(output_root / f"p10_smoke_rank{rank}.jsonl", {
                "rank": rank, "epoch": epoch + 1, "global_step": global_step,
                "world_size": world_size, "batch_per_rank": args.batch_size,
                "train_batches": train_batches, "val_rng_preserved": smoke_val_rng_preserved,
                "gradient_audit": smoke_grad_audit,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if use_cuda else None,
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if use_cuda else None,
            })
        if (
            is_main_process(rank)
            and args.sample_every > 0
            and (epoch + 1) % args.sample_every == 0
            and val_prompts
        ):
            raw_model.eval()
            run_qualitative(
                epoch + 1,
                raw_model,
                device,
                cfg,
                val_prompts,
                output_root,
                args.sample_cfg_scale,
                args.sample_num_steps,
                vis_seed,
            )
        if use_ddp:
            dist.barrier()
        if is_main_process(rank):
            save_rotating(epoch)
        if use_ddp:
            dist.barrier()

    if use_ddp:
        dist.barrier()
    if is_main_process(rank) and last_completed_epoch >= 0 and not stopped_reason:
        torch.save(
            checkpoint_payload(
                last_completed_epoch,
                full=not args.state_early_ddim_probe_gradients,
            ),
            last_path,
        )
        if args.state_early_ddim_probe_gradients:
            print("saved no-update probe checkpoint ->", last_path)
        else:
            print("saved resumable last ->", last_path)
        if val_loader is None and not best_path.exists():
            torch.save(checkpoint_payload(last_completed_epoch, full=False), best_path)
            print("saved inference checkpoint ->", best_path)
    if use_ddp:
        dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
