"""DDP 分布式训练工具：初始化 / 清理 / rank 判断。

兼容单卡（直接 python script.py）与多卡（torchrun --nproc_per_node=N script.py）。
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def setup_distributed(backend: str = "nccl") -> tuple[int, int, int]:
    """初始化分布式进程组。

    Returns:
        (rank, local_rank, world_size)。
        单卡时返回 (0, 0, 1) 且不初始化进程组。
    """
    if "RANK" not in os.environ:
        return 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if backend not in ("nccl", "gloo"):
        raise ValueError(f"unsupported distributed backend: {backend!r}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if backend == "nccl":
        try:
            # PyTorch 2.11+：显式绑定 NCCL rank 与 CUDA device，避免 barrier
            # 通过 rank 猜测设备并打印 c10d_logger warning。
            dist.init_process_group(backend=backend, device_id=device)
        except TypeError:
            # 兼容仍不支持 device_id 参数的旧版 PyTorch。
            dist.init_process_group(backend=backend)
    else:
        # 多卡离线生成的各 rank 独立推理，仅在结束时使用 barrier；Gloo 控制面
        # 可避免为无张量通信的任务创建 NCCL communicator。
        dist.init_process_group(backend=backend)
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """销毁分布式进程组（仅在已初始化时调用）。"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def get_device(local_rank: int) -> torch.device:
    """根据 local_rank 返回设备；无 CUDA 时回退到 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    return torch.device("cpu")
