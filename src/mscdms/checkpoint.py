from __future__ import annotations

from dataclasses import fields

import torch

from .config import MSCDMConfig


def load_checkpoint(path: str, map_location=None):
    """
    加载训练保存的完整 checkpoint（含 model state_dict 与 cfg 等）。
    PyTorch 2.6+ 默认 torch.load(..., weights_only=True) 无法反序列化 MSCDMConfig，
    故对可信本地文件使用 weights_only=False。
    """
    kw = {"map_location": map_location}
    try:
        kw["weights_only"] = False
        return torch.load(path, **kw)
    except TypeError:
        return torch.load(path, map_location=map_location)


def config_from_checkpoint(checkpoint) -> MSCDMConfig:
    """兼容旧 checkpoint 中的 dataclass 与新版纯字典配置。"""
    if not isinstance(checkpoint, dict) or "cfg" not in checkpoint:
        return MSCDMConfig()
    saved = checkpoint["cfg"]
    if isinstance(saved, MSCDMConfig):
        # 旧 checkpoint 可能 pickle 了当时版本的 dataclass 实例；新字段不会
        # 自动出现在该实例的 __dict__ 中，因此用当前默认值补齐后重建。
        defaults = MSCDMConfig()
        return MSCDMConfig(
            **{
                field.name: getattr(saved, field.name, getattr(defaults, field.name))
                for field in fields(MSCDMConfig)
            }
        )
    if isinstance(saved, dict):
        known = {f.name for f in fields(MSCDMConfig)}
        return MSCDMConfig(**{k: v for k, v in saved.items() if k in known})
    raise TypeError(f"不支持的 checkpoint cfg 类型: {type(saved)!r}")
