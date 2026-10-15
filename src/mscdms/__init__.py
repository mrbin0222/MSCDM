"""Structured conditioning and multiscale microscopy generation.

Load text-generation dependencies when that model is requested, so standalone
segmentation and evaluation tools can use their own execution environments.
"""
from .config import MSCDMConfig

__all__ = ["MSCDM", "MSCDMConfig"]


def __getattr__(name):
    if name == "MSCDM":
        from .model import MSCDM
        globals()[name] = MSCDM
        return MSCDM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
