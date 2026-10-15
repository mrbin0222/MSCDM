from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class RelationPolicy:
    """可复现的细胞邻接图构造策略。"""
    name: str = 'mask_touch_or_knn_v1'
    touch_pixels: int = 2
    center_distance_threshold: float = 0.08
    knn: int = 4
    use_delaunay: bool = False

def bbox_xywh_from_mask(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0 = float(xs.min())
    y0 = float(ys.min())
    x1 = float(xs.max() + 1)
    y1 = float(ys.max() + 1)
    return [x0, y0, x1 - x0, y1 - y0]
