"""Segmentation photometry."""
from __future__ import annotations

import numpy as np
from PIL import Image


SCHEMA = 'exp4.p14.renderer_per_instance_p1p99_inverse.v1'


def fit_renderer_photometry(real512: np.ndarray, *, source_role: str) -> dict:
    if source_role != 'training_side_only':
        raise ValueError('photometry metadata must come from training-side reference')
    if real512.shape != (512, 512, 3) or real512.dtype != np.uint8:
        raise ValueError('existing isolated renderer reference must be RGB uint8 at 512')
    values = real512.astype(np.float32)/255.0
    low, high = np.percentile(values, [1.0, 99.0], axis=(0, 1)).astype(np.float32)
    high = np.maximum(high, low+np.float32(1e-6))
    return {'schema': SCHEMA, 'percentiles': [1.0, 99.0],
            'low_01': low.tolist(), 'high_01': high.tolist(),
            'source_role': source_role, 'foreground_scalar_statistics_used': True,
            'fit_rule': 'fixed Stage26 per-channel percentiles; no target-score optimization',
            'pixel_copy_from_real_foreground': False}


def _parameters(meta):
    if meta.get('schema') != SCHEMA or meta.get('source_role') != 'training_side_only':
        raise ValueError('missing or wrong train-side photometry schema')
    low, high = (np.asarray(meta[k], dtype=np.float32).reshape(1, 1, 3) for k in ('low_01', 'high_01'))
    if not np.isfinite(low).all() or not np.isfinite(high).all() or not (high > low).all():
        raise ValueError('invalid renderer affine parameters')
    return low, high


def renderer_tensor_rgb(real_rgb_8: np.ndarray, meta: dict) -> np.ndarray:
    """Return RGB float32 [-1,1], including the original clipping rule."""
    low, high = _parameters(meta)
    values = np.asarray(real_rgb_8, dtype=np.float32)/255.0
    return (np.clip((values-low)/(high-low), 0.0, 1.0)*2.0-1.0).astype(np.float32)


def renderer_to_native_rgb8_float(model_rgb_m11: np.ndarray, meta: dict) -> np.ndarray:
    """Affine inverse into source-normalized 0..255 without 8-bit requantizing."""
    low, high = _parameters(meta)
    model = np.asarray(model_rgb_m11, dtype=np.float32)
    if not np.isfinite(model).all():
        raise ValueError('nonfinite model RGB')
    return ((np.clip((model+1.0)*.5, 0.0, 1.0)*(high-low)+low)*255.0).astype(np.float32)


def compose_native512(model_rgb_m11, reference_background_rgb8, support, meta):
    out = renderer_to_native_rgb8_float(model_rgb_m11, meta)
    out[~support] = np.asarray(reference_background_rgb8, dtype=np.float32)[~support]
    return out


def inverse_project_float(rgb8_float: np.ndarray, geometry: dict) -> np.ndarray:
    """Bicubic RGB float resize retains affine texture; labels never enter."""
    if rgb8_float.shape != (512, 512, 3):
        raise ValueError('renderer inverse projection requires 512 RGB')
    side = geometry['square_side_native']
    square = np.stack([np.asarray(Image.fromarray(np.asarray(rgb8_float[..., c], dtype=np.float32), mode='F').resize((side, side), Image.Resampling.BICUBIC)) for c in range(3)], -1)
    ox, oy = geometry['canvas_offset_xy']; x0, y0, x1, y1 = geometry['bounds_xyxy']
    return np.clip(square[oy:oy+y1-y0, ox:ox+x1-x0], 0.0, 255.0).astype(np.float32)


def paste_native_float(destination, labels, crop, target_id, projected, geometry):
    x, y, n = (crop[k] for k in ('crop_left', 'crop_top', 'crop_size'))
    gx0, gy0, gx1, gy1 = geometry['bounds_xyxy']
    if projected.shape != (gy1-gy0, gx1-gx0, 3):
        raise ValueError('projected coordinate axes/bounds mismatch')
    x0, y0, x1, y1 = max(x, gx0), max(y, gy0), min(x+n, gx1), min(y+n, gy1)
    if x0 >= x1 or y0 >= y1:
        raise ValueError('empty target/crop overlap')
    alpha = labels[y0-y:y1-y, x0-x:x1-x] == target_id
    patch = destination[y0-y:y1-y, x0-x:x1-x]
    patch[alpha] = projected[y0-gy0:y1-gy0, x0-gx0:x1-gx0][alpha]


def gray_uint16(native_rgb8_float):
    gray = np.asarray(native_rgb8_float, dtype=np.float32).mean(-1)
    return np.rint(np.clip(gray, 0.0, 255.0)*257.0).astype(np.uint16)
