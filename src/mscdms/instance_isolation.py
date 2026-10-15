"""Deterministic single-instance compositing for microscopy crops.

The original exp4 cropper used an instance bounding box only to locate a
rectangular patch.  Consequently, pixels from neighbouring labelled instances
were retained.  This module constructs a clean target image instead:

    isolated = target_alpha * source + (1 - target_alpha) * matched_background

Only pixels belonging to the requested instance may contribute source signal.
The replacement background is synthesized from unlabelled pixels in the same
parent image, keeping microscopy modality and channel statistics matched
without copying another labelled instance.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class IsolationConfig:
    crop_size: int = 512
    padding_ratio: float = 0.30
    background_mode: str = "matched"
    background_exclusion_radius: int = 2
    background_blur_radius: float = 0.65
    background_intensity_quantile: float = 0.50
    background_noise_scale: float = 0.15
    inner_feather_radius: float = 0.0
    min_safe_background_pixels: int = 128


@dataclass(frozen=True)
class IsolationDiagnostics:
    target_pixels_native: int
    target_fraction_native: float
    original_other_instance_count: int
    original_other_labeled_pixels: int
    original_other_labeled_fraction: float
    background_pool_pixels: int
    background_pool_fraction: float
    background_pool_fallback: bool
    retained_other_instance_pixels: int = 0


def stable_seed(seed: int, *parts: object) -> int:
    raw = "|".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def square_bounds(
    bbox_xywh: list[int] | tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    """Match the bounds used by the original exp4 instance cropper."""

    x, y, width, height = (int(value) for value in bbox_xywh)
    side = max(width, height) * (1.0 + 2.0 * padding_ratio)
    side = max(1, int(round(side)))
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    x0 = int(round(center_x - side / 2.0))
    y0 = int(round(center_y - side / 2.0))
    x1 = x0 + side
    y1 = y0 + side
    return (
        max(0, x0),
        max(0, y0),
        min(image_width, x1),
        min(image_height, y1),
    )


def _safe_background_mask(
    instance_map: np.ndarray,
    exclusion_radius: int,
    min_pixels: int,
) -> tuple[np.ndarray, bool]:
    labelled = np.asarray(instance_map != 0, dtype=np.uint8) * 255
    if exclusion_radius > 0:
        kernel = exclusion_radius * 2 + 1
        expanded = np.asarray(
            Image.fromarray(labelled, mode="L").filter(
                ImageFilter.MaxFilter(kernel)
            )
        )
        safe = expanded == 0
    else:
        safe = labelled == 0

    fallback = False
    if int(safe.sum()) < min_pixels:
        safe = instance_map == 0
        fallback = True
    if int(safe.sum()) == 0:
        safe = np.ones(instance_map.shape, dtype=bool)
        fallback = True
    return safe, fallback


def _matched_background(
    height: int,
    width: int,
    background_pixels: np.ndarray,
    rng: np.random.Generator,
    blur_radius: float,
    noise_scale: float,
) -> np.ndarray:
    """Generate channel-correlated background with matched robust statistics."""

    pixels = np.asarray(background_pixels, dtype=np.float32).reshape(-1, 3)
    indices = rng.integers(0, len(pixels), size=height * width)
    sampled = pixels[indices].reshape(height, width, 3)

    if blur_radius > 0:
        sampled_u8 = np.clip(np.rint(sampled), 0, 255).astype(np.uint8)
        sampled = np.asarray(
            Image.fromarray(sampled_u8, mode="RGB").filter(
                ImageFilter.GaussianBlur(radius=blur_radius)
            ),
            dtype=np.float32,
        )

    target_mean = pixels.mean(axis=0)
    target_std = pixels.std(axis=0)
    current_mean = sampled.reshape(-1, 3).mean(axis=0)
    current_std = sampled.reshape(-1, 3).std(axis=0)
    scale = np.divide(
        target_std,
        current_std,
        out=np.zeros_like(target_std),
        where=current_std > 1e-6,
    )
    sampled = (
        (sampled - current_mean) * scale * noise_scale + target_mean
    )

    lower = np.percentile(pixels, 0.5, axis=0)
    upper = np.percentile(pixels, 99.5, axis=0)
    sampled = np.minimum(np.maximum(sampled, lower), upper)
    return np.clip(np.rint(sampled), 0, 255).astype(np.uint8)


def _dark_background_pool(
    pixels: np.ndarray,
    *,
    intensity_quantile: float,
    min_pixels: int,
) -> tuple[np.ndarray, bool]:
    """Reject bright residual fluorescence from nominal background pixels."""

    if not 0 < intensity_quantile <= 1:
        raise ValueError("background_intensity_quantile must be in (0, 1]")
    pixels = np.asarray(pixels, dtype=np.uint8).reshape(-1, 3)
    if intensity_quantile >= 1:
        return pixels, False
    intensity = pixels.astype(np.float32).max(axis=1)
    cutoff = float(np.quantile(intensity, intensity_quantile))
    selected = pixels[intensity <= cutoff]
    if len(selected) < min_pixels:
        return pixels, True
    return selected, False


def _inner_alpha(target_mask: np.ndarray, radius: float) -> np.ndarray:
    """Optionally soften the target edge without importing outside pixels."""

    binary = np.asarray(target_mask, dtype=np.float32)
    if radius <= 0:
        return binary
    blurred = np.asarray(
        Image.fromarray((binary * 255).astype(np.uint8), mode="L").filter(
            ImageFilter.GaussianBlur(radius=radius)
        ),
        dtype=np.float32,
    )
    return binary * (blurred / 255.0)


def isolate_instance(
    image: np.ndarray,
    instance_map: np.ndarray,
    instance_id: int,
    bbox_xywh: list[int] | tuple[int, int, int, int],
    *,
    config: IsolationConfig,
    seed: int,
) -> tuple[Image.Image, Image.Image, IsolationDiagnostics]:
    """Return isolated RGB crop, binary target mask, and diagnostics."""

    image = np.asarray(image, dtype=np.uint8)
    instance_map = np.asarray(instance_map)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image array, got {image.shape}")
    if instance_map.ndim != 2:
        raise ValueError(f"Expected 2-D instance map, got {instance_map.shape}")
    if image.shape[:2] != instance_map.shape:
        raise ValueError(
            f"Image/map shape mismatch: {image.shape[:2]} vs {instance_map.shape}"
        )
    if config.background_mode not in {"matched", "zero", "median"}:
        raise ValueError(
            f"Unsupported background mode: {config.background_mode}"
        )

    image_height, image_width = instance_map.shape
    x0, y0, x1, y1 = square_bounds(
        bbox_xywh,
        image_width,
        image_height,
        config.padding_ratio,
    )
    patch = image[y0:y1, x0:x1]
    labels = instance_map[y0:y1, x0:x1]
    target_patch = labels == int(instance_id)
    target_pixels = int(target_patch.sum())
    if target_pixels == 0:
        raise ValueError(f"Instance {instance_id} absent from crop bounds")

    other_labels = np.unique(labels[(labels != 0) & (labels != instance_id)])
    other_pixels = int(((labels != 0) & (labels != instance_id)).sum())

    patch_height, patch_width = target_patch.shape
    side = max(patch_height, patch_width)
    offset_y = (side - patch_height) // 2
    offset_x = (side - patch_width) // 2

    safe_background, pool_fallback = _safe_background_mask(
        instance_map,
        config.background_exclusion_radius,
        config.min_safe_background_pixels,
    )
    background_pixels, dark_pool_fallback = _dark_background_pool(
        image[safe_background],
        intensity_quantile=config.background_intensity_quantile,
        min_pixels=config.min_safe_background_pixels,
    )
    pool_fallback = pool_fallback or dark_pool_fallback
    rng = np.random.default_rng(seed)
    if config.background_mode == "matched":
        background = _matched_background(
            side,
            side,
            background_pixels,
            rng,
            config.background_blur_radius,
            config.background_noise_scale,
        )
    elif config.background_mode == "median":
        median = np.median(background_pixels, axis=0)
        background = np.broadcast_to(
            np.clip(np.rint(median), 0, 255).astype(np.uint8),
            (side, side, 3),
        ).copy()
    else:
        background = np.zeros((side, side, 3), dtype=np.uint8)

    source_canvas = np.zeros((side, side, 3), dtype=np.uint8)
    target_canvas = np.zeros((side, side), dtype=bool)
    source_canvas[
        offset_y : offset_y + patch_height,
        offset_x : offset_x + patch_width,
    ] = patch
    target_canvas[
        offset_y : offset_y + patch_height,
        offset_x : offset_x + patch_width,
    ] = target_patch

    alpha = _inner_alpha(target_canvas, config.inner_feather_radius)[..., None]
    composite = (
        alpha * source_canvas.astype(np.float32)
        + (1.0 - alpha) * background.astype(np.float32)
    )
    composite = np.clip(np.rint(composite), 0, 255).astype(np.uint8)

    isolated = Image.fromarray(composite, mode="RGB").resize(
        (config.crop_size, config.crop_size),
        Image.Resampling.BICUBIC,
    )
    mask = Image.fromarray(target_canvas.astype(np.uint8) * 255, mode="L").resize(
        (config.crop_size, config.crop_size),
        Image.Resampling.NEAREST,
    )

    diagnostics = IsolationDiagnostics(
        target_pixels_native=target_pixels,
        target_fraction_native=target_pixels / float(side * side),
        original_other_instance_count=len(other_labels),
        original_other_labeled_pixels=other_pixels,
        original_other_labeled_fraction=other_pixels / float(side * side),
        background_pool_pixels=len(background_pixels),
        background_pool_fraction=(
            len(background_pixels) / float(instance_map.size)
        ),
        background_pool_fallback=pool_fallback,
    )
    return isolated, mask, diagnostics
