"""Deterministic RGB channel attenuation for controlled underwater stress tests."""
from __future__ import annotations

import numpy as np


def apply_color_attenuation(
    image: np.ndarray,
    *,
    red_scale: float,
    green_scale: float,
    blue_scale: float,
    blue_green_veil: float,
) -> np.ndarray:
    """Attenuate warm channels and add a fixed blue-green veiling component."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB uint8 image.")
    scales = np.asarray([red_scale, green_scale, blue_scale], dtype=np.float32)
    if np.any(scales < 0) or np.any(scales > 1) or not 0 <= blue_green_veil <= 1:
        raise ValueError("Color attenuation parameters must be within [0, 1].")
    pixels = image.astype(np.float32) / 255.0
    attenuated = pixels * scales
    veil = np.asarray([0.0, 0.55, 0.85], dtype=np.float32) * blue_green_veil
    return np.clip((attenuated + veil) * 255.0, 0, 255).astype(np.uint8)
