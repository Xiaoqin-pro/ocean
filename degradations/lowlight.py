"""Deterministic low-light degradation."""
from __future__ import annotations

import numpy as np


def apply_lowlight(image: np.ndarray, *, exposure: float, gamma: float) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB uint8 image.")
    if not 0 < exposure <= 1 or gamma < 1:
        raise ValueError("Low-light exposure must be in (0, 1] and gamma must be at least 1.")
    pixels = image.astype(np.float32) / 255.0
    degraded = np.power(np.clip(pixels * exposure, 0, 1), gamma)
    return np.clip(degraded * 255.0, 0, 255).astype(np.uint8)
