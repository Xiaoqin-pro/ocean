"""Deterministic image-only turbidity and scattering approximation."""
from __future__ import annotations

import hashlib

import numpy as np


def _sample_phase(sample_id: str) -> tuple[float, float]:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return (digest[0] / 255.0 * 2 * np.pi, digest[1] / 255.0 * 2 * np.pi)


def apply_turbidity(
    image: np.ndarray,
    sample_id: str,
    *,
    transmission: float,
    airlight_rgb: tuple[float, float, float],
    spatial_variation: float,
) -> np.ndarray:
    """Apply a stable haze model; the same sample and parameters always match."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB uint8 image.")
    if not 0 < transmission <= 1 or not 0 <= spatial_variation < transmission:
        raise ValueError("Transmission and spatial variation are invalid.")
    airlight = np.asarray(airlight_rgb, dtype=np.float32)
    if airlight.shape != (3,) or np.any(airlight < 0) or np.any(airlight > 1):
        raise ValueError("airlight_rgb must contain three values in [0, 1].")
    height, width = image.shape[:2]
    y, x = np.mgrid[0:height, 0:width]
    phase_x, phase_y = _sample_phase(sample_id)
    texture = (np.sin((2 * np.pi * x / max(width, 1)) + phase_x) + np.cos((2 * np.pi * y / max(height, 1)) + phase_y) + 2) / 4
    local_transmission = np.clip(transmission - spatial_variation * texture, 0.05, 1.0)[..., None]
    pixels = image.astype(np.float32) / 255.0
    degraded = pixels * local_transmission + airlight * (1 - local_transmission)
    return np.clip(degraded * 255.0, 0, 255).astype(np.uint8)
