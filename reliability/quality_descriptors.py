"""Input-only image-quality descriptors used by deployable DARC grouping."""
from __future__ import annotations

import cv2
import numpy as np


DESCRIPTOR_NAMES = (
    "mean_luminance", "luminance_std", "luminance_p10", "mean_saturation",
    "log_r_over_g", "log_b_over_g", "normalized_laplacian_variance", "dark_channel_mean",
)


def image_quality_descriptors(image: np.ndarray) -> np.ndarray:
    """Compute the eight frozen, label-free DARC-Seg descriptors from RGB uint8."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB uint8 image.")
    pixels = image.astype(np.float32) / 255.0
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[..., 1] / 255.0
    laplacian = cv2.Laplacian((luminance * 255.0).astype(np.uint8), cv2.CV_64F)
    normalized_laplacian = float(laplacian.var() / (255.0 ** 2))
    dark_channel = pixels.min(axis=2)
    epsilon = 1e-6
    values = np.array((
        luminance.mean(), luminance.std(), np.quantile(luminance, 0.10), saturation.mean(),
        np.log((red.mean() + epsilon) / (green.mean() + epsilon)),
        np.log((blue.mean() + epsilon) / (green.mean() + epsilon)),
        normalized_laplacian, dark_channel.mean(),
    ), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Quality descriptors must be finite.")
    return values

