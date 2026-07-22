"""Image-only Gaussian blur degradation."""
from __future__ import annotations

import cv2
import numpy as np


def apply_blur(image: np.ndarray, *, kernel_size: int, sigma: float) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB uint8 image.")
    if kernel_size < 3 or kernel_size % 2 == 0 or sigma <= 0:
        raise ValueError("Gaussian blur requires an odd kernel of at least 3 and positive sigma.")
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101)
