from __future__ import annotations

from typing import Final

import numpy as np
from PIL import Image


CLASS_NAMES: Final[list[str]] = [
    "background_waterbody",
    "human_divers",
    "aquatic_plants",
    "wrecks_ruins",
    "robots_instruments",
    "reefs_invertebrates",
    "fish_vertebrates",
    "seafloor_rocks",
]

# Official SUIM semantic-mask RGB codes. PIL is deliberately used so that
# the array channel order is RGB, never OpenCV's default BGR.
RGB_TO_CLASS: Final[dict[tuple[int, int, int], int]] = {
    (0, 0, 0): 0,
    (0, 0, 255): 1,
    (0, 255, 0): 2,
    (0, 255, 255): 3,
    (255, 0, 0): 4,
    (255, 0, 255): 5,
    (255, 255, 0): 6,
    (255, 255, 255): 7,
}
CLASS_TO_RGB: Final[dict[int, tuple[int, int, int]]] = {
    class_id: rgb for rgb, class_id in RGB_TO_CLASS.items()
}
ID2LABEL: Final[dict[int, str]] = dict(enumerate(CLASS_NAMES))
LABEL2ID: Final[dict[str, int]] = {name: index for index, name in ID2LABEL.items()}


def rgb_mask_to_index(mask: Image.Image | np.ndarray, *, strict: bool = True) -> np.ndarray:
    """Convert an RGB SUIM mask into an HxW uint8 class-index mask."""
    rgb = (
        np.asarray(mask.convert("RGB"), dtype=np.uint8)
        if isinstance(mask, Image.Image)
        else np.asarray(mask, dtype=np.uint8)
    )
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected an HxWx3 RGB mask, got {rgb.shape}")

    indexed = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for color, class_id in RGB_TO_CLASS.items():
        indexed[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = class_id

    unknown = indexed == 255
    if strict and np.any(unknown):
        colors = np.unique(rgb[unknown].reshape(-1, 3), axis=0)
        raise ValueError(
            f"Found {len(colors)} unknown RGB colors; first values: {colors[:20].tolist()}"
        )
    return indexed


def rgb_mask_to_index_bit_threshold(mask: Image.Image | np.ndarray) -> tuple[np.ndarray, int]:
    """Quantize SUIM's RGB bit-code masks with an explicit 128 channel threshold.

    Some official BMP masks contain interpolated transition colors at class
    boundaries. SUIM's eight official codes are the eight RGB cube corners, so
    thresholding each channel is equivalent to choosing the nearest official
    code while remaining fully deterministic. The returned count records every
    pixel that was not already an exact official RGB color.
    """
    rgb = (
        np.asarray(mask.convert("RGB"), dtype=np.uint8)
        if isinstance(mask, Image.Image)
        else np.asarray(mask, dtype=np.uint8)
    )
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected an HxWx3 RGB mask, got {rgb.shape}")
    exact = np.zeros(rgb.shape[:2], dtype=bool)
    for color in RGB_TO_CLASS:
        exact |= np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)
    indexed = (
        (rgb[..., 0] >= 128).astype(np.uint8) * 4
        + (rgb[..., 1] >= 128).astype(np.uint8) * 2
        + (rgb[..., 2] >= 128).astype(np.uint8)
    )
    return indexed, int((~exact).sum())


def index_mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert an HxW SUIM class-index mask back to RGB for inspection."""
    indexed = np.asarray(mask)
    if indexed.ndim != 2:
        raise ValueError(f"Expected an HxW indexed mask, got {indexed.shape}")
    unknown = np.setdiff1d(np.unique(indexed), np.arange(len(CLASS_NAMES)))
    if unknown.size:
        raise ValueError(f"Cannot visualize invalid class ids: {unknown.tolist()}")

    rgb = np.zeros((*indexed.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_TO_RGB.items():
        rgb[indexed == class_id] = color
    return rgb
