"""Deterministic UIIS COCO-instance to SUIM-compatible semantic masks."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np


# UIIS category IDs follow the dataset's COCO annotations.  Values are the
# frozen SUIM-compatible semantic IDs defined in datasets.label_mapping.
UIIS_TO_SUIM_CLASS = {
    1: 6,  # fish -> fish_vertebrates
    2: 5,  # reefs -> reefs_invertebrates
    3: 2,  # aquatic plants -> aquatic_plants
    4: 3,  # wrecks/ruins -> wrecks_ruins
    5: 1,  # human divers -> human_divers
    6: 4,  # robots -> robots_instruments
    7: 7,  # sea-floor -> seafloor_rocks
}


def validate_categories(categories: Iterable[dict[str, Any]]) -> None:
    found = {int(category["id"]): str(category["name"]).strip().lower() for category in categories}
    expected = {
        1: "fish", 2: "reefs", 3: "aquatic plants", 4: "wrecks/ruins",
        5: "human divers", 6: "robots", 7: "sea-floor",
    }
    if found != expected:
        raise ValueError(f"Unexpected UIIS category mapping: {found}")


def rasterize_semantic_mask(*, height: int, width: int, annotations: Iterable[dict[str, Any]]) -> tuple[np.ndarray, dict[str, int]]:
    """Rasterize COCO polygons with a fixed, auditable overlap policy.

    Background is zero.  Larger instances are painted first, then smaller
    ones, so a smaller visible instance has priority in an overlap. Equal-area
    annotations use ascending annotation ID.  Every cross-class overwrite is
    counted and reported rather than silently ignored.
    """
    if height <= 0 or width <= 0:
        raise ValueError("Mask dimensions must be positive.")
    ordered = sorted(annotations, key=lambda item: (-float(item.get("area", 0.0)), int(item.get("id", -1))))
    mask = np.zeros((height, width), dtype=np.uint8)
    counters = {"annotations": 0, "polygon_parts": 0, "cross_class_overlap_pixels": 0, "same_class_overlap_pixels": 0}
    for annotation in ordered:
        category = int(annotation["category_id"])
        if category not in UIIS_TO_SUIM_CLASS:
            raise ValueError(f"Unknown UIIS category id: {category}")
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, list):
            raise ValueError("UIIS conversion currently requires polygon-list segmentations.")
        instance = np.zeros_like(mask)
        for polygon in segmentation:
            values = np.asarray(polygon, dtype=np.float32)
            if values.ndim != 1 or len(values) < 6 or len(values) % 2:
                raise ValueError(f"Invalid polygon for annotation {annotation.get('id')}")
            points = np.rint(values.reshape(-1, 2)).astype(np.int32).reshape(1, -1, 2)
            cv2.fillPoly(instance, points, color=1)
            counters["polygon_parts"] += 1
        target = UIIS_TO_SUIM_CLASS[category]
        occupied = instance.astype(bool)
        counters["cross_class_overlap_pixels"] += int((occupied & (mask != 0) & (mask != target)).sum())
        counters["same_class_overlap_pixels"] += int((occupied & (mask == target)).sum())
        mask[occupied] = target
        counters["annotations"] += 1
    return mask, counters
