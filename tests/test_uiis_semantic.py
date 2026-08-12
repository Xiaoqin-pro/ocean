import numpy as np
import pytest

from datasets.uiis_semantic import UIIS_TO_SUIM_CLASS, rasterize_semantic_mask, validate_categories


def test_uiis_mapping_covers_exactly_the_seven_foreground_semantic_classes():
    assert set(UIIS_TO_SUIM_CLASS) == set(range(1, 8))
    assert set(UIIS_TO_SUIM_CLASS.values()) == set(range(1, 8))


def test_rasterization_uses_fixed_smaller_instance_overlap_priority():
    annotations = [
        {"id": 1, "category_id": 1, "area": 100, "segmentation": [[0, 0, 9, 0, 9, 9, 0, 9]]},
        {"id": 2, "category_id": 5, "area": 4, "segmentation": [[4, 4, 6, 4, 6, 6, 4, 6]]},
    ]
    mask, report = rasterize_semantic_mask(height=10, width=10, annotations=annotations)
    assert mask[1, 1] == 6
    assert mask[5, 5] == 1
    assert report["cross_class_overlap_pixels"] > 0


def test_category_validation_rejects_wrong_names():
    categories = [{"id": item, "name": "wrong"} for item in range(1, 8)]
    with pytest.raises(ValueError):
        validate_categories(categories)
