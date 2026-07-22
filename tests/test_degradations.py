from pathlib import Path

import numpy as np

from degradations.registry import build_image_degradation, load_conditions


ROOT = Path(__file__).resolve().parents[1]


def test_registry_has_the_fixed_pilot_conditions() -> None:
    conditions = load_conditions(ROOT / "configs" / "degradation_pilot.yaml")
    assert len(conditions) == 13
    assert [item.name for item in conditions] == [
        "clean", "color_s1", "color_s2", "color_s3", "turbidity_s1", "turbidity_s2", "turbidity_s3",
        "lowlight_s1", "lowlight_s2", "lowlight_s3", "blur_s1", "blur_s2", "blur_s3",
    ]


def test_degradations_are_image_only_and_deterministic() -> None:
    image = np.arange(9 * 11 * 3, dtype=np.uint8).reshape(9, 11, 3)
    for condition in load_conditions(ROOT / "configs" / "degradation_pilot.yaml"):
        degradation = build_image_degradation(condition)
        first = degradation(image, "sample_001")
        second = degradation(image, "sample_001")
        assert first.dtype == np.uint8
        assert first.shape == image.shape
        assert np.array_equal(first, second)
