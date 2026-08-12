from __future__ import annotations

import numpy as np
import pytest
import torch

from degradations.registry import Condition
from reliability.dth_horizon import (
    HORIZON_IGNORE, apply_coupled_degradation, boundary_mask, first_failure_horizons,
    trajectory_metadata, trajectory_seed, validate_monotonic_operator_parameters,
)


def _conditions() -> list[Condition]:
    return [
        Condition("clean", "clean", 0, {}),
        Condition("color_s1", "color_attenuation", 1, {"red_scale": .75, "green_scale": .88, "blue_scale": .97, "blue_green_veil": .02}),
        Condition("color_s2", "color_attenuation", 2, {"red_scale": .52, "green_scale": .76, "blue_scale": .93, "blue_green_veil": .05}),
        Condition("color_s3", "color_attenuation", 3, {"red_scale": .32, "green_scale": .63, "blue_scale": .88, "blue_green_veil": .08}),
        Condition("turbidity_s1", "turbidity", 1, {"transmission": .88, "airlight_rgb": [.15, .48, .67], "spatial_variation": .04}),
        Condition("turbidity_s2", "turbidity", 2, {"transmission": .68, "airlight_rgb": [.18, .52, .70], "spatial_variation": .08}),
        Condition("turbidity_s3", "turbidity", 3, {"transmission": .48, "airlight_rgb": [.22, .56, .73], "spatial_variation": .12}),
        Condition("lowlight_s1", "lowlight", 1, {"exposure": .78, "gamma": 1.08}),
        Condition("lowlight_s2", "lowlight", 2, {"exposure": .56, "gamma": 1.20}),
        Condition("lowlight_s3", "lowlight", 3, {"exposure": .36, "gamma": 1.35}),
        Condition("blur_s1", "blur", 1, {"kernel_size": 3, "sigma": .8}),
        Condition("blur_s2", "blur", 2, {"kernel_size": 7, "sigma": 1.6}),
        Condition("blur_s3", "blur", 3, {"kernel_size": 11, "sigma": 2.5}),
    ]


def test_first_failure_labels_recovery_and_eligible_denominator_are_exact():
    labels = torch.tensor([[0, 0, 0, 0, 255]])
    clean = torch.tensor([[0, 0, 0, 0, 0]])
    s1 = torch.tensor([[1, 0, 0, 0, 0]])  # H1 then recovery at s2
    s2 = torch.tensor([[0, 1, 0, 0, 0]])  # H2 then recovery at s3
    s3 = torch.tensor([[0, 0, 1, 0, 0]])  # H3; final pixel censored
    result = first_failure_horizons(clean, s1, s2, s3, labels)
    assert result.horizon.tolist() == [[1, 2, 3, 4, HORIZON_IGNORE]]
    assert result.eligible.sum().item() == 4
    assert result.post_failure_recovery.tolist() == [[True, True, False, False, False]]
    assert result.s1_wrong_to_s2_correct.sum().item() == 1
    assert result.s2_wrong_to_s3_correct.sum().item() == 1


def test_coupled_turbidity_seed_is_shared_across_severities_and_pixels_stay_aligned():
    conditions = _conditions()
    image = np.arange(9 * 11 * 3, dtype=np.uint8).reshape(9, 11, 3)
    meta = trajectory_metadata("scene_7", "turbidity", conditions)
    assert trajectory_seed("scene_7", "turbidity") == meta["trajectory_seed"]
    s1, s2, s3 = [apply_coupled_degradation(image, item, sample_id="scene_7", family="turbidity") for item in conditions[4:7]]
    assert s1.shape == s2.shape == s3.shape == image.shape
    assert s1.dtype == s2.dtype == s3.dtype == np.uint8
    assert not np.array_equal(s1, s2) and not np.array_equal(s2, s3)
    assert trajectory_metadata("scene_7", "turbidity", conditions)["trajectory_hash"] == meta["trajectory_hash"]


def test_severity_directions_are_checked_and_invalid_registry_is_rejected():
    conditions = _conditions()
    validate_monotonic_operator_parameters(conditions)
    broken = list(conditions)
    broken[11] = Condition("blur_s2", "blur", 2, {"kernel_size": 7, "sigma": .4})
    with pytest.raises(ValueError, match="Blur sigma"):
        validate_monotonic_operator_parameters(broken)


def test_boundary_mask_and_horizon_shapes_preserve_pixel_alignment():
    labels = torch.tensor([[0, 0, 1], [0, 1, 1], [2, 2, 2]])
    boundary = boundary_mask(labels, radius=1)
    prediction = labels.clone()
    result = first_failure_horizons(prediction, prediction, prediction, prediction, labels)
    assert boundary.shape == labels.shape == result.horizon.shape
    assert result.eligible.all() and result.horizon.eq(4).all()
