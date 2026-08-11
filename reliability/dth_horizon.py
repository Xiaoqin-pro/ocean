"""Deterministic primitives for the DTH Gate-1 conditional horizon audit.

The module is deliberately model-agnostic.  It defines labels only for pixels
that are correct on the clean image and never opens a formal split.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from degradations.blur import apply_blur
from degradations.color_attenuation import apply_color_attenuation
from degradations.lowlight import apply_lowlight
from degradations.registry import Condition
from degradations.turbidity import apply_turbidity


IGNORE_INDEX = 255
HORIZON_IGNORE = 0
HORIZON_NAMES = {1: "H1", 2: "H2", 3: "H3", 4: "censored"}
FAMILIES = ("color", "turbidity", "lowlight", "blur")
SCHEMA = "dth_coupled_horizon_trajectory_v1"


@dataclass(frozen=True)
class HorizonLabels:
    horizon: torch.Tensor
    eligible: torch.Tensor
    post_failure_recovery: torch.Tensor
    s1_wrong_to_s2_correct: torch.Tensor
    s2_wrong_to_s3_correct: torch.Tensor
    transitions: Mapping[str, torch.Tensor]


def trajectory_seed(sample_id: str, family: str) -> int:
    """A stable per-scene/family latent seed, shared by s1/s2/s3."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown degradation family: {family}")
    digest = hashlib.sha256(f"{SCHEMA}|{sample_id}|{family}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def trajectory_token(sample_id: str, family: str) -> str:
    """Token passed to spatial operators so their latent pattern cannot vary by severity."""
    return f"dth:{family}:{trajectory_seed(sample_id, family):016x}"


def _canonical_hash(value: Mapping[str, Any]) -> str:
    serialised = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest().upper()


def family_conditions(conditions: Sequence[Condition], family: str) -> tuple[Condition, Condition, Condition]:
    if family not in FAMILIES:
        raise ValueError(f"Unknown degradation family: {family}")
    wanted = [f"{family}_s1", f"{family}_s2", f"{family}_s3"]
    by_name = {condition.name: condition for condition in conditions}
    if any(name not in by_name for name in wanted):
        raise ValueError(f"Missing coupled trajectory conditions for {family}.")
    selected = tuple(by_name[name] for name in wanted)
    if tuple(condition.severity for condition in selected) != (1, 2, 3):
        raise ValueError(f"{family} trajectory severities must be exactly s1/s2/s3.")
    return selected  # type: ignore[return-value]


def validate_monotonic_operator_parameters(conditions: Sequence[Condition]) -> None:
    """Reject a registry whose declared severity directions are not ordered."""
    color = family_conditions(conditions, "color")
    if not (color[0].parameters["red_scale"] > color[1].parameters["red_scale"] > color[2].parameters["red_scale"]):
        raise ValueError("Color attenuation must increase by decreasing red_scale.")
    if not (color[0].parameters["blue_green_veil"] < color[1].parameters["blue_green_veil"] < color[2].parameters["blue_green_veil"]):
        raise ValueError("Color attenuation must increase blue_green_veil.")

    turbidity = family_conditions(conditions, "turbidity")
    if not (turbidity[0].parameters["transmission"] > turbidity[1].parameters["transmission"] > turbidity[2].parameters["transmission"]):
        raise ValueError("Turbidity must increase by decreasing transmission.")
    if not (turbidity[0].parameters["spatial_variation"] < turbidity[1].parameters["spatial_variation"] < turbidity[2].parameters["spatial_variation"]):
        raise ValueError("Turbidity spatial variation must increase with severity.")

    lowlight = family_conditions(conditions, "lowlight")
    if not (lowlight[0].parameters["exposure"] > lowlight[1].parameters["exposure"] > lowlight[2].parameters["exposure"]):
        raise ValueError("Low light must increase by decreasing exposure.")
    if not (lowlight[0].parameters["gamma"] < lowlight[1].parameters["gamma"] < lowlight[2].parameters["gamma"]):
        raise ValueError("Low-light gamma must increase with severity.")

    blur = family_conditions(conditions, "blur")
    if not (blur[0].parameters["sigma"] < blur[1].parameters["sigma"] < blur[2].parameters["sigma"]):
        raise ValueError("Blur sigma must increase with severity.")


def trajectory_metadata(sample_id: str, family: str, conditions: Sequence[Condition]) -> dict[str, Any]:
    selected = family_conditions(conditions, family)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "sample_id": str(sample_id),
        "family": family,
        "trajectory_seed": trajectory_seed(str(sample_id), family),
        "trajectory_token": trajectory_token(str(sample_id), family),
        "conditions": [{"name": condition.name, "severity": condition.severity, "parameters": condition.parameters} for condition in selected],
    }
    payload["trajectory_hash"] = _canonical_hash(payload)
    return payload


def apply_coupled_degradation(image: np.ndarray, condition: Condition, *, sample_id: str, family: str) -> np.ndarray:
    """Apply one condition using the same per-family spatial latent at all severities."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("DTH trajectories require RGB uint8 images.")
    parameters = condition.parameters
    if condition.degradation_type == "clean":
        result = image.copy()
    elif condition.degradation_type == "color_attenuation":
        result = apply_color_attenuation(image, **parameters)
    elif condition.degradation_type == "turbidity":
        result = apply_turbidity(image, trajectory_token(sample_id, family), **parameters)
    elif condition.degradation_type == "lowlight":
        result = apply_lowlight(image, **parameters)
    elif condition.degradation_type == "blur":
        result = apply_blur(image, **parameters)
    else:
        raise ValueError(f"Unsupported DTH degradation: {condition.degradation_type}")
    if result.shape != image.shape or result.dtype != np.uint8:
        raise ValueError("A coupled DTH trajectory must preserve RGB shape and uint8 type.")
    return result


def first_failure_horizons(
    clean_prediction: torch.Tensor, s1_prediction: torch.Tensor, s2_prediction: torch.Tensor,
    s3_prediction: torch.Tensor, labels: torch.Tensor, *, ignore_index: int = IGNORE_INDEX,
) -> HorizonLabels:
    """Create H1/H2/H3/censored labels while retaining non-monotonic recovery facts."""
    tensors = (clean_prediction, s1_prediction, s2_prediction, s3_prediction, labels)
    if any(value.shape != labels.shape for value in tensors) or labels.ndim < 2:
        raise ValueError("Predictions and labels must have matching spatial shapes.")
    valid = labels.ne(ignore_index)
    clean_correct = clean_prediction.eq(labels) & valid
    correct1, correct2, correct3 = s1_prediction.eq(labels) & valid, s2_prediction.eq(labels) & valid, s3_prediction.eq(labels) & valid
    horizon = torch.full_like(labels, HORIZON_IGNORE)
    horizon[clean_correct & ~correct1] = 1
    horizon[clean_correct & correct1 & ~correct2] = 2
    horizon[clean_correct & correct1 & correct2 & ~correct3] = 3
    horizon[clean_correct & correct1 & correct2 & correct3] = 4
    failed = horizon.ge(1) & horizon.le(3)
    recovery = (horizon.eq(1) & (correct2 | correct3)) | (horizon.eq(2) & correct3)
    return HorizonLabels(
        horizon=horizon,
        eligible=clean_correct,
        post_failure_recovery=recovery & failed,
        s1_wrong_to_s2_correct=clean_correct & ~correct1 & correct2,
        s2_wrong_to_s3_correct=clean_correct & ~correct2 & correct3,
        transitions={
            "c_to_w_s1": clean_correct & ~correct1,
            "w_to_c_s1": valid & ~clean_correct & correct1,
            "c_to_c_s1": clean_correct & correct1,
            "w_to_w_s1": valid & ~clean_correct & ~correct1,
            "c_to_w_s2": correct1 & ~correct2,
            "w_to_c_s2": valid & ~correct1 & correct2,
            "c_to_c_s2": correct1 & correct2,
            "w_to_w_s2": valid & ~correct1 & ~correct2,
            "c_to_w_s3": correct2 & ~correct3,
            "w_to_c_s3": valid & ~correct2 & correct3,
            "c_to_c_s3": correct2 & correct3,
            "w_to_w_s3": valid & ~correct2 & ~correct3,
        },
    )


def boundary_mask(labels: torch.Tensor, *, radius: int = 3, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Class-change boundary mask used only for Gate-1 stratification."""
    if labels.ndim != 2 or radius < 1:
        raise ValueError("Expected one [H,W] label map and a positive radius.")
    valid = labels.ne(ignore_index)
    value = labels.float().masked_fill(~valid, 0.0)[None, None]
    kernel = 2 * radius + 1
    maximum = torch.nn.functional.max_pool2d(value, kernel, stride=1, padding=radius)
    minimum = -torch.nn.functional.max_pool2d(-value, kernel, stride=1, padding=radius)
    return maximum[0, 0].ne(minimum[0, 0]) & valid


def status_counts(horizon: torch.Tensor, mask: torch.Tensor) -> dict[str, int]:
    if horizon.shape != mask.shape:
        raise ValueError("Horizon and reporting mask must align exactly.")
    return {name: int((horizon.eq(code) & mask).sum().item()) for code, name in HORIZON_NAMES.items()}
