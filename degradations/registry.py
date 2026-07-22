"""Load fixed degradation conditions and expose image-only transforms."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from degradations.blur import apply_blur
from degradations.color_attenuation import apply_color_attenuation
from degradations.lowlight import apply_lowlight
from degradations.turbidity import apply_turbidity


@dataclass(frozen=True)
class Condition:
    name: str
    degradation_type: str
    severity: int
    parameters: dict[str, Any]


def load_conditions(config_path: Path) -> list[Condition]:
    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    rows = payload.get("conditions", [])
    conditions = [
        Condition(
            name=str(row["name"]),
            degradation_type=str(row["degradation_type"]),
            severity=int(row["severity"]),
            parameters=dict(row.get("parameters", {})),
        )
        for row in rows
    ]
    names = [condition.name for condition in conditions]
    if len(conditions) != 13 or len(names) != len(set(names)):
        raise ValueError("The degradation pilot requires exactly 13 uniquely named fixed conditions.")
    if not any(condition.name == "clean" and condition.degradation_type == "clean" and condition.severity == 0 for condition in conditions):
        raise ValueError("The degradation pilot requires one clean severity-0 condition.")
    return conditions


def build_image_degradation(condition: Condition):
    """Return a callable that changes RGB pixels only and preserves shape/dtype."""
    parameters = condition.parameters
    if condition.degradation_type == "clean":
        return lambda image, sample_id: image.copy()
    if condition.degradation_type == "color_attenuation":
        return lambda image, sample_id: apply_color_attenuation(image, **parameters)
    if condition.degradation_type == "turbidity":
        return lambda image, sample_id: apply_turbidity(image, sample_id, **parameters)
    if condition.degradation_type == "lowlight":
        return lambda image, sample_id: apply_lowlight(image, **parameters)
    if condition.degradation_type == "blur":
        return lambda image, sample_id: apply_blur(image, **parameters)
    raise ValueError(f"Unsupported degradation type: {condition.degradation_type}")


def validate_image_only(degradation, image: np.ndarray, sample_id: str) -> None:
    result = degradation(image, sample_id)
    if result.shape != image.shape or result.dtype != np.uint8:
        raise ValueError("Degradation must preserve image shape and return uint8 pixels.")
