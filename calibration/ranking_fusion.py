"""Calibration-only logistic fusion for frozen uncertainty features."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = ("calibrated_msp", "local_disagreement", "probability_margin")


@dataclass(frozen=True)
class FusionParameters:
    feature_names: tuple[str, ...]
    mean: list[float]
    scale: list[float]
    coefficients: list[float]
    intercept: float
    sample_count: int
    error_rate: float
    c: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def fit_logistic_fusion(features: np.ndarray, errors: np.ndarray, *, c: float = 1.0, seed: int = 20260722) -> tuple[StandardScaler, LogisticRegression, FusionParameters]:
    values = np.asarray(features, dtype=np.float32)
    targets = np.asarray(errors, dtype=np.uint8).reshape(-1)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES) or len(values) != len(targets):
        raise ValueError("Fusion features must be [N, 3] and aligned with binary error labels.")
    if not np.isfinite(values).all() or targets.min() < 0 or targets.max() > 1 or targets.min() == targets.max():
        raise ValueError("Fusion fitting requires finite features and both correct/error samples.")
    scaler = StandardScaler()
    normalized = scaler.fit_transform(values)
    classifier = LogisticRegression(C=c, solver="lbfgs", max_iter=200, random_state=seed)
    classifier.fit(normalized, targets)
    parameters = FusionParameters(
        feature_names=FEATURE_NAMES, mean=scaler.mean_.astype(float).tolist(), scale=scaler.scale_.astype(float).tolist(),
        coefficients=classifier.coef_[0].astype(float).tolist(), intercept=float(classifier.intercept_[0]),
        sample_count=len(targets), error_rate=float(targets.mean()), c=float(c),
    )
    return scaler, classifier, parameters


def fusion_score(features: np.ndarray, scaler: StandardScaler, classifier: LogisticRegression) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES) or not np.isfinite(values).all():
        raise ValueError("Fusion scoring requires finite [N, 3] features.")
    return classifier.predict_proba(scaler.transform(values))[:, 1].astype(np.float32)
