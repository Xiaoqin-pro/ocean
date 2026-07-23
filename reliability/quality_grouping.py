"""Frozen train-only standardization and KMeans quality grouping."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class FrozenQualityGrouping:
    mean: np.ndarray
    scale: np.ndarray
    centers: np.ndarray
    seed: int

    def predict(self, descriptors: np.ndarray) -> np.ndarray:
        values = np.asarray(descriptors, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.mean) or not np.isfinite(values).all():
            raise ValueError("Descriptor matrix has invalid shape or non-finite values.")
        standardized = (values - self.mean) / self.scale
        distances = ((standardized[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1).astype(np.int64)


def fit_quality_grouping(train_descriptors: np.ndarray, *, groups: int, seed: int) -> FrozenQualityGrouping:
    values = np.asarray(train_descriptors, dtype=np.float64)
    if values.ndim != 2 or len(values) < groups or groups < 2 or not np.isfinite(values).all():
        raise ValueError("Need finite train-only descriptors for at least two quality groups.")
    scaler = StandardScaler().fit(values)
    model = KMeans(n_clusters=groups, random_state=seed, n_init=20, algorithm="lloyd").fit(scaler.transform(values))
    return FrozenQualityGrouping(scaler.mean_.astype(np.float64), scaler.scale_.astype(np.float64), model.cluster_centers_.astype(np.float64), int(seed))

