"""Image-clustered conformal risk control.

The finite-sample correction follows the official CRC implementation:
https://github.com/aangelopoulos/conformal-risk/blob/main/core/get_lhat.py
The project exposes the parameter direction explicitly because coverage makes
our monotone loss *increase*, whereas many CRC examples use the opposite
direction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CRCSelection:
    coverage: float
    index: int
    empirical_risk: float
    corrected_risk: float
    alpha: float
    sample_count: int
    monotonic_direction: str


def validate_loss_table(losses: np.ndarray, coverages: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(losses, dtype=np.float64)
    grid = np.asarray(coverages, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != len(grid):
        raise ValueError("Loss table must have shape [independent_clusters, coverages].")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("CRC losses must be finite and bounded in [0, 1].")
    if len(grid) == 0 or np.any((grid <= 0) | (grid > 1)) or np.any(np.diff(grid) <= 0):
        raise ValueError("Coverages must be strictly increasing in (0, 1].")
    if np.any(np.diff(values, axis=1) < -1e-10):
        raise ValueError("CRC requires a non-decreasing loss curve for every cluster.")
    return values, grid


def select_crc_coverage(losses: np.ndarray, coverages: np.ndarray, alpha: float, *, bound: float = 1.0, monotonic_direction: str = "increasing") -> CRCSelection:
    """Select the largest safe coverage using the official CRC correction.

    The official code assumes an ordered loss table and uses
    ``n/(n+1)*mean_loss + B/(n+1)``.  Here a larger coverage has larger loss,
    so the first corrected-risk violation determines the previous coverage.
    If no grid point violates, the largest coverage is safely selected.
    """
    if not (0.0 < alpha < 1.0 and 0.0 < bound <= 1.0):
        raise ValueError("alpha and bound must lie in (0, 1].")
    if monotonic_direction != "increasing":
        raise ValueError("DARC-Seg coverage losses must declare monotonic_direction='increasing'.")
    values, grid = validate_loss_table(losses, coverages)
    count = values.shape[0]
    empirical = values.mean(axis=0)
    corrected = count / (count + 1.0) * empirical + bound / (count + 1.0)
    violations = np.flatnonzero(corrected >= alpha)
    index = len(grid) - 1 if len(violations) == 0 else max(int(violations[0]) - 1, 0)
    return CRCSelection(float(grid[index]), index, float(empirical[index]), float(corrected[index]), float(alpha), int(count), monotonic_direction)


def select_naive_coverage(losses: np.ndarray, coverages: np.ndarray, alpha: float) -> CRCSelection:
    """Weak empirical selector without the CRC finite-sample correction."""
    values, grid = validate_loss_table(losses, coverages)
    empirical = values.mean(axis=0)
    safe = np.flatnonzero(empirical <= alpha)
    index = int(safe[-1]) if len(safe) else 0
    return CRCSelection(float(grid[index]), index, float(empirical[index]), float(empirical[index]), float(alpha), int(values.shape[0]), "increasing")

