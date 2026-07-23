"""Tie-aware per-image selective-risk curves.

The acceptance policy keeps the lowest-uncertainty pixels.  At a score tie
that crosses a requested coverage, the reported risk is the tie-independent
expected risk, rather than an arbitrary raster-order choice.
"""
from __future__ import annotations

import numpy as np

from metrics.uncertainty_ranking import score_groups


def coverage_grid(start: float = 0.01, stop: float = 1.0, step: float = 0.01) -> np.ndarray:
    if not (0.0 < start <= stop <= 1.0 and step > 0.0):
        raise ValueError("Coverage grid must lie in (0, 1].")
    grid = np.arange(start, stop + step * 0.25, step, dtype=np.float64)
    grid[-1] = min(grid[-1], stop)
    if not np.isclose(grid[-1], stop):
        grid = np.append(grid, stop)
    return np.round(grid, 10)


def selective_risk_curve(scores: np.ndarray, errors: np.ndarray, coverages: np.ndarray) -> np.ndarray:
    """Return expected accepted-pixel error rate for each coverage.

    ``scores`` is uncertainty, so lower values are accepted first.  The
    implementation is invariant to pixel order, including at ties.
    """
    coverages = np.asarray(coverages, dtype=np.float64)
    if coverages.ndim != 1 or len(coverages) == 0 or np.any((coverages <= 0) | (coverages > 1)):
        raise ValueError("Coverages must be a non-empty vector in (0, 1].")
    groups = score_groups(scores, errors)
    widths = groups.widths.astype(np.float64)
    group_errors = groups.errors.astype(np.float64)
    count = int(widths.sum())
    cumulative_width = np.cumsum(widths)
    cumulative_errors = np.cumsum(group_errors)
    risks = np.empty(len(coverages), dtype=np.float64)
    for index, coverage in enumerate(coverages):
        accepted = max(1, int(np.ceil(count * coverage)))
        group = int(np.searchsorted(cumulative_width, accepted, side="left"))
        before_width = cumulative_width[group - 1] if group else 0.0
        before_errors = cumulative_errors[group - 1] if group else 0.0
        take = accepted - before_width
        expected_errors = before_errors + take * group_errors[group] / widths[group]
        risks[index] = expected_errors / accepted
    return risks


def monotone_envelope(risks: np.ndarray) -> np.ndarray:
    """Return L(c)=max_{c'<=c} r(c'), the CRC-compatible loss curve."""
    risks = np.asarray(risks, dtype=np.float64)
    if risks.ndim != 1 or len(risks) == 0 or not np.isfinite(risks).all() or np.any((risks < 0) | (risks > 1)):
        raise ValueError("Risks must be a finite non-empty vector in [0, 1].")
    return np.maximum.accumulate(risks)


def curve_summary(scores: np.ndarray, errors: np.ndarray, coverages: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    risks = selective_risk_curve(scores, errors, coverages)
    return risks, monotone_envelope(risks)

