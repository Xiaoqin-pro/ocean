"""Tie-aware uncertainty ranking metrics for frozen segmentation logits."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.special import digamma


SCORE_NAMES = (
    "raw_msp", "calibrated_msp", "entropy", "probability_margin",
    "logit_margin", "energy", "local_disagreement",
)


@dataclass(frozen=True)
class ScoreGroups:
    widths: np.ndarray
    errors: np.ndarray
    unique_score_count: int
    largest_tie_group: int
    tied_pixel_fraction: float
    score_dtype: str
    sort_seconds: float


def uncertainty_scores(logits: torch.Tensor, *, temperature: float) -> dict[str, torch.Tensor]:
    """Return float32 scores where larger values mean greater error uncertainty."""
    if logits.ndim != 4 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Expected [N,C,H,W] logits and a finite positive temperature.")
    logits = logits.float()
    raw_probabilities = torch.softmax(logits, dim=1)
    scaled_logits = logits / temperature
    probabilities = torch.softmax(scaled_logits, dim=1)
    top_probabilities = probabilities.topk(2, dim=1).values
    top_logits = scaled_logits.topk(2, dim=1).values
    prediction = scaled_logits.argmax(dim=1)
    one_hot = functional.one_hot(prediction, num_classes=logits.shape[1]).movedim(-1, 1).to(logits.dtype)
    local_class_fraction = functional.avg_pool2d(functional.pad(one_hot, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
    predicted_support = local_class_fraction.gather(1, prediction.unsqueeze(1)).squeeze(1)
    return {
        "raw_msp": 1.0 - raw_probabilities.max(dim=1).values,
        "calibrated_msp": 1.0 - probabilities.max(dim=1).values,
        "entropy": -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).eps).log()).sum(dim=1),
        "probability_margin": 1.0 - (top_probabilities[:, 0] - top_probabilities[:, 1]),
        "logit_margin": -(top_logits[:, 0] - top_logits[:, 1]),
        "energy": -torch.logsumexp(scaled_logits, dim=1),
        "local_disagreement": 1.0 - predicted_support,
    }


def _groups_from_sorted(sorted_scores: np.ndarray, sorted_errors: np.ndarray, *, score_dtype: str, sort_seconds: float) -> ScoreGroups:
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    widths = np.diff(np.r_[starts, len(sorted_scores)]).astype(np.int64, copy=False)
    errors = np.add.reduceat(sorted_errors.astype(np.float64, copy=False), starts)
    tied = widths[widths > 1].sum()
    return ScoreGroups(widths=widths, errors=errors, unique_score_count=len(widths), largest_tie_group=int(widths.max()), tied_pixel_fraction=float(tied / len(sorted_scores)), score_dtype=score_dtype, sort_seconds=sort_seconds)


def score_groups(scores: np.ndarray, errors: np.ndarray, *, discrete_histogram: bool = False) -> ScoreGroups:
    """Aggregate a score into ascending tie groups without raster-order dependence."""
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    errors = np.asarray(errors, dtype=bool).reshape(-1)
    if len(values) == 0 or len(values) != len(errors) or not np.isfinite(values).all():
        raise ValueError("Scores must be finite and aligned with a non-empty error vector.")
    started = time.perf_counter()
    if discrete_histogram:
        # A 3x3 replicated neighbourhood gives support k/9, hence disagreement
        # levels 0..8/9. Bincount avoids an O(n log n) pixel sort.
        levels = np.rint(values * 9.0).astype(np.int16)
        if np.any(levels < 0) or np.any(levels > 9) or not np.allclose(values, levels / 9.0, atol=2e-6):
            raise ValueError("local_disagreement must be a 3x3 discrete histogram score.")
        counts = np.bincount(levels, minlength=10)
        group_errors = np.bincount(levels, weights=errors.astype(np.float64), minlength=10)
        chosen = counts > 0
        return ScoreGroups(widths=counts[chosen].astype(np.int64), errors=group_errors[chosen], unique_score_count=int(chosen.sum()), largest_tie_group=int(counts.max()), tied_pixel_fraction=float(counts[counts > 1].sum() / len(values)), score_dtype=str(values.dtype), sort_seconds=time.perf_counter() - started)
    order = np.argsort(values, kind="quicksort")
    return _groups_from_sorted(values[order], errors[order], score_dtype=str(values.dtype), sort_seconds=time.perf_counter() - started)


def _aurc_from_groups(widths: np.ndarray, errors: np.ndarray) -> float:
    """Tie-aware expected AURC, integrated without allocating a per-pixel risk array."""
    counts_before = np.r_[0, np.cumsum(widths[:-1], dtype=np.float64)]
    errors_before = np.r_[0.0, np.cumsum(errors[:-1], dtype=np.float64)]
    widths_float = widths.astype(np.float64)
    slopes = errors / widths_float
    reciprocal_sums = digamma(counts_before + widths_float + 1.0) - digamma(counts_before + 1.0)
    risk_sums = widths_float * slopes + (errors_before - slopes * counts_before) * reciprocal_sums
    first_risk = (errors[0] / widths_float[0])
    total_count = float(widths.sum())
    total_errors = float(errors.sum())
    last_risk = total_errors / total_count
    # np.trapezoid(risk, coverage), with coverage = 1/n,...,1.
    return float((risk_sums.sum() - 0.5 * (first_risk + last_risk)) / total_count)


def _oracle_aurc(count: int, errors: int) -> float:
    correct = count - errors
    widths = np.array([width for width in (correct, errors) if width > 0], dtype=np.int64)
    group_errors = np.array([value for width, value in ((correct, 0.0), (errors, float(errors))) if width > 0], dtype=np.float64)
    return _aurc_from_groups(widths, group_errors)


def _selection_errors(groups: ScoreGroups, kept_lowest: int) -> float:
    """Expected errors selected by keeping exactly k low-uncertainty pixels."""
    count = int(groups.widths.sum())
    if kept_lowest <= 0:
        return 0.0
    if kept_lowest >= count:
        return float(groups.errors.sum())
    cumulative_width = np.cumsum(groups.widths)
    group = int(np.searchsorted(cumulative_width, kept_lowest, side="left"))
    before_width = int(cumulative_width[group - 1]) if group else 0
    before_errors = float(groups.errors[:group].sum())
    take = kept_lowest - before_width
    return before_errors + take * float(groups.errors[group]) / float(groups.widths[group])


def _auroc(groups: ScoreGroups) -> float:
    positives = float(groups.errors.sum())
    count = float(groups.widths.sum())
    negatives = count - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    before = np.r_[0.0, np.cumsum(groups.widths[:-1], dtype=np.float64)]
    average_rank = before + (groups.widths.astype(np.float64) + 1.0) * 0.5
    rank_sum = float((groups.errors * average_rank).sum())
    return float((rank_sum - positives * (positives + 1.0) * 0.5) / (positives * negatives))


def _average_precision(groups: ScoreGroups) -> float:
    positives = float(groups.errors.sum())
    if positives == 0:
        return float("nan")
    errors = groups.errors[::-1]
    widths = groups.widths[::-1].astype(np.float64)
    precision = np.cumsum(errors) / np.cumsum(widths)
    return float((errors / positives * precision).sum())


def metrics_from_groups(groups: ScoreGroups, *, coverages: tuple[float, ...] = (0.9, 0.8, 0.7), top_fractions: tuple[float, ...] = (0.05, 0.1, 0.2)) -> dict[str, float]:
    """Tie-aware ranking metrics; all cutoff metrics use expected group selection."""
    started = time.perf_counter()
    count, error_count = int(groups.widths.sum()), int(round(groups.errors.sum()))
    aurc = _aurc_from_groups(groups.widths, groups.errors)
    oracle = _oracle_aurc(count, error_count)
    result: dict[str, float] = {
        "pixels": float(count), "error_rate": error_count / count,
        "error_auroc": _auroc(groups), "error_auprc": _average_precision(groups),
        "aurc": aurc, "oracle_aurc": oracle, "eaurc": aurc - oracle,
        "unique_score_count": float(groups.unique_score_count), "tie_group_count": float(np.count_nonzero(groups.widths > 1)),
        "largest_tie_group": float(groups.largest_tie_group), "tied_pixel_fraction": groups.tied_pixel_fraction,
        "score_dtype": groups.score_dtype, "sort_seconds": groups.sort_seconds, "metric_seconds": time.perf_counter() - started,
    }
    for coverage in coverages:
        kept = max(1, int(np.ceil(count * coverage)))
        result[f"risk_at_{int(coverage * 100)}_coverage"] = _selection_errors(groups, kept) / kept
    for fraction in top_fractions:
        selected = max(1, int(np.ceil(count * fraction)))
        selected_errors = error_count - _selection_errors(groups, count - selected)
        result[f"top_{int(fraction * 100)}_uncertainty_precision"] = selected_errors / selected
        result[f"top_{int(fraction * 100)}_uncertainty_recall"] = selected_errors / error_count if error_count else float("nan")
    return result


def ranking_metrics(scores: np.ndarray, errors: np.ndarray, *, discrete_histogram: bool = False, coverages: tuple[float, ...] = (0.9, 0.8, 0.7), top_fractions: tuple[float, ...] = (0.05, 0.1, 0.2)) -> dict[str, float]:
    return metrics_from_groups(score_groups(scores, errors, discrete_histogram=discrete_histogram), coverages=coverages, top_fractions=top_fractions)


def ranking_metrics_by_region(scores: np.ndarray, errors: np.ndarray, regions: dict[str, np.ndarray], *, discrete_histogram: bool = False, coverages: tuple[float, ...] = (0.9, 0.8, 0.7), top_fractions: tuple[float, ...] = (0.05, 0.1, 0.2)) -> dict[str, dict[str, float]]:
    """Evaluate regions from one sort, or from direct histograms for local disagreement."""
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    errors = np.asarray(errors, dtype=bool).reshape(-1)
    if len(values) == 0 or len(values) != len(errors) or not np.isfinite(values).all():
        raise ValueError("Scores must be finite and aligned with a non-empty error vector.")
    if discrete_histogram:
        return {name: ranking_metrics(values[np.asarray(mask, dtype=bool).reshape(-1)], errors[np.asarray(mask, dtype=bool).reshape(-1)], discrete_histogram=True, coverages=coverages, top_fractions=top_fractions) for name, mask in regions.items()}
    started = time.perf_counter()
    order = np.argsort(values, kind="quicksort")
    sorted_values, sorted_errors = values[order], errors[order]
    sort_seconds = time.perf_counter() - started
    output: dict[str, dict[str, float]] = {}
    for name, mask in regions.items():
        chosen = np.asarray(mask, dtype=bool).reshape(-1)[order]
        if not chosen.any():
            raise ValueError(f"Region {name} has no valid pixels.")
        groups = _groups_from_sorted(sorted_values[chosen], sorted_errors[chosen], score_dtype=str(values.dtype), sort_seconds=sort_seconds)
        output[name] = metrics_from_groups(groups, coverages=coverages, top_fractions=top_fractions)
    return output
