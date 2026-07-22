"""Error-ranking scores and tie-aware selective-prediction diagnostics."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional


SCORE_NAMES = (
    "raw_msp",
    "calibrated_msp",
    "entropy",
    "probability_margin",
    "logit_margin",
    "energy",
    "local_disagreement",
)


def uncertainty_scores(logits: torch.Tensor, *, temperature: float) -> dict[str, torch.Tensor]:
    """Return scores where a larger value means a more likely segmentation error."""
    if logits.ndim != 4 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Expected [N,C,H,W] logits and a finite positive temperature.")
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


def _oracle_aurc(errors: np.ndarray) -> float:
    count = len(errors)
    oracle = np.concatenate([np.zeros(count - int(errors.sum()), dtype=np.float64), np.ones(int(errors.sum()), dtype=np.float64)])
    return _tie_aware_aurc(np.arange(count, dtype=np.float64), oracle)


def _tie_aware_aurc(sorted_scores: np.ndarray, sorted_errors: np.ndarray) -> float:
    """Expected AURC under random ordering within equal-score groups."""
    if not np.any(np.diff(sorted_scores) == 0):
        risk = np.cumsum(sorted_errors, dtype=np.float64) / np.arange(1, len(sorted_errors) + 1)
        coverage = np.arange(1, len(sorted_errors) + 1, dtype=np.float64) / len(sorted_errors)
        return float(np.trapezoid(risk, coverage))
    changes = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1, len(sorted_scores)]
    running_errors = 0.0
    total = len(sorted_scores)
    risk_parts: list[np.ndarray] = []
    for start, end in zip(changes[:-1], changes[1:]):
        width, group_errors = end - start, float(sorted_errors[start:end].sum())
        within = np.arange(1, width + 1, dtype=np.float64)
        risk = (running_errors + within * group_errors / width) / (start + within)
        risk_parts.append(risk)
        running_errors += group_errors
    all_risk = np.concatenate(risk_parts)
    coverage = np.arange(1, total + 1, dtype=np.float64) / total
    return float(np.trapezoid(all_risk, coverage))


def _auroc_from_sorted(sorted_scores: np.ndarray, sorted_errors: np.ndarray) -> float:
    positives = int(sorted_errors.sum())
    negatives = len(sorted_errors) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    if not np.any(np.diff(sorted_scores) == 0):
        rank_sum = float(np.dot(sorted_errors, np.arange(1, len(sorted_errors) + 1, dtype=np.float64)))
        return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))
    changes = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1, len(sorted_scores)]
    rank_sum = 0.0
    for start, end in zip(changes[:-1], changes[1:]):
        rank_sum += float(sorted_errors[start:end].sum()) * (start + 1 + end) / 2.0
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def _average_precision_from_sorted(sorted_scores: np.ndarray, sorted_errors: np.ndarray) -> float:
    positives = int(sorted_errors.sum())
    if positives == 0:
        return float("nan")
    sorted_scores, sorted_errors = sorted_scores[::-1], sorted_errors[::-1]
    if not np.any(np.diff(sorted_scores) == 0):
        cumulative = np.cumsum(sorted_errors, dtype=np.float64)
        positions = np.arange(1, len(sorted_errors) + 1, dtype=np.float64)
        return float((cumulative[sorted_errors.astype(bool)] / positions[sorted_errors.astype(bool)]).sum() / positives)
    changes = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1, len(sorted_scores)]
    cumulative = 0.0
    average_precision = 0.0
    for start, end in zip(changes[:-1], changes[1:]):
        group_errors = float(sorted_errors[start:end].sum())
        cumulative += group_errors
        average_precision += (group_errors / positives) * (cumulative / end)
    return float(average_precision)


def _ranking_metrics_from_sorted(ordered_scores: np.ndarray, ordered_errors: np.ndarray, *, coverages: tuple[float, ...], top_fractions: tuple[float, ...]) -> dict[str, float]:
    """Metrics from ascending uncertainty scores; internal helper avoids repeat sorts."""
    if len(ordered_scores) == 0:
        raise ValueError("Ranking metrics require at least one valid pixel.")
    ordered_errors = np.asarray(ordered_errors, dtype=np.float64)
    total_errors = int(ordered_errors.sum())
    result = {
        "pixels": float(len(ordered_errors)),
        "error_rate": float(ordered_errors.mean()),
        "error_auroc": _auroc_from_sorted(ordered_scores, ordered_errors),
        "error_auprc": _average_precision_from_sorted(ordered_scores, ordered_errors),
        "aurc": _tie_aware_aurc(ordered_scores, ordered_errors),
        "oracle_aurc": _oracle_aurc(ordered_errors),
    }
    result["eaurc"] = result["aurc"] - result["oracle_aurc"]
    for coverage in coverages:
        kept = max(1, int(np.ceil(len(ordered_errors) * coverage)))
        result[f"risk_at_{int(coverage * 100)}_coverage"] = float(ordered_errors[:kept].mean())
    for fraction in top_fractions:
        selected = max(1, int(np.ceil(len(ordered_errors) * fraction)))
        chosen = ordered_errors[-selected:]
        result[f"top_{int(fraction * 100)}_uncertainty_precision"] = float(chosen.mean())
        result[f"top_{int(fraction * 100)}_uncertainty_recall"] = float(chosen.sum() / total_errors) if total_errors else float("nan")
    return result


def ranking_metrics(scores: np.ndarray, errors: np.ndarray, *, coverages: tuple[float, ...] = (0.9, 0.8, 0.7), top_fractions: tuple[float, ...] = (0.05, 0.1, 0.2)) -> dict[str, float]:
    """Compute error-ranking metrics for uncertainty scores without ground-truth leakage."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    errors = np.asarray(errors, dtype=bool).reshape(-1)
    if len(scores) == 0 or len(scores) != len(errors) or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite and aligned with a non-empty error vector.")
    order = np.argsort(scores, kind="stable")
    return _ranking_metrics_from_sorted(scores[order], errors[order], coverages=coverages, top_fractions=top_fractions)


def ranking_metrics_by_region(scores: np.ndarray, errors: np.ndarray, regions: dict[str, np.ndarray], *, coverages: tuple[float, ...] = (0.9, 0.8, 0.7), top_fractions: tuple[float, ...] = (0.05, 0.1, 0.2)) -> dict[str, dict[str, float]]:
    """Derive full/boundary/interior metrics from one score ordering per image."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    errors = np.asarray(errors, dtype=bool).reshape(-1)
    if len(scores) == 0 or len(scores) != len(errors) or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite and aligned with a non-empty error vector.")
    order = np.argsort(scores, kind="stable")
    ordered_scores, ordered_errors = scores[order], errors[order]
    results: dict[str, dict[str, float]] = {}
    for name, mask in regions.items():
        ordered_mask = np.asarray(mask, dtype=bool).reshape(-1)[order]
        results[name] = _ranking_metrics_from_sorted(ordered_scores[ordered_mask], ordered_errors[ordered_mask], coverages=coverages, top_fractions=top_fractions)
    return results
