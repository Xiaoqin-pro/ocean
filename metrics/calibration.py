from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def _flatten_probabilities(probabilities: torch.Tensor, target: torch.Tensor, ignore_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    if probabilities.ndim < 2:
        raise ValueError("Probabilities must have shape (N, C, ...).")
    classes = probabilities.shape[1]
    if probabilities.ndim == 2:
        flattened = probabilities
    else:
        flattened = probabilities.movedim(1, -1).reshape(-1, classes)
    flat_target = target.reshape(-1).to(torch.long)
    if flattened.shape[0] != flat_target.numel():
        raise ValueError("Probability and target dimensions do not match.")
    valid = (flat_target != ignore_index) & (flat_target >= 0) & (flat_target < classes)
    probabilities = flattened[valid].to(torch.float64)
    target = flat_target[valid]
    if probabilities.numel() == 0:
        raise ValueError("No valid pixels remain after ignore_index filtering.")
    if not torch.allclose(probabilities.sum(dim=1), torch.ones(len(probabilities), dtype=probabilities.dtype, device=probabilities.device), atol=1e-4):
        raise ValueError("Calibration metrics require softmax probabilities, not logits.")
    return probabilities.clamp_min(torch.finfo(probabilities.dtype).eps), target


def nll(probabilities: torch.Tensor, target: torch.Tensor, *, ignore_index: int = 255) -> float:
    probs, labels = _flatten_probabilities(probabilities, target, ignore_index)
    return float(-torch.log(probs[torch.arange(len(labels), device=labels.device), labels]).mean())


def brier_score(probabilities: torch.Tensor, target: torch.Tensor, *, ignore_index: int = 255) -> float:
    probs, labels = _flatten_probabilities(probabilities, target, ignore_index)
    one_hot = torch.nn.functional.one_hot(labels, probs.shape[1]).to(probs.dtype)
    return float(((probs - one_hot) ** 2).sum(dim=1).mean())


def expected_calibration_error(probabilities: torch.Tensor, target: torch.Tensor, *, bins: int = 15, ignore_index: int = 255) -> float:
    probs, labels = _flatten_probabilities(probabilities, target, ignore_index)
    confidence, prediction = probs.max(dim=1)
    correct = prediction.eq(labels).to(torch.float64)
    result = torch.zeros((), dtype=torch.float64, device=probs.device)
    for lower, upper in zip(torch.arange(bins, device=probs.device) / bins, torch.arange(1, bins + 1, device=probs.device) / bins):
        in_bin = (confidence > lower) & (confidence <= upper) if lower > 0 else (confidence >= lower) & (confidence <= upper)
        if in_bin.any():
            result += in_bin.to(torch.float64).mean() * (correct[in_bin].mean() - confidence[in_bin].mean()).abs()
    return float(result)


def classwise_ece(probabilities: torch.Tensor, target: torch.Tensor, *, bins: int = 15, ignore_index: int = 255) -> list[float]:
    probs, labels = _flatten_probabilities(probabilities, target, ignore_index)
    results: list[float] = []
    for class_id in range(probs.shape[1]):
        confidence = probs[:, class_id]
        event = labels.eq(class_id).to(torch.float64)
        ece = torch.zeros((), dtype=torch.float64, device=probs.device)
        for lower, upper in zip(torch.arange(bins, device=probs.device) / bins, torch.arange(1, bins + 1, device=probs.device) / bins):
            in_bin = (confidence > lower) & (confidence <= upper) if lower > 0 else (confidence >= lower) & (confidence <= upper)
            if in_bin.any():
                ece += in_bin.to(torch.float64).mean() * (event[in_bin].mean() - confidence[in_bin].mean()).abs()
        results.append(float(ece))
    return results


def error_detection_auroc(probabilities: torch.Tensor, target: torch.Tensor, *, ignore_index: int = 255) -> float:
    probs, labels = _flatten_probabilities(probabilities, target, ignore_index)
    uncertainty = 1 - probs.max(dim=1).values
    errors = probs.argmax(dim=1).ne(labels).cpu().numpy()
    if errors.min() == errors.max():
        return float("nan")
    return float(roc_auc_score(errors, uncertainty.cpu().numpy()))


def risk_coverage_curve(probabilities: torch.Tensor, target: torch.Tensor, *, ignore_index: int = 255) -> dict[str, np.ndarray]:
    probs, labels = _flatten_probabilities(probabilities, target, ignore_index)
    uncertainty = 1 - probs.max(dim=1).values
    errors = probs.argmax(dim=1).ne(labels).to(torch.float64)
    order = torch.argsort(uncertainty)  # Keep most certain predictions first.
    cumulative_risk = torch.cumsum(errors[order], dim=0) / torch.arange(1, len(errors) + 1, device=probs.device)
    coverage = torch.arange(1, len(errors) + 1, device=probs.device, dtype=torch.float64) / len(errors)
    return {"coverage": coverage.cpu().numpy(), "risk": cumulative_risk.cpu().numpy()}


def aurc(probabilities: torch.Tensor, target: torch.Tensor, *, ignore_index: int = 255) -> float:
    curve = risk_coverage_curve(probabilities, target, ignore_index=ignore_index)
    return float(np.trapezoid(curve["risk"], curve["coverage"]))
