from __future__ import annotations

from typing import Any

import torch


def confusion_matrix(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Return a matrix whose rows are ground truth and columns predictions."""
    prediction = prediction.detach().to(torch.long).reshape(-1)
    target = target.detach().to(torch.long).reshape(-1)
    if prediction.numel() != target.numel():
        raise ValueError("Prediction and target must contain the same number of pixels.")
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    valid &= (prediction >= 0) & (prediction < num_classes)
    encoded = target[valid] * num_classes + prediction[valid]
    return torch.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def metrics_from_confusion_matrix(matrix: torch.Tensor) -> dict[str, Any]:
    """Compute metrics; classes absent from both target and prediction are NaN and excluded from means."""
    matrix = matrix.to(torch.float64)
    true_count = matrix.sum(dim=1)
    predicted_count = matrix.sum(dim=0)
    true_positive = matrix.diag()
    total = matrix.sum()
    pixel_accuracy = true_positive.sum() / total if total > 0 else torch.tensor(float("nan"))

    class_accuracy = torch.where(true_count > 0, true_positive / true_count, torch.nan)
    union = true_count + predicted_count - true_positive
    iou = torch.where(union > 0, true_positive / union, torch.nan)
    dice_denominator = true_count + predicted_count
    dice = torch.where(dice_denominator > 0, 2 * true_positive / dice_denominator, torch.nan)
    return {
        "confusion_matrix": matrix.to(torch.long),
        "pixel_accuracy": float(pixel_accuracy),
        "mean_accuracy": float(torch.nanmean(class_accuracy)),
        "per_class_accuracy": class_accuracy.tolist(),
        "per_class_iou": iou.tolist(),
        "miou": float(torch.nanmean(iou)),
        "per_class_dice": dice.tolist(),
        "mean_dice": float(torch.nanmean(dice)),
    }


def segmentation_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int = 255,
) -> dict[str, Any]:
    return metrics_from_confusion_matrix(
        confusion_matrix(prediction, target, num_classes=num_classes, ignore_index=ignore_index)
    )
