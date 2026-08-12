"""Uncertainty-weighted semantic trajectory distillation losses."""
from __future__ import annotations

import torch
import torch.nn.functional as functional


def semantic_trajectory_distillation_loss(
    clean_logits: torch.Tensor,
    degraded_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 2.0,
    confidence_power: float = 1.0,
    class_balance: bool = False,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Distill high-confidence clean semantics into a degraded trajectory view."""
    if clean_logits.ndim != 4 or degraded_logits.ndim != 4:
        raise ValueError("Trajectory logits must be four-dimensional tensors.")
    if clean_logits.shape[0] != degraded_logits.shape[0]:
        raise ValueError("Clean and degraded batches must have equal size.")
    if clean_logits.shape[-2:] != degraded_logits.shape[-2:]:
        clean_logits = functional.interpolate(clean_logits, size=degraded_logits.shape[-2:], mode="bilinear", align_corners=False)
    scale = float(temperature)
    teacher = functional.softmax(clean_logits.detach().float() / scale, dim=1)
    student_log = functional.log_softmax(degraded_logits.float() / scale, dim=1)
    confidence = teacher.max(dim=1).values.detach().pow(float(confidence_power))
    valid = functional.interpolate(labels.float().unsqueeze(1), size=degraded_logits.shape[-2:], mode="nearest")[:, 0].ne(ignore_index)
    if not bool(valid.any()):
        return degraded_logits.sum() * 0.0
    pixel_weight = confidence
    if class_balance:
        native_labels = functional.interpolate(labels.float().unsqueeze(1), size=degraded_logits.shape[-2:], mode="nearest")[:, 0].long()
        classes = int(degraded_logits.shape[1]); counts = torch.bincount(native_labels[valid].clamp_min(0), minlength=classes).float().clamp_min(1.0)
        inverse = counts.rsqrt(); inverse = inverse / inverse.mean().clamp_min(1e-6)
        pixel_weight = pixel_weight * inverse[native_labels.clamp_min(0)]
    per_pixel = functional.kl_div(student_log, teacher, reduction="none").sum(1) * pixel_weight
    normalizer = pixel_weight[valid].sum().clamp_min(1e-6)
    return per_pixel[valid].sum() / normalizer * (scale * scale)


__all__ = ["semantic_trajectory_distillation_loss"]
