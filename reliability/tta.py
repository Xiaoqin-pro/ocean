"""Unsupervised test-time adaptation losses."""
from __future__ import annotations

import torch
import torch.nn.functional as functional


def confident_pseudo_label_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, threshold: float = 0.7, temperature: float = 1.0) -> tuple[torch.Tensor, float]:
    """Confidence-masked pseudo-label CE with a detached teacher."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("TTA logits must have the same shape.")
    teacher_prob = functional.softmax(teacher_logits.detach() / float(temperature), dim=1)
    confidence, pseudo = teacher_prob.max(1)
    mask = confidence.ge(float(threshold))
    if not bool(mask.any()):
        return student_logits.sum() * 0.0, 0.0
    loss = functional.cross_entropy(student_logits, pseudo, reduction="none")
    return loss[mask].mean(), float(mask.float().mean())


def prediction_entropy(logits: torch.Tensor) -> torch.Tensor:
    probabilities = functional.softmax(logits, dim=1)
    return -(probabilities * probabilities.clamp_min(1e-6).log()).sum(1).mean()
