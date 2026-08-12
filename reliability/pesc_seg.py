"""Paired enhancement semantic consistency losses."""
from __future__ import annotations

import torch
import torch.nn.functional as functional


def paired_enhancement_consistency_loss(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    *,
    temperature: float = 2.0,
    confidence_threshold: float = 0.55,
) -> tuple[torch.Tensor, float]:
    """Symmetric confidence-gated KL between raw and enhanced predictions."""
    if first_logits.shape != second_logits.shape:
        raise ValueError("Paired logits must have equal shapes.")
    scale = float(temperature)
    first_prob = functional.softmax(first_logits.detach() / scale, dim=1)
    second_prob = functional.softmax(second_logits.detach() / scale, dim=1)
    first_log = functional.log_softmax(first_logits / scale, dim=1)
    second_log = functional.log_softmax(second_logits / scale, dim=1)
    confidence = 0.5 * (first_prob.max(1).values + second_prob.max(1).values)
    mask = confidence.ge(float(confidence_threshold))
    forward = functional.kl_div(first_log, second_prob, reduction="none").sum(1)
    backward = functional.kl_div(second_log, first_prob, reduction="none").sum(1)
    value = 0.5 * (forward + backward)
    if not bool(mask.any()):
        return value.mean() * 0.0, 0.0
    return value[mask].mean() * (scale * scale), float(mask.float().mean())
