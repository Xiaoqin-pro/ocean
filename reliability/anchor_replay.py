"""Source-anchor replay losses for cross-domain adaptation."""
from __future__ import annotations

import torch
import torch.nn.functional as functional


def source_logit_anchor_loss(student: torch.Tensor, teacher: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """Distill a frozen source model while allowing a shared adapter to learn UIIS."""
    if student.shape != teacher.shape:
        raise ValueError("Student and teacher logits must have the same shape.")
    scale = float(temperature)
    teacher_prob = functional.softmax(teacher.detach() / scale, dim=1)
    student_log_prob = functional.log_softmax(student / scale, dim=1)
    return functional.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (scale * scale)
