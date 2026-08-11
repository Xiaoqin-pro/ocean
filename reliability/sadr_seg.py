"""Semantic-aware degradation restoration front-end."""
from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


class SADRFrontEnd(nn.Module):
    """Zero-initialized residual image rectifier with bounded output."""

    def __init__(self, channels: int = 3, hidden: int = 32, gated: bool = False) -> None:
        super().__init__()
        self.gated = gated
        self.body = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 3, padding=1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)
        self.scale = nn.Parameter(torch.tensor(0.10))
        if gated:
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, 8, 1),
                nn.GELU(),
                nn.Conv2d(8, 1, 1),
            )
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, -1.5)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = torch.tanh(self.body(image)) * self.scale.clamp(0.01, 0.50)
        if self.gated:
            residual = residual * torch.sigmoid(self.gate(image))
        return image + residual, residual


def segmentation_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    resized = functional.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
    return functional.cross_entropy(resized, labels, ignore_index=255)


def confidence_distillation(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    student_log = functional.log_softmax(student.float(), dim=1)
    teacher_prob = functional.softmax(teacher.float(), dim=1).detach()
    confidence = teacher_prob.max(1, keepdim=True).values
    value = functional.kl_div(student_log, teacher_prob, reduction="none").sum(1, keepdim=True)
    return (value * confidence).mean()
