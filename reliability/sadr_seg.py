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


class FrequencySADRFrontEnd(nn.Module):
    """Task-aware rectifier with separate low/high-frequency residual paths."""

    def __init__(self, channels: int = 3, hidden: int = 24) -> None:
        super().__init__()
        self.low = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )
        self.high = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )
        self.fuse = nn.Conv2d(hidden * 2, channels, 1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)
        self.scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low = functional.avg_pool2d(image, kernel_size=3, stride=1, padding=1)
        high = image - low
        low_feature = self.low(low)
        high_feature = self.high(high)
        residual = torch.tanh(self.fuse(torch.cat([low_feature, high_feature], dim=1))) * self.scale.clamp(0.01, 0.50)
        return image + residual, residual


class RoutedSADRFrontEnd(nn.Module):
    """Soft degradation-family mixture of residual experts.

    The router is trained with the synthetic family label but only consumes
    the image at inference.  All experts start at zero, so the module is an
    identity map before optimization and cannot damage the frozen expert at
    initialization.
    """

    def __init__(self, channels: int = 3, hidden: int = 16, families: int = 4) -> None:
        super().__init__()
        self.families = families
        experts = []
        for _ in range(families):
            expert = nn.Sequential(
                nn.Conv2d(channels, hidden, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, channels, 3, padding=1),
            )
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)
            experts.append(expert)
        self.experts = nn.ModuleList(experts)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 16),
            nn.GELU(),
            nn.Linear(16, families),
        )
        self.scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.router(image)
        weights = functional.softmax(router_logits, dim=1)
        expert_residuals = torch.stack([expert(image) for expert in self.experts], dim=1)
        residual = (expert_residuals * weights[:, :, None, None, None]).sum(dim=1)
        residual = torch.tanh(residual) * self.scale.clamp(0.01, 0.50)
        return image + residual, residual, router_logits


def segmentation_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    resized = functional.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
    return functional.cross_entropy(resized, labels, ignore_index=255)


def confidence_distillation(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    student_log = functional.log_softmax(student.float(), dim=1)
    teacher_prob = functional.softmax(teacher.float(), dim=1).detach()
    confidence = teacher_prob.max(1, keepdim=True).values
    value = functional.kl_div(student_log, teacher_prob, reduction="none").sum(1, keepdim=True)
    return (value * confidence).mean()


def semantic_view_consistency(student: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Penalize semantic disagreement among the three restored severity views.

    The training loader concatenates s1/s2/s3 in that order.  Matching each
    view to the detached mean prediction keeps the loss symmetric while
    avoiding a moving teacher network.
    """
    if student.ndim != 4 or batch_size <= 0 or student.shape[0] != 3 * batch_size:
        raise ValueError("Expected logits for three views concatenated by batch.")
    probabilities = functional.softmax(student.float(), dim=1).reshape(3, batch_size, student.shape[1], student.shape[2], student.shape[3])
    target = probabilities.mean(dim=0).detach()
    target = target.unsqueeze(0).expand_as(probabilities)
    divergence = functional.kl_div(probabilities.clamp_min(1e-6).log(), target, reduction="none")
    return divergence.sum(dim=2).mean()


def gradient_reconstruction_loss(restored: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Match first-order image gradients while preserving the task objective."""
    if restored.shape != clean.shape or restored.ndim != 4:
        raise ValueError("Expected matching BCHW restored and clean tensors.")
    restored_dx = restored[..., :, 1:] - restored[..., :, :-1]
    clean_dx = clean[..., :, 1:] - clean[..., :, :-1]
    restored_dy = restored[..., 1:, :] - restored[..., :-1, :]
    clean_dy = clean[..., 1:, :] - clean[..., :-1, :]
    return (restored_dx - clean_dx).abs().mean() + (restored_dy - clean_dy).abs().mean()


def semantic_feature_consistency(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Match global semantic directions while leaving spatial details learnable."""
    if student.ndim != 4 or teacher.ndim != 4 or student.shape[0] != teacher.shape[0] or student.shape[1] != teacher.shape[1]:
        raise ValueError("Expected matching [N,C,H,W] feature tensors.")
    student_vector = functional.adaptive_avg_pool2d(student.float(), 1).flatten(1)
    teacher_vector = functional.adaptive_avg_pool2d(teacher.float(), 1).flatten(1).detach()
    similarity = functional.cosine_similarity(student_vector, teacher_vector, dim=1)
    return (1.0 - similarity).mean()


def semantic_compositionality_loss(
    composite_delta: torch.Tensor,
    first_delta: torch.Tensor,
    second_delta: torch.Tensor,
) -> torch.Tensor:
    """Match a composite semantic correction to the sum of two atomic ones.

    ``delta`` is defined in the frozen expert's logit space as the change
    induced by the trainable front-end relative to the unadapted image.  The
    atomic target is detached deliberately: the composite view receives the
    direct gradient while the two atomic corrections provide a stable local
    reference.  A normalized smooth-L1 term keeps the loss meaningful when
    correction magnitudes differ across images and avoids the scale sensitivity
    of an unnormalized logit penalty.
    """
    if composite_delta.shape != first_delta.shape or composite_delta.shape != second_delta.shape:
        raise ValueError("Compositionality deltas must have matching shapes.")
    if composite_delta.ndim < 2:
        raise ValueError("Compositionality deltas must have a batch dimension.")
    target = (first_delta.detach() + second_delta.detach()).float()
    prediction = composite_delta.float()
    scale = target.abs().mean(dim=tuple(range(1, target.ndim)), keepdim=True).clamp_min(1e-3)
    error = functional.smooth_l1_loss(prediction, target, reduction="none")
    return (error / scale).mean()


def semantic_order_consistency_loss(first_delta: torch.Tensor, second_delta: torch.Tensor) -> torch.Tensor:
    """Match semantic corrections for two application orders symmetrically."""
    if first_delta.shape != second_delta.shape or first_delta.ndim < 2:
        raise ValueError("Order-consistency deltas must have matching batched shapes.")
    first = first_delta.float()
    second = second_delta.float()
    first_flat = first.flatten(1)
    second_flat = second.flatten(1)
    cosine = 1.0 - functional.cosine_similarity(first_flat, second_flat, dim=1, eps=1e-6)
    scale = ((first.detach().abs().mean(dim=tuple(range(1, first.ndim)), keepdim=True) + second.detach().abs().mean(dim=tuple(range(1, second.ndim)), keepdim=True)) / 2.0).clamp_min(1e-3)
    forward = functional.smooth_l1_loss(first, second.detach(), reduction="none") / scale
    reverse = functional.smooth_l1_loss(second, first.detach(), reduction="none") / scale
    return cosine.mean() + 0.1 * (forward.mean() + reverse.mean()) / 2.0


def semantic_probability_order_loss(first_logits: torch.Tensor, second_logits: torch.Tensor) -> torch.Tensor:
    """Symmetric KL consistency for the two orders of a composite view."""
    if first_logits.shape != second_logits.shape or first_logits.ndim < 2:
        raise ValueError("Order-consistency logits must have matching batched shapes.")
    first_log_prob = functional.log_softmax(first_logits.float(), dim=1)
    second_log_prob = functional.log_softmax(second_logits.float(), dim=1)
    first_prob = first_log_prob.exp().detach()
    second_prob = second_log_prob.exp().detach()
    forward = functional.kl_div(first_log_prob, second_prob, reduction="none").sum(dim=1).mean()
    reverse = functional.kl_div(second_log_prob, first_prob, reduction="none").sum(dim=1).mean()
    return 0.5 * (forward + reverse)
