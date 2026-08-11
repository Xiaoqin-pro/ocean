"""Degradation-Adaptive Prototype Transport (DAPT) modules."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from reliability.spt_seg import semantic_prototype_transport_loss


@dataclass
class DAPTOutput:
    logits: torch.Tensor
    raw_features: tuple[torch.Tensor, ...]
    corrected_features: tuple[torch.Tensor, ...]
    residuals: tuple[torch.Tensor, ...]
    router_logits: tuple[torch.Tensor, ...]
    prototypes: tuple[torch.Tensor, ...]
    severity_scores: tuple[torch.Tensor, ...]


class DegradationAdaptivePrototypeRectifier(nn.Module):
    """Class-prototype experts modulated by a global degradation token."""

    def __init__(self, channels: int, num_classes: int = 8, reduction: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        hidden = max(8, channels // reduction)
        token_hidden = max(8, channels // 4)
        self.norm = nn.GroupNorm(1, channels, affine=False)
        self.down = nn.Conv2d(channels, hidden, 1)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.expert_up = nn.Conv2d(hidden, num_classes * channels, 1)
        self.router = nn.Conv2d(channels, num_classes, 1)
        self.prototype_query = nn.Conv2d(channels, channels, 1, bias=False)
        self.prototypes = nn.Parameter(torch.randn(num_classes, channels) * 0.02)
        self.token_encoder = nn.Sequential(nn.Linear(channels, token_hidden), nn.GELU())
        self.token_scale = nn.Linear(token_hidden, hidden)
        self.token_shift = nn.Linear(token_hidden, hidden)
        self.severity_head = nn.Linear(token_hidden, 1)
        nn.init.zeros_(self.expert_up.weight)
        nn.init.zeros_(self.expert_up.bias)
        nn.init.zeros_(self.token_scale.weight)
        nn.init.zeros_(self.token_scale.bias)
        nn.init.zeros_(self.token_shift.weight)
        nn.init.zeros_(self.token_shift.bias)

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.norm(feature)
        pooled = normalized.mean(dim=(-2, -1))
        token = self.token_encoder(pooled)
        latent = functional.gelu(self.down(normalized))
        latent = functional.gelu(self.depthwise(latent))
        scale = 0.25 * torch.tanh(self.token_scale(token)).unsqueeze(-1).unsqueeze(-1)
        shift = 0.25 * torch.tanh(self.token_shift(token)).unsqueeze(-1).unsqueeze(-1)
        latent = latent * (1.0 + scale) + shift
        batch, _, height, width = feature.shape
        experts = self.expert_up(latent).reshape(batch, self.num_classes, self.channels, height, width)
        local_logits = self.router(normalized)
        query = functional.normalize(self.prototype_query(normalized), dim=1)
        anchors = functional.normalize(self.prototypes, dim=1)
        similarity = torch.einsum("bchw,kc->bkhw", query, anchors)
        router_logits = local_logits + 0.25 * similarity
        probabilities = router_logits.softmax(1)
        residual = (experts * probabilities.unsqueeze(2)).sum(1)
        severity = self.severity_head(token)
        return feature + residual, residual, router_logits, self.prototypes, severity


class DAPTSegformer(nn.Module):
    def __init__(self, base_model: nn.Module, hidden_sizes: tuple[int, ...] = (32, 64, 160, 256), num_classes: int = 8, reduction: int = 4) -> None:
        super().__init__()
        self.base_model = base_model
        self.rectifiers = nn.ModuleList(DegradationAdaptivePrototypeRectifier(channels, num_classes, reduction) for channels in hidden_sizes)

    def forward(self, pixel_values: torch.Tensor) -> DAPTOutput:
        encoded = self.base_model.segformer(pixel_values, output_hidden_states=True, return_dict=True)
        raw = tuple(encoded.hidden_states)
        values = tuple(rectifier(feature) for rectifier, feature in zip(self.rectifiers, raw, strict=True))
        corrected = tuple(item[0] for item in values); residuals = tuple(item[1] for item in values); routers = tuple(item[2] for item in values); prototypes = tuple(item[3] for item in values); scores = tuple(item[4] for item in values)
        logits = self.base_model.decode_head(corrected)
        return DAPTOutput(logits, raw, corrected, residuals, routers, prototypes, scores)


def degradation_rank_loss(scores: tuple[torch.Tensor, ...], *, margin: float = 0.1) -> torch.Tensor:
    """Enforce clean < s1 < s2 < s3 using a margin ranking objective."""
    if len(scores) != 4:
        raise ValueError("DAPT expects four trajectory severity scores.")
    terms = []
    for previous, current in zip(scores[:-1], scores[1:], strict=True):
        terms.append(functional.relu(margin - (current - previous)).mean())
    return torch.stack(terms).mean()


def rectifier_parameter_count(model: DAPTSegformer) -> int:
    return sum(parameter.numel() for parameter in model.rectifiers.parameters())


__all__ = ["DAPTOutput", "DAPTSegformer", "DegradationAdaptivePrototypeRectifier", "degradation_rank_loss", "rectifier_parameter_count", "semantic_prototype_transport_loss"]
