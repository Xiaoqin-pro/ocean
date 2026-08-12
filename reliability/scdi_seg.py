"""Semantic-conditional degradation inversion modules."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from reliability.sdtc_seg import resized_labels


@dataclass
class SCDIOutput:
    logits: torch.Tensor
    raw_features: tuple[torch.Tensor, ...]
    corrected_features: tuple[torch.Tensor, ...]
    residuals: tuple[torch.Tensor, ...]
    router_logits: tuple[torch.Tensor, ...]


class SemanticInverseExpertRectifier(nn.Module):
    """Route each location through a soft mixture of class-specific experts."""

    def __init__(self, channels: int, num_classes: int = 8, reduction: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        hidden = max(8, channels // reduction)
        self.norm = nn.GroupNorm(1, channels, affine=False)
        self.down = nn.Conv2d(channels, hidden, 1)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.expert_up = nn.Conv2d(hidden, num_classes * channels, 1)
        self.router = nn.Conv2d(channels, num_classes, 1)
        nn.init.zeros_(self.expert_up.weight)
        nn.init.zeros_(self.expert_up.bias)

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.norm(feature)
        latent = functional.gelu(self.down(normalized))
        latent = functional.gelu(self.depthwise(latent))
        batch, _, height, width = feature.shape
        experts = self.expert_up(latent).reshape(batch, self.num_classes, self.channels, height, width)
        router_logits = self.router(normalized)
        probabilities = router_logits.softmax(1)
        residual = (experts * probabilities.unsqueeze(2)).sum(1)
        return feature + residual, residual, router_logits


class SCDISegformer(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        hidden_sizes: tuple[int, ...] = (32, 64, 160, 256),
        num_classes: int = 8,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.rectifiers = nn.ModuleList(
            SemanticInverseExpertRectifier(channels, num_classes, reduction) for channels in hidden_sizes
        )

    def forward(self, pixel_values: torch.Tensor) -> SCDIOutput:
        encoded = self.base_model.segformer(pixel_values, output_hidden_states=True, return_dict=True)
        raw = tuple(encoded.hidden_states)
        values = tuple(rectifier(feature) for rectifier, feature in zip(self.rectifiers, raw, strict=True))
        corrected = tuple(item[0] for item in values)
        residuals = tuple(item[1] for item in values)
        routers = tuple(item[2] for item in values)
        logits = self.base_model.decode_head(corrected)
        return SCDIOutput(logits, raw, corrected, residuals, routers)


def semantic_router_loss(
    router_logits: tuple[torch.Tensor, ...], labels: torch.Tensor, *, ignore_index: int = 255
) -> torch.Tensor:
    if not router_logits:
        raise ValueError("SCDI requires at least one router scale.")
    losses = []
    for logits in router_logits:
        target = resized_labels(labels, logits)
        losses.append(functional.cross_entropy(logits.float(), target, ignore_index=ignore_index))
    return torch.stack(losses).mean()


def rectifier_parameter_count(model: SCDISegformer) -> int:
    return sum(parameter.numel() for parameter in model.rectifiers.parameters())
