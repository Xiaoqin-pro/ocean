"""Online semantic group-robust losses for SGRE-Seg."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


def classwise_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int = 8,
    ignore_index: int = 255,
    minimum_pixels: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return differentiable per-class losses and a class-presence mask."""
    if logits.shape[-2:] != labels.shape[-2:]:
        logits = functional.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
    pixel_loss = functional.cross_entropy(logits.float(), labels, ignore_index=ignore_index, reduction="none")
    flat_labels = labels.reshape(-1)
    valid = flat_labels.ne(ignore_index) & flat_labels.ge(0) & flat_labels.lt(num_classes)
    indices = flat_labels[valid]
    counts = torch.bincount(indices, minlength=num_classes)
    sums = pixel_loss.reshape(-1)[valid].new_zeros(num_classes)
    sums.index_add_(0, indices, pixel_loss.reshape(-1)[valid])
    return sums / counts.clamp_min(1), counts.ge(minimum_pixels)


@dataclass
class SemanticGroupState:
    ema_losses: torch.Tensor
    observed: torch.Tensor
    momentum: float
    temperature: float

    @classmethod
    def create(
        cls,
        views: int = 4,
        classes: int = 8,
        *,
        momentum: float = 0.9,
        temperature: float = 0.5,
    ) -> "SemanticGroupState":
        return cls(torch.ones(views, classes), torch.zeros(views, classes, dtype=torch.bool), momentum, temperature)

    def weights(self, device: torch.device) -> torch.Tensor:
        return (self.ema_losses / self.temperature).flatten().softmax(0).reshape_as(self.ema_losses).to(device)

    def update(self, view_index: int, losses: torch.Tensor, present: torch.Tensor) -> None:
        losses = losses.detach().float().cpu()
        present = present.detach().bool().cpu()
        first = present & ~self.observed[view_index]
        continuing = present & self.observed[view_index]
        self.ema_losses[view_index, first] = losses[first]
        self.ema_losses[view_index, continuing] = (
            self.momentum * self.ema_losses[view_index, continuing]
            + (1.0 - self.momentum) * losses[continuing]
        )
        self.observed[view_index, present] = True

    def state_dict(self) -> dict[str, object]:
        return {
            "ema_losses": self.ema_losses.clone(),
            "observed": self.observed.clone(),
            "momentum": self.momentum,
            "temperature": self.temperature,
        }
