"""Small self-contained low-rank wrappers for the frozen UIIS-F4 baseline."""
from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        self.to(device=base.weight.device, dtype=base.weight.dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        delta = (inputs @ self.lora_A.t()) @ self.lora_B.t()
        return self.base(inputs) + self.scaling * delta


def replace_lora_modules(model: nn.Module, rank: int = 4, alpha: float | None = None, targets: Iterable[str] = ("q_proj", "v_proj")) -> list[str]:
    alpha = float(rank if alpha is None else alpha)
    replaced: list[str] = []
    target_set = set(targets)

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and name in target_set:
                setattr(parent, name, LoRALinear(child, rank=rank, alpha=alpha))
                replaced.append(full_name)
            else:
                visit(child, full_name)

    visit(model)
    if not replaced:
        raise RuntimeError("No target Linear modules were replaced by LoRA.")
    return replaced


def lora_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]
