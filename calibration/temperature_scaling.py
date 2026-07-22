"""Scalar temperature scaling fitted only with calibration logits and labels."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    initial_nll: float
    final_nll: float
    iterations: int
    converged: bool
    valid_pixels: int


def scale_logits(logits: torch.Tensor, temperature: float | torch.Tensor) -> torch.Tensor:
    value = float(temperature.detach().cpu()) if isinstance(temperature, torch.Tensor) and temperature.ndim == 0 else temperature
    if not torch.isfinite(torch.as_tensor(value)) or float(value) <= 0:
        raise ValueError("Temperature must be finite and positive.")
    return logits / temperature


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = 255,
    min_temperature: float = 0.05,
    max_temperature: float = 20.0,
    max_iter: int = 100,
) -> TemperatureFit:
    if logits.ndim != 4 or labels.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError("Expected logits [N,C,H,W] and matching labels [N,H,W].")
    valid_pixels = int(labels.ne(ignore_index).sum().item())
    if not valid_pixels:
        raise ValueError("No valid labels available for temperature fitting.")
    logits = logits.detach().float()
    labels = labels.detach().long()
    initial = float(functional.cross_entropy(logits, labels, ignore_index=ignore_index).item())
    log_temperature = torch.nn.Parameter(torch.zeros((), device=logits.device, dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")
    iterations = 0
    def closure() -> torch.Tensor:
        nonlocal iterations
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(min_temperature, max_temperature)
        loss = functional.cross_entropy(logits / temperature.to(logits.dtype), labels, ignore_index=ignore_index)
        loss.backward()
        iterations += 1
        return loss
    optimizer.step(closure)
    temperature = float(torch.exp(log_temperature).clamp(min_temperature, max_temperature).item())
    final = float(functional.cross_entropy(scale_logits(logits, temperature), labels, ignore_index=ignore_index).item())
    if not final <= initial + 1e-7:
        raise RuntimeError(f"Temperature fitting worsened NLL: {initial} -> {final}")
    return TemperatureFit(temperature, initial, final, iterations, True, valid_pixels)
