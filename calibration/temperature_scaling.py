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
    optimizer_n_iter: int
    function_evaluations: int
    converged: bool
    valid_pixels: int
    at_boundary: bool
    nll_improvement: float
    finite_result: bool


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
    function_evaluations = 0
    def closure() -> torch.Tensor:
        nonlocal function_evaluations
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(min_temperature, max_temperature)
        loss = functional.cross_entropy(logits / temperature.to(logits.dtype), labels, ignore_index=ignore_index)
        loss.backward()
        function_evaluations += 1
        return loss
    optimizer.step(closure)
    temperature = float(torch.exp(log_temperature).clamp(min_temperature, max_temperature).item())
    final = float(functional.cross_entropy(scale_logits(logits, temperature), labels, ignore_index=ignore_index).item())
    if not final <= initial + 1e-7:
        raise RuntimeError(f"Temperature fitting worsened NLL: {initial} -> {final}")
    state = optimizer.state[log_temperature]
    optimizer_n_iter = int(state.get("n_iter", 0))
    function_evaluations = int(state.get("func_evals", function_evaluations))
    finite_result = bool(torch.isfinite(torch.tensor(temperature)) and torch.isfinite(torch.tensor(final)))
    at_boundary = abs(temperature - min_temperature) < 1e-8 or abs(temperature - max_temperature) < 1e-8
    # LBFGS exposes no numerical convergence flag.  This conservative diagnostic
    # never interprets the number of closure calls as an optimizer iteration.
    converged = bool(finite_result and final <= initial + 1e-7 and not at_boundary and optimizer_n_iter < max_iter)
    return TemperatureFit(temperature, initial, final, optimizer_n_iter, function_evaluations, converged, valid_pixels, at_boundary, initial - final, finite_result)


def fit_temperature_from_batches(batch_factory, *, ignore_index: int = 255, min_temperature: float = 0.05, max_temperature: float = 20.0, max_iter: int = 100) -> TemperatureFit:
    """Fit one scalar with a re-iterable batch factory and pixel-weighted NLL."""
    first_loss = None
    valid_pixels = 0
    for logits, labels in batch_factory():
        count = int(labels.ne(ignore_index).sum())
        if count:
            value = functional.cross_entropy(logits.float(), labels.long(), ignore_index=ignore_index, reduction="sum")
            first_loss = value if first_loss is None else first_loss + value
            valid_pixels += count
    if not valid_pixels or first_loss is None:
        raise ValueError("No valid labels available for temperature fitting.")
    initial = float((first_loss / valid_pixels).item())
    # The production cache factory yields CUDA batches.  Gradients are only
    # required for this scalar, so each batch is back-propagated immediately;
    # retaining a graph for the whole pooled calibration set would defeat the
    # purpose of streaming and can exhaust GPU memory.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parameter = torch.nn.Parameter(torch.zeros((), device=device, dtype=torch.float64))
    optimizer = torch.optim.LBFGS([parameter], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")
    function_evaluations = 0
    def closure() -> torch.Tensor:
        nonlocal function_evaluations
        optimizer.zero_grad(); total_value = 0.0; count = 0
        for logits, labels in batch_factory():
            count += int(labels.ne(ignore_index).sum())
            # Recreate this tiny graph for every batch: after ``backward`` the
            # previous graph is intentionally released to keep the fit bounded.
            temperature = torch.exp(parameter).clamp(min_temperature, max_temperature).to(torch.float32)
            loss = functional.cross_entropy(logits.float() / temperature, labels.long(), ignore_index=ignore_index, reduction="sum") / valid_pixels
            total_value += float(loss.detach())
            loss.backward()
        if count != valid_pixels:
            raise RuntimeError("The streaming batch factory changed between optimization passes.")
        function_evaluations += 1
        return torch.tensor(total_value, device=device)
    optimizer.step(closure)
    temperature = float(torch.exp(parameter).clamp(min_temperature, max_temperature).item())
    with torch.no_grad():
        total = 0.0
        for logits, labels in batch_factory():
            total += float(functional.cross_entropy(logits.float() / temperature, labels.long(), ignore_index=ignore_index, reduction="sum").item())
        final = total / valid_pixels
    if final > initial + 1e-7: raise RuntimeError(f"Temperature fitting worsened NLL: {initial} -> {final}")
    state = optimizer.state[parameter]
    optimizer_n_iter = int(state.get("n_iter", 0))
    function_evaluations = int(state.get("func_evals", function_evaluations))
    finite_result = bool(torch.isfinite(torch.tensor(temperature)) and torch.isfinite(torch.tensor(final)))
    boundary = abs(temperature-min_temperature)<1e-8 or abs(temperature-max_temperature)<1e-8
    converged = bool(finite_result and final <= initial + 1e-7 and not boundary and optimizer_n_iter < max_iter)
    return TemperatureFit(temperature, initial, final, optimizer_n_iter, function_evaluations, converged, valid_pixels, boundary, initial-final, finite_result)
