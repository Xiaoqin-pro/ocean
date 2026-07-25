"""Evaluate fixed scalar temperatures from cached validation/calibration logits.

This script never loads a model, image, split CSV, or official TEST data.  It
only consumes the 26 frozen cache entries created from the formal checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.label_mapping import CLASS_NAMES  # noqa: E402
from metrics.segmentation import confusion_matrix, metrics_from_confusion_matrix  # noqa: E402
from scripts.evaluate_baseline import json_safe  # noqa: E402


CONDITIONS = (
    "clean", "color_s1", "color_s2", "color_s3", "turbidity_s1", "turbidity_s2", "turbidity_s3",
    "lowlight_s1", "lowlight_s2", "lowlight_s3", "blur_s1", "blur_s2", "blur_s3",
)
SPLITS = ("calibration", "val")
METHODS = ("raw", "clean_global", "pooled", "per_degradation")
SEGMENTATION_FIELDS = ("miou", "pixel_accuracy", "mean_accuracy", "mean_dice")
PER_CLASS_FIELDS = ("per_class_iou", "per_class_dice", "per_class_accuracy")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def atomic_csv(table: pd.DataFrame, path: Path) -> None:
    """Write a complete CSV before replacing the published result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", dir=path.parent, delete=False, encoding="utf-8", newline="") as handle:
        temporary = Path(handle.name)
        table.to_csv(handle, index=False)
    os.replace(temporary, path)


def atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=path.parent, delete=False, encoding="utf-8") as handle:
        temporary = Path(handle.name)
        json.dump(json_safe(value), handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def validate_cache_payload(
    payload: dict[str, Any],
    *,
    split: str,
    condition: str,
    checkpoint_sha256: str,
    degradation_config_sha256: str,
) -> None:
    """Fail closed when a frozen-logit cache does not match this protocol."""
    if payload.get("split") != split or payload.get("condition") != condition:
        raise ValueError(f"Cache metadata does not match {split}/{condition}.")
    if payload.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"Cache checkpoint hash mismatch for {split}/{condition}.")
    if payload.get("degradation_config_sha256") != degradation_config_sha256:
        raise ValueError(f"Cache degradation-config hash mismatch for {split}/{condition}.")
    sample_ids = payload.get("sample_id")
    logits = payload.get("logits")
    labels = payload.get("labels")
    if not isinstance(sample_ids, list) or logits is None or labels is None:
        raise ValueError(f"Cache is missing sample IDs, logits, or labels for {split}/{condition}.")
    if len(sample_ids) != len(logits) or len(sample_ids) != len(labels):
        raise ValueError(f"Cache sample/logit/label counts differ for {split}/{condition}.")
    if len({str(sample_id) for sample_id in sample_ids}) != len(sample_ids):
        raise ValueError(f"Cache contains duplicate sample IDs for {split}/{condition}.")


def temperature_for(method: str, payload: dict[str, Any], temperatures: dict[str, Any]) -> float:
    if method == "raw":
        return 1.0
    if method == "clean_global":
        return float(temperatures["clean_global"])
    if method == "pooled":
        return float(temperatures["pooled"])
    if method == "per_degradation":
        degradation_type = str(payload["degradation_type"])
        return float(temperatures["per_degradation"][degradation_type])
    raise ValueError(f"Unknown registered method: {method}")


@dataclass
class _GpuMetrics:
    """Accumulate calibration terms on CUDA and defer only score ordering."""

    classes: int
    bins: int = 15
    count: int = 0
    nll_sum: float = 0.0
    brier_sum: float = 0.0
    bin_count: torch.Tensor = field(init=False)
    bin_confidence: torch.Tensor = field(init=False)
    bin_correct: torch.Tensor = field(init=False)
    class_bin_count: torch.Tensor = field(init=False)
    class_bin_probability: torch.Tensor = field(init=False)
    class_bin_event: torch.Tensor = field(init=False)
    confidences: list[torch.Tensor] = field(default_factory=list)
    errors: list[torch.Tensor] = field(default_factory=list)

    def __post_init__(self) -> None:
        device = torch.device("cuda")
        self.bin_count = torch.zeros(self.bins, dtype=torch.long, device=device)
        self.bin_confidence = torch.zeros(self.bins, dtype=torch.float64, device=device)
        self.bin_correct = torch.zeros(self.bins, dtype=torch.float64, device=device)
        self.class_bin_count = torch.zeros((self.classes, self.bins), dtype=torch.long, device=device)
        self.class_bin_probability = torch.zeros((self.classes, self.bins), dtype=torch.float64, device=device)
        self.class_bin_event = torch.zeros((self.classes, self.bins), dtype=torch.float64, device=device)

    def update(self, probabilities: torch.Tensor, target: torch.Tensor, *, ignore_index: int) -> None:
        valid = target.ne(ignore_index) & target.ge(0) & target.lt(self.classes)
        if not bool(valid.any()):
            return
        probabilities = probabilities.permute(0, 2, 3, 1)[valid].float()
        target = target[valid].long()
        prediction = probabilities.argmax(dim=1)
        confidence = probabilities.max(dim=1).values
        correct = prediction.eq(target)
        true_probability = probabilities.gather(1, target[:, None]).squeeze(1).clamp_min(1e-12)
        self.count += int(target.numel())
        self.nll_sum += float((-true_probability.log()).sum(dtype=torch.float64).item())
        self.brier_sum += float((probabilities.square().sum(dim=1) - 2.0 * true_probability + 1.0).sum(dtype=torch.float64).item())
        bin_index = (confidence * self.bins).long().clamp_max(self.bins - 1)
        self.bin_count += torch.bincount(bin_index, minlength=self.bins)
        self.bin_confidence += torch.bincount(bin_index, weights=confidence.to(torch.float64), minlength=self.bins)
        self.bin_correct += torch.bincount(bin_index, weights=correct.to(torch.float64), minlength=self.bins)
        for class_id in range(self.classes):
            probability = probabilities[:, class_id]
            class_bin = (probability * self.bins).long().clamp_max(self.bins - 1)
            self.class_bin_count[class_id] += torch.bincount(class_bin, minlength=self.bins)
            self.class_bin_probability[class_id] += torch.bincount(class_bin, weights=probability.to(torch.float64), minlength=self.bins)
            self.class_bin_event[class_id] += torch.bincount(class_bin, weights=target.eq(class_id).to(torch.float64), minlength=self.bins)
        self.confidences.append(confidence.detach())
        self.errors.append((~correct).detach())

    @staticmethod
    def _ranking_metrics(confidence: torch.Tensor, errors: torch.Tensor) -> dict[str, float]:
        """Compute global error AUROC and AURC on CUDA without CPU pixel copies.

        Float scores are almost entirely unique after softmax; equal-score groups
        are recorded so a future protocol can diagnose any material tie rate.
        """
        uncertainty = 1.0 - confidence
        values, order = torch.sort(uncertainty)
        ordered_errors = errors[order].to(torch.float64)
        count = int(ordered_errors.numel())
        positives = int(errors.sum().item())
        negatives = count - positives
        if not positives or not negatives:
            auroc = float("nan")
        else:
            # Scores are float32 softmax outputs; the deterministic secondary
            # index order is immaterial unless values are exactly equal.
            ranks = torch.arange(1, count + 1, device=confidence.device, dtype=torch.float64)
            rank_sum = (ranks * ordered_errors).sum(dtype=torch.float64)
            auroc = float(((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)).item())
        cumulative = 0.0
        risk_sum = 0.0
        chunk = 2_000_000
        for start in range(0, count, chunk):
            stop = min(start + chunk, count)
            values_chunk = torch.cumsum(ordered_errors[start:stop], dim=0, dtype=torch.float64) + cumulative
            ranks_chunk = torch.arange(start + 1, stop + 1, device=confidence.device, dtype=torch.float64)
            risk_sum += float((values_chunk / ranks_chunk).sum(dtype=torch.float64).item())
            cumulative = float(values_chunk[-1].item())
        first_risk = float(ordered_errors[0].item())
        last_risk = cumulative / count
        aurc = (risk_sum - 0.5 * (first_risk + last_risk)) / count
        tie_fraction = float(values[1:].eq(values[:-1]).sum().item() / max(1, count - 1))
        return {"error_auroc": auroc, "aurc": aurc, "score_tie_fraction": tie_fraction}

    def metrics(self) -> dict[str, Any]:
        if not self.count:
            raise ValueError("No valid pixels were available for cached evaluation.")
        counts = self.bin_count.cpu().numpy().astype(np.float64)
        populated = counts > 0
        confidence = np.zeros(self.bins, dtype=np.float64)
        accuracy = np.zeros(self.bins, dtype=np.float64)
        confidence[populated] = self.bin_confidence.cpu().numpy()[populated] / counts[populated]
        accuracy[populated] = self.bin_correct.cpu().numpy()[populated] / counts[populated]
        classwise: list[float] = []
        class_counts = self.class_bin_count.cpu().numpy().astype(np.float64)
        class_probability = self.class_bin_probability.cpu().numpy()
        class_event = self.class_bin_event.cpu().numpy()
        for class_id in range(self.classes):
            present = class_counts[class_id] > 0
            probability = np.zeros(self.bins, dtype=np.float64)
            event = np.zeros(self.bins, dtype=np.float64)
            probability[present] = class_probability[class_id, present] / class_counts[class_id, present]
            event[present] = class_event[class_id, present] / class_counts[class_id, present]
            classwise.append(float(np.sum((class_counts[class_id] / self.count) * np.abs(event - probability))))
        confidence_values = torch.cat(self.confidences)
        error_values = torch.cat(self.errors)
        ranking = self._ranking_metrics(confidence_values, error_values)
        correct_values = ~error_values
        result = {
            "nll": self.nll_sum / self.count,
            "brier_score": self.brier_sum / self.count,
            "ece": float(np.sum((counts / self.count) * np.abs(accuracy - confidence))),
            "classwise_ece": classwise,
            "mean_confidence": float(confidence_values.mean().item()),
            "mean_correct_confidence": float(confidence_values[correct_values].mean().item()) if bool(correct_values.any()) else float("nan"),
            "mean_wrong_confidence": float(confidence_values[error_values].mean().item()) if bool(error_values.any()) else float("nan"),
            **ranking,
        }
        del confidence_values, error_values, self.confidences[:], self.errors[:]
        torch.cuda.empty_cache()
        return result


def evaluate_cache_methods(payload: dict[str, Any], temperatures: dict[str, float], *, classes: int = 8, ignore_index: int = 255) -> dict[str, dict[str, Any]]:
    """Evaluate all registered temperatures together, sharing each upsampled batch."""
    if set(temperatures) != set(METHODS):
        raise ValueError("Every registered temperature method must be evaluated together.")
    if any((not np.isfinite(value) or value <= 0) for value in temperatures.values()):
        raise ValueError("Temperatures must be finite and positive.")
    logits = payload["logits"]
    labels = payload["labels"]
    if logits.dtype != torch.float16 or labels.ndim != 3 or logits.ndim != 4:
        raise ValueError("Cache format is invalid; expected float16 logits and [N,H,W] labels.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the frozen AMP inference path.")
    device = torch.device("cuda")
    matrix = torch.zeros((classes, classes), dtype=torch.long, device=device)
    accumulators = {method: _GpuMetrics(classes) for method in METHODS}
    for start in range(0, len(labels), 4):
        target = labels[start:start + 4].long().to(device, non_blocking=True)
        low_resolution = logits[start:start + 4].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=True):
            upsampled = functional.interpolate(low_resolution, size=target.shape[-2:], mode="bilinear", align_corners=False)
            raw_prediction = upsampled.argmax(dim=1)
        matrix += confusion_matrix(raw_prediction, target, num_classes=classes, ignore_index=ignore_index).to(device)
        for method in METHODS:
            temperature = temperatures[method]
            with torch.amp.autocast("cuda", enabled=True):
                scaled_logits = upsampled if temperature == 1.0 else upsampled / temperature
                scaled_prediction = scaled_logits.argmax(dim=1)
                probabilities = torch.softmax(scaled_logits, dim=1)
            if not torch.equal(raw_prediction, scaled_prediction):
                raise AssertionError("Positive scalar temperature changed an argmax prediction.")
            accumulators[method].update(probabilities, target, ignore_index=ignore_index)
    segmentation = metrics_from_confusion_matrix(matrix.cpu())
    results: dict[str, dict[str, Any]] = {}
    for method, accumulator in accumulators.items():
        result = dict(segmentation)
        result.update(accumulator.metrics())
        results[method] = result
    return results


def compare_raw(metrics: pd.DataFrame, pilot_path: Path) -> dict[str, float]:
    pilot = pd.read_csv(pilot_path)
    merged = metrics.loc[metrics["method"] == "raw"].merge(
        pilot, on=["condition", "degradation_type", "severity", "split"], suffixes=("_cache", "_pilot"), validate="one_to_one",
    )
    if len(merged) != 26:
        raise AssertionError(f"Expected 26 raw cache rows, found {len(merged)}.")
    # The frozen AMP forward pass is replayed on CUDA.  CUDA reduction kernels
    # can change a handful of boundary argmax values across otherwise identical
    # runs, so rerun-level segmentation comparisons use a documented 1e-4
    # tolerance.  Within a single cache, temperature invariants remain exact.
    tolerances = {"miou": 1e-4, "pixel_accuracy": 1e-4, "mean_dice": 1e-4, "nll": 5e-4, "brier_score": 5e-4, "ece": 5e-4}
    differences: dict[str, float] = {}
    for name, tolerance in tolerances.items():
        maximum = float(np.max(np.abs(merged[f"{name}_cache"] - merged[f"{name}_pilot"])))
        differences[name] = maximum
        if maximum > tolerance:
            raise AssertionError(f"Raw cache does not reproduce frozen pilot {name}: max difference {maximum} > {tolerance}.")
    return differences


def partial_paths(output: Path, split: str, condition: str) -> tuple[Path, Path]:
    root = output / "partial_results"
    return root / f"{split}_{condition}_metrics.csv", root / f"{split}_{condition}_per_class_metrics.csv"


def load_complete_partial(output: Path, split: str, condition: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    metrics_path, per_class_path = partial_paths(output, split, condition)
    if not metrics_path.is_file() or not per_class_path.is_file():
        return None
    metrics = pd.read_csv(metrics_path)
    per_class = pd.read_csv(per_class_path)
    if (
        len(metrics) != len(METHODS)
        or set(metrics["method"]) != set(METHODS)
        or metrics.duplicated(["method", "condition", "split"]).any()
        or len(per_class) != len(METHODS) * len(CLASS_NAMES)
        or not np.isfinite(metrics[["miou", "nll", "brier_score", "ece", "aurc"]].to_numpy(dtype=float)).all()
    ):
        return None
    return metrics, per_class


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pre-registered temperature scaling methods from frozen caches.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "temperature_scaling.yaml")
    parser.add_argument("--restart", action="store_true", help="Ignore any complete per-condition partial results and recompute them.")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    experiment = config["experiment"]
    if list(experiment["splits"]) not in (["calibration", "val"], ["calibration", "confirmation"]):
        raise ValueError("Only calibration plus val/confirmation are permitted; official TEST is locked.")
    output = ROOT / experiment["output_dir"]
    cache_root = ROOT / experiment.get("cache_dir", str((output / "cache").relative_to(ROOT)))
    temperature_path = output / "temperatures.json"
    if not temperature_path.is_file():
        raise FileNotFoundError("Fit temperatures before cached evaluation.")
    temperatures = json.loads(temperature_path.read_text(encoding="utf-8"))
    expected_checkpoint_sha256 = sha256(ROOT / experiment["checkpoint"])
    expected_degradation_config_sha256 = sha256(ROOT / experiment["degradation_config"])
    rows: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    for split in tuple(experiment["splits"]):
        for condition in CONDITIONS:
            if not args.restart:
                partial = load_complete_partial(output, split, condition)
                if partial is not None:
                    cached_rows, cached_per_class = partial
                    rows.extend(cached_rows.to_dict("records"))
                    per_class.extend(cached_per_class.to_dict("records"))
                    print(f"reused complete partial {split:11s} {condition:14s}", flush=True)
                    continue
            cache_path = cache_root / split / f"{condition}.pt"
            if not cache_path.is_file():
                raise FileNotFoundError(cache_path)
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            validate_cache_payload(
                payload,
                split=split,
                condition=condition,
                checkpoint_sha256=expected_checkpoint_sha256,
                degradation_config_sha256=expected_degradation_config_sha256,
            )
            method_temperatures = {method: temperature_for(method, payload, temperatures) for method in METHODS}
            cache_metrics = evaluate_cache_methods(payload, method_temperatures)
            condition_rows: list[dict[str, Any]] = []
            condition_per_class: list[dict[str, Any]] = []
            raw_segmentation = {name: cache_metrics["raw"][name] for name in (*SEGMENTATION_FIELDS, *PER_CLASS_FIELDS)}
            for method in METHODS:
                temperature = method_temperatures[method]
                metrics = cache_metrics[method]
                segmentation = {name: metrics[name] for name in (*SEGMENTATION_FIELDS, *PER_CLASS_FIELDS)}
                if segmentation != raw_segmentation:
                    raise AssertionError(f"Segmentation metrics changed for {method}/{split}/{condition}.")
                condition_rows.append({
                    "method": method, "fit_scope": "none" if method == "raw" else method,
                    "condition": condition, "degradation_type": payload["degradation_type"], "severity": payload["severity"], "split": split,
                    "temperature": temperature,
                    **{name: metrics[name] for name in ("miou", "pixel_accuracy", "mean_accuracy", "mean_dice", "nll", "brier_score", "ece", "error_auroc", "aurc", "mean_confidence", "mean_correct_confidence", "mean_wrong_confidence")},
                })
                for class_id, class_name in enumerate(CLASS_NAMES):
                    condition_per_class.append({
                        "method": method, "condition": condition, "split": split, "class_id": class_id, "class_name": class_name,
                        "iou": metrics["per_class_iou"][class_id], "dice": metrics["per_class_dice"][class_id],
                        "accuracy": metrics["per_class_accuracy"][class_id], "classwise_ece": metrics["classwise_ece"][class_id],
                    })
            metrics_path, per_class_path = partial_paths(output, split, condition)
            atomic_csv(pd.DataFrame(condition_rows), metrics_path)
            atomic_csv(pd.DataFrame(condition_per_class), per_class_path)
            rows.extend(condition_rows)
            per_class.extend(condition_per_class)
            print(f"completed {split:11s} {condition:14s}; wrote resumable partial results", flush=True)
    table = pd.DataFrame(rows)
    if len(table) != 104 or table.duplicated(["method", "condition", "split"]).any():
        raise AssertionError("Expected exactly 104 uniquely keyed result rows.")
    pilot_value = experiment.get("pilot_metrics")
    raw_differences = compare_raw(table, ROOT / pilot_value) if pilot_value else {}
    atomic_csv(table, output / "metrics.csv")
    atomic_csv(pd.DataFrame(per_class), output / "per_class_metrics.csv")
    training_config_path = ROOT / experiment.get("training_config", "") if experiment.get("training_config") else None
    split_hashes: dict[str, str] = {}
    if training_config_path and training_config_path.is_file():
        training_config = load_yaml(training_config_path)
        split_dir = ROOT / training_config["data"]["split_dir"]
        split_hashes = {split: sha256(split_dir / f"{split}.csv") for split in experiment["splits"]}
    metadata = {
        "config": str(config_path.relative_to(ROOT)), "config_sha256": sha256(config_path),
        "checkpoint": experiment["checkpoint"], "checkpoint_sha256": expected_checkpoint_sha256,
        "degradation_config": experiment["degradation_config"], "degradation_config_sha256": expected_degradation_config_sha256,
        "cache_entries": 26, "cache_manifest_sha256": sha256(cache_root / "manifest.json"), "split_csv_sha256": split_hashes,
        "result_rows": 104, "methods": list(METHODS), "splits_evaluated": list(experiment["splits"]),
        "official_test_evaluated": False, "model_retrained": False, "raw_cache_vs_frozen_pilot_max_abs_difference": raw_differences,
        "raw_cache_reference_available": bool(pilot_value),
        "cache_logits_dtype": "float16", "cache_integrity_verified": True, "segmentation_invariants_verified": True,
        "execution": {
            "shared_upsample_across_methods": True,
            "gpu_streaming_calibration": True,
            "atomic_per_condition_partials": True,
            "partial_result_count": len(CONDITIONS) * len(experiment["splits"]),
        },
    }
    atomic_json(metadata, output / "evaluation_metadata.json")


if __name__ == "__main__":
    main()
