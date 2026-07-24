"""Evaluate fixed scalar temperatures from cached validation/calibration logits.

This script never loads a model, image, split CSV, or official TEST data.  It
only consumes the 26 frozen cache entries created from the formal checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
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
from scripts.evaluate_baseline import CalibrationAccumulator, json_safe  # noqa: E402
from scripts.run_degradation_pilot import ErrorDetectionAccumulator  # noqa: E402


CONDITIONS = (
    "clean", "color_s1", "color_s2", "color_s3", "turbidity_s1", "turbidity_s2", "turbidity_s3",
    "lowlight_s1", "lowlight_s2", "lowlight_s3", "blur_s1", "blur_s2", "blur_s3",
)
SPLITS = ("calibration", "val")
METHODS = ("raw", "clean_global", "pooled", "per_degradation")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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


def evaluate_cache(payload: dict[str, Any], temperature: float, *, classes: int = 8, ignore_index: int = 255) -> dict[str, Any]:
    """Evaluate one cache in small batches, preserving raw segmentation predictions."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    logits = payload["logits"]
    labels = payload["labels"]
    if logits.dtype != torch.float16 or labels.ndim != 3 or logits.ndim != 4:
        raise ValueError("Cache format is invalid; expected float16 logits and [N,H,W] labels.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the frozen AMP inference path.")
    device = torch.device("cuda")
    matrix = torch.zeros((classes, classes), dtype=torch.long)
    calibration = CalibrationAccumulator(classes)
    errors = ErrorDetectionAccumulator()
    for start in range(0, len(labels), 4):
        target = labels[start:start + 4].long().to(device, non_blocking=True)
        low_resolution = logits[start:start + 4].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=True):
            upsampled = functional.interpolate(low_resolution, size=target.shape[-2:], mode="bilinear", align_corners=False)
            raw_prediction = upsampled.argmax(dim=1)
            scaled_logits = upsampled if temperature == 1.0 else upsampled / temperature
            scaled_prediction = scaled_logits.argmax(dim=1)
            probabilities = torch.softmax(scaled_logits, dim=1)
        if not torch.equal(raw_prediction, scaled_prediction):
            raise AssertionError("Positive scalar temperature changed an argmax prediction.")
        matrix += confusion_matrix(raw_prediction, target, num_classes=classes, ignore_index=ignore_index).cpu()
        calibration.update(probabilities, target, ignore_index)
        errors.update(probabilities, target, ignore_index)
    result = metrics_from_confusion_matrix(matrix)
    result.update(calibration.metrics())
    result.update(errors.metrics())
    return result


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pre-registered temperature scaling methods from frozen caches.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "temperature_scaling.yaml")
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
    raw_segmentation: dict[tuple[str, str], dict[str, Any]] = {}
    for split in tuple(experiment["splits"]):
        for condition in CONDITIONS:
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
            for method in METHODS:
                temperature = temperature_for(method, payload, temperatures)
                metrics = evaluate_cache(payload, temperature)
                key = (split, condition)
                segmentation = {name: metrics[name] for name in ("miou", "pixel_accuracy", "mean_accuracy", "mean_dice", "per_class_iou", "per_class_dice", "per_class_accuracy")}
                if method == "raw":
                    raw_segmentation[key] = segmentation
                elif segmentation != raw_segmentation[key]:
                    raise AssertionError(f"Segmentation metrics changed for {method}/{split}/{condition}.")
                rows.append({
                    "method": method, "fit_scope": "none" if method == "raw" else method,
                    "condition": condition, "degradation_type": payload["degradation_type"], "severity": payload["severity"], "split": split,
                    "temperature": temperature,
                    **{name: metrics[name] for name in ("miou", "pixel_accuracy", "mean_accuracy", "mean_dice", "nll", "brier_score", "ece", "error_auroc", "aurc", "mean_confidence", "mean_correct_confidence", "mean_wrong_confidence")},
                })
                for class_id, class_name in enumerate(CLASS_NAMES):
                    per_class.append({
                        "method": method, "condition": condition, "split": split, "class_id": class_id, "class_name": class_name,
                        "iou": metrics["per_class_iou"][class_id], "dice": metrics["per_class_dice"][class_id],
                        "accuracy": metrics["per_class_accuracy"][class_id], "classwise_ece": metrics["classwise_ece"][class_id],
                    })
                print(f"{method:16s} {split:11s} {condition:14s} T={temperature:.4f} NLL={metrics['nll']:.6f} ECE={metrics['ece']:.6f}")
    table = pd.DataFrame(rows)
    if len(table) != 104 or table.duplicated(["method", "condition", "split"]).any():
        raise AssertionError("Expected exactly 104 uniquely keyed result rows.")
    pilot_value = experiment.get("pilot_metrics")
    raw_differences = compare_raw(table, ROOT / pilot_value) if pilot_value else {}
    table.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(per_class).to_csv(output / "per_class_metrics.csv", index=False)
    metadata = {
        "config": str(config_path.relative_to(ROOT)), "config_sha256": sha256(config_path),
        "checkpoint": experiment["checkpoint"], "checkpoint_sha256": expected_checkpoint_sha256,
        "degradation_config": experiment["degradation_config"], "degradation_config_sha256": expected_degradation_config_sha256,
        "cache_entries": 26, "result_rows": 104, "methods": list(METHODS), "splits_evaluated": list(SPLITS),
        "official_test_evaluated": False, "model_retrained": False, "raw_cache_vs_frozen_pilot_max_abs_difference": raw_differences,
        "cache_logits_dtype": "float16", "cache_integrity_verified": True, "segmentation_invariants_verified": True,
    }
    (output / "evaluation_metadata.json").write_text(json.dumps(json_safe(metadata), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
