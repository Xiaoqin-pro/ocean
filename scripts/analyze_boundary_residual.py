"""Compare raw and clean-global calibration on mask boundaries and interiors."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_temperature_scaling import CONDITIONS, load_yaml, sha256, validate_cache_payload  # noqa: E402


@dataclass
class RegionStats:
    bins: int
    count: int = 0
    nll_sum: float = 0.0
    brier_sum: float = 0.0
    bin_count: np.ndarray = field(init=False)
    bin_confidence: np.ndarray = field(init=False)
    bin_correct: np.ndarray = field(init=False)
    confidence: list[np.ndarray] = field(default_factory=list)
    errors: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.bin_count = np.zeros(self.bins, dtype=np.int64)
        self.bin_confidence = np.zeros(self.bins, dtype=np.float64)
        self.bin_correct = np.zeros(self.bins, dtype=np.float64)

    def update(self, probabilities: torch.Tensor, labels: torch.Tensor, region: torch.Tensor) -> None:
        classes = probabilities.shape[1]
        probs = probabilities.permute(0, 2, 3, 1)[region].detach().float().cpu().numpy()
        target = labels[region].detach().cpu().numpy()
        if not len(target):
            return
        prediction = probs.argmax(axis=1)
        confidence = probs.max(axis=1)
        self.count += len(target)
        self.nll_sum += float(-np.log(np.clip(probs[np.arange(len(target)), target], 1e-12, 1.0)).sum())
        self.brier_sum += float(np.square(probs - np.eye(classes, dtype=np.float32)[target]).sum())
        index = np.minimum((confidence * self.bins).astype(int), self.bins - 1)
        for bin_id in range(self.bins):
            chosen = index == bin_id
            if chosen.any():
                self.bin_count[bin_id] += int(chosen.sum())
                self.bin_confidence[bin_id] += float(confidence[chosen].sum())
                self.bin_correct[bin_id] += float((prediction[chosen] == target[chosen]).sum())
        self.confidence.append(confidence)
        self.errors.append(prediction != target)

    def result(self) -> dict[str, float]:
        populated = self.bin_count > 0
        confidence = np.zeros(self.bins)
        accuracy = np.zeros(self.bins)
        confidence[populated] = self.bin_confidence[populated] / self.bin_count[populated]
        accuracy[populated] = self.bin_correct[populated] / self.bin_count[populated]
        ece = float(np.sum((self.bin_count / self.count) * np.abs(accuracy - confidence)))
        all_confidence = np.concatenate(self.confidence)
        errors = np.concatenate(self.errors)
        error_rate = float(errors.mean())
        uncertainty = 1.0 - all_confidence
        auroc = float("nan") if errors.min() == errors.max() else float(roc_auc_score(errors, uncertainty))
        order = np.argsort(uncertainty)
        risk = np.cumsum(errors[order], dtype=np.float64) / np.arange(1, len(errors) + 1)
        coverage = np.arange(1, len(errors) + 1, dtype=np.float64) / len(errors)
        aurc = float(np.trapezoid(risk, coverage))
        oracle_errors = np.concatenate([np.zeros((len(errors) - int(errors.sum()),), dtype=bool), np.ones((int(errors.sum()),), dtype=bool)])
        oracle_risk = np.cumsum(oracle_errors, dtype=np.float64) / np.arange(1, len(oracle_errors) + 1)
        oracle_aurc = float(np.trapezoid(oracle_risk, coverage))
        return {"pixels": self.count, "error_rate": error_rate, "nll": self.nll_sum / self.count, "brier_score": self.brier_sum / self.count, "ece": ece, "error_auroc": auroc, "aurc": aurc, "oracle_aurc": oracle_aurc, "eaurc": aurc - oracle_aurc, "mean_confidence": float(all_confidence.mean()), "mean_wrong_confidence": float(all_confidence[errors].mean()) if errors.any() else float("nan")}


def boundary_mask(labels: torch.Tensor, radius: int) -> torch.Tensor:
    values = labels.float().unsqueeze(1)
    maximum = functional.max_pool2d(values, kernel_size=2 * radius + 1, stride=1, padding=radius)
    minimum = -functional.max_pool2d(-values, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return maximum[:, 0].ne(minimum[:, 0])


def pixel_summary(probabilities: torch.Tensor, labels: torch.Tensor, region: torch.Tensor) -> dict[str, float | int]:
    """Return additive per-image statistics for one evaluation region."""
    classes = probabilities.shape[1]
    probs = probabilities.permute(0, 2, 3, 1)[region].detach().float().cpu().numpy()
    target = labels[region].detach().cpu().numpy()
    if not len(target):
        return {"pixels": 0, "errors": 0, "nll_sum": 0.0, "brier_sum": 0.0, "confidence_sum": 0.0, "wrong_confidence_sum": 0.0, "wrong_pixels": 0}
    prediction = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    errors = prediction != target
    return {
        "pixels": int(len(target)),
        "errors": int(errors.sum()),
        "nll_sum": float(-np.log(np.clip(probs[np.arange(len(target)), target], 1e-12, 1.0)).sum()),
        "brier_sum": float(np.square(probs - np.eye(classes, dtype=np.float32)[target]).sum()),
        "confidence_sum": float(confidence.sum()),
        "wrong_confidence_sum": float(confidence[errors].sum()),
        "wrong_pixels": int(errors.sum()),
    }


def bootstrap_boundary_error_gap(per_image: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    """Cluster-bootstrap boundary minus interior error rate by original image."""
    rows: list[dict[str, object]] = []
    for index, ((condition, method), group) in enumerate(per_image.groupby(["condition", "method"], sort=True)):
        pivot = group.pivot(index="sample_id", columns="region", values=["pixels", "errors"]).dropna()
        boundary_pixels = pivot[("pixels", "boundary")].to_numpy(dtype=np.float64)
        interior_pixels = pivot[("pixels", "interior")].to_numpy(dtype=np.float64)
        boundary_errors = pivot[("errors", "boundary")].to_numpy(dtype=np.float64)
        interior_errors = pivot[("errors", "interior")].to_numpy(dtype=np.float64)
        if len(pivot) == 0 or (boundary_pixels <= 0).any() or (interior_pixels <= 0).any():
            raise ValueError(f"Boundary bootstrap requires both regions for every image: {condition}/{method}.")
        observed = boundary_errors.sum() / boundary_pixels.sum() - interior_errors.sum() / interior_pixels.sum()
        generator = np.random.default_rng(seed + index)
        draws = generator.integers(0, len(pivot), size=(iterations, len(pivot)))
        sampled = boundary_errors[draws].sum(axis=1) / boundary_pixels[draws].sum(axis=1)
        sampled -= interior_errors[draws].sum(axis=1) / interior_pixels[draws].sum(axis=1)
        rows.append({
            "condition": condition,
            "method": method,
            "cluster_unit": "original_sample_id",
            "samples": int(len(pivot)),
            "iterations": iterations,
            "boundary_minus_interior_error_rate": float(observed),
            "ci95_low": float(np.quantile(sampled, 0.025)),
            "ci95_high": float(np.quantile(sampled, 0.975)),
        })
    return pd.DataFrame(rows)


def atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def resolve_boundary_cache_context(experiment: dict[str, object]) -> tuple[str, Path]:
    """Resolve the registered non-test split and its frozen cache location."""
    splits = list(experiment["splits"])
    if splits not in (["calibration", "val"], ["calibration", "confirmation"]):
        raise ValueError("Official TEST is locked.")
    evaluation_split = str(experiment.get("evaluation_split", "val"))
    if evaluation_split not in set(splits):
        raise ValueError("Boundary evaluation split is not registered in the frozen cache protocol.")
    cache_dir = experiment.get("cache_dir")
    if cache_dir is None:
        cache_dir = f"{experiment['output_dir']}/cache"
    cache_root = ROOT / str(cache_dir) / evaluation_split
    return evaluation_split, cache_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze residual calibration by boundary/interior region from frozen validation caches.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "temperature_scaling.yaml")
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    args = parser.parse_args()
    config = load_yaml(args.config.resolve())
    experiment = config["experiment"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temperatures = json.loads((ROOT / experiment["output_dir"] / "temperatures.json").read_text(encoding="utf-8"))
    expected_checkpoint = sha256(ROOT / experiment["checkpoint"])
    expected_degradation = sha256(ROOT / experiment["degradation_config"])
    evaluation_split, cache_root = resolve_boundary_cache_context(experiment)
    rows: list[dict[str, object]] = []
    per_image_rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        payload = torch.load(cache_root / f"{condition}.pt", map_location="cpu", weights_only=False)
        validate_cache_payload(payload, split=evaluation_split, condition=condition, checkpoint_sha256=expected_checkpoint, degradation_config_sha256=expected_degradation)
        accumulators = {(method, region): RegionStats(int(config["metrics"]["ece_bins"])) for method in ("raw", "clean_global") for region in ("boundary", "interior")}
        for start in range(0, len(payload["labels"]), 4):
            labels = payload["labels"][start:start + 4].to(device)
            boundary = boundary_mask(labels, args.radius)
            valid = labels.ge(0) & labels.lt(8)
            regions = {"boundary": valid & boundary, "interior": valid & ~boundary}
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = functional.interpolate(payload["logits"][start:start + 4].to(device), size=labels.shape[-2:], mode="bilinear", align_corners=False)
                probabilities = {"raw": torch.softmax(logits, dim=1), "clean_global": torch.softmax(logits / float(temperatures["clean_global"]), dim=1)}
            for method, values in probabilities.items():
                for region, mask in regions.items():
                    accumulators[(method, region)].update(values, labels, mask)
            for image_index, sample_id in enumerate(payload["sample_id"][start:start + len(labels)]):
                for method, values in probabilities.items():
                    for region, mask in regions.items():
                        summary = pixel_summary(values[image_index:image_index + 1], labels[image_index:image_index + 1], mask[image_index:image_index + 1])
                        per_image_rows.append({"split": evaluation_split, "condition": condition, "method": method, "sample_id": sample_id, "region": region, "boundary_radius": args.radius, **summary})
        valid_pixels = sum(accumulator.count for (method, _), accumulator in accumulators.items() if method == "raw")
        for (method, region), accumulator in accumulators.items():
            rows.append({"condition": condition, "method": method, "region": region, "boundary_radius": args.radius, "region_pixel_fraction": accumulator.count / valid_pixels, **accumulator.result()})
        print(f"processed {condition}")
    output = ROOT / experiment.get("boundary_output_dir", "outputs/residual_calibration_analysis")
    output.mkdir(parents=True, exist_ok=True)
    aggregate = pd.DataFrame(rows)
    per_image = pd.DataFrame(per_image_rows)
    bootstrap = bootstrap_boundary_error_gap(per_image, args.bootstrap_iterations, args.bootstrap_seed)
    atomic_csv(aggregate, output / "boundary_interior_metrics.csv")
    atomic_csv(per_image, output / "boundary_per_image_metrics.csv")
    atomic_csv(bootstrap, output / "boundary_error_gap_bootstrap.csv")
    metadata = {
        "split_evaluated": evaluation_split,
        "conditions": CONDITIONS,
        "boundary_radius": args.radius,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.bootstrap_seed,
        "cluster_unit": "original_sample_id",
        "checkpoint_sha256": expected_checkpoint,
        "degradation_config_sha256": expected_degradation,
        "official_suim_test_evaluated": False,
        "model_retrained": False,
    }
    temporary = output / "boundary_metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary.replace(output / "boundary_metadata.json")


if __name__ == "__main__":
    main()
