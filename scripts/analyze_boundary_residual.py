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
        return {"pixels": self.count, "error_rate": error_rate, "nll": self.nll_sum / self.count, "brier_score": self.brier_sum / self.count, "ece": ece, "error_auroc": auroc, "aurc": aurc, "oracle_aurc": oracle_aurc, "eaurc": aurc - oracle_aurc, "mean_wrong_confidence": float(all_confidence[errors].mean()) if errors.any() else float("nan")}


def boundary_mask(labels: torch.Tensor, radius: int) -> torch.Tensor:
    values = labels.float().unsqueeze(1)
    maximum = functional.max_pool2d(values, kernel_size=2 * radius + 1, stride=1, padding=radius)
    minimum = -functional.max_pool2d(-values, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return maximum[:, 0].ne(minimum[:, 0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze residual calibration by boundary/interior region from frozen validation caches.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "temperature_scaling.yaml")
    parser.add_argument("--radius", type=int, default=3)
    args = parser.parse_args()
    config = load_yaml(args.config.resolve())
    experiment = config["experiment"]
    if list(experiment["splits"]) != ["calibration", "val"]:
        raise ValueError("Official TEST is locked.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temperatures = json.loads((ROOT / experiment["output_dir"] / "temperatures.json").read_text(encoding="utf-8"))
    expected_checkpoint = sha256(ROOT / experiment["checkpoint"])
    expected_degradation = sha256(ROOT / experiment["degradation_config"])
    cache_root = ROOT / experiment["output_dir"] / "cache" / "val"
    rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        payload = torch.load(cache_root / f"{condition}.pt", map_location="cpu", weights_only=False)
        validate_cache_payload(payload, split="val", condition=condition, checkpoint_sha256=expected_checkpoint, degradation_config_sha256=expected_degradation)
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
        for (method, region), accumulator in accumulators.items():
            rows.append({"condition": condition, "method": method, "region": region, "boundary_radius": args.radius, **accumulator.result()})
        print(f"processed {condition}")
    output = ROOT / "outputs" / "residual_calibration_analysis"
    pd.DataFrame(rows).to_csv(output / "boundary_interior_metrics.csv", index=False)


if __name__ == "__main__":
    main()
