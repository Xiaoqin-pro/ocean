"""Diagnose residual calibration after frozen scalar temperature scaling.

Only frozen validation logits are read.  This analysis never loads images or a
SegFormer model and refuses any split other than ``val``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_temperature_scaling import CONDITIONS, load_yaml, sha256, temperature_for, validate_cache_payload  # noqa: E402


METHODS = ("clean_global", "pooled", "per_degradation")
METRICS = ("nll", "brier_score", "ece")


def image_metrics(logits: torch.Tensor, labels: torch.Tensor, temperature: float, bins: int) -> dict[str, float]:
    """Compute NLL, Brier and ECE independently for one image."""
    probabilities = torch.softmax(logits / temperature, dim=1)[0].permute(1, 2, 0).reshape(-1, logits.shape[1]).float().cpu().numpy()
    targets = labels[0].reshape(-1).cpu().numpy()
    valid = (targets >= 0) & (targets < logits.shape[1])
    probabilities, targets = probabilities[valid], targets[valid]
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    nll = float(-np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0)).mean())
    one_hot = np.eye(logits.shape[1], dtype=np.float32)[targets]
    brier = float(np.square(probabilities - one_hot).sum(axis=1).mean())
    index = np.minimum((confidence * bins).astype(int), bins - 1)
    ece = 0.0
    for bin_id in range(bins):
        chosen = index == bin_id
        if chosen.any():
            ece += float(chosen.mean() * abs((prediction[chosen] == targets[chosen]).mean() - confidence[chosen].mean()))
    return {"nll": nll, "brier_score": brier, "ece": ece}


def bootstrap_difference(values: np.ndarray, *, iterations: int, seed: int) -> tuple[float, float, float]:
    """Image-level paired bootstrap for candidate minus clean-global metric."""
    rng = np.random.default_rng(seed)
    count = len(values)
    draws = rng.integers(0, count, size=(iterations, count))
    means = values[draws].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze residual scalar-calibration behavior from frozen validation caches.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "temperature_scaling.yaml")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    experiment = config["experiment"]
    if list(experiment["splits"]) != ["calibration", "val"]:
        raise ValueError("Temperature protocol must be restricted to calibration and val; official TEST is locked.")
    output = ROOT / "outputs" / "residual_calibration_analysis"
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "experiments" / "temperature_scaling_metrics.csv"
    per_class_source = ROOT / "experiments" / "temperature_scaling_per_class.csv"
    temperatures = json.loads((ROOT / experiment["output_dir"] / "temperatures.json").read_text(encoding="utf-8"))
    expected_checkpoint = sha256(ROOT / experiment["checkpoint"])
    expected_degradation = sha256(ROOT / experiment["degradation_config"])
    bins = int(config["metrics"]["ece_bins"])

    summary = pd.read_csv(source)
    val = summary.loc[summary["split"] == "val"].copy()
    clean = val.loc[val["method"] == "clean_global"].set_index("condition")
    delta_rows: list[dict[str, object]] = []
    for method in ("raw", "pooled", "per_degradation"):
        candidate = val.loc[val["method"] == method].set_index("condition")
        for condition in CONDITIONS:
            for metric in ("nll", "ece", "brier_score", "aurc", "error_auroc", "mean_wrong_confidence"):
                delta_rows.append({"condition": condition, "method": method, "reference": "clean_global", "metric": metric, "delta_candidate_minus_clean_global": float(candidate.loc[condition, metric] - clean.loc[condition, metric])})
    pd.DataFrame(delta_rows).to_csv(output / "condition_deltas.csv", index=False)

    image_rows: list[dict[str, object]] = []
    cache_root = ROOT / experiment["output_dir"] / "cache" / "val"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for condition in CONDITIONS:
        payload = torch.load(cache_root / f"{condition}.pt", map_location="cpu", weights_only=False)
        validate_cache_payload(payload, split="val", condition=condition, checkpoint_sha256=expected_checkpoint, degradation_config_sha256=expected_degradation)
        for start in range(0, len(payload["labels"]), 4):
            labels = payload["labels"][start:start + 4].to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = functional.interpolate(payload["logits"][start:start + 4].to(device), size=labels.shape[-2:], mode="bilinear", align_corners=False)
            for offset, sample_id in enumerate(payload["sample_id"][start:start + len(labels)]):
                single_logits, single_labels = logits[offset:offset + 1], labels[offset:offset + 1]
                for method in METHODS:
                    temperature = temperature_for(method, payload, temperatures)
                    image_rows.append({"condition": condition, "sample_id": sample_id, "method": method, **image_metrics(single_logits, single_labels, temperature, bins)})
        print(f"processed {condition}")
    image_table = pd.DataFrame(image_rows)
    image_table.to_csv(output / "per_image_metrics.csv", index=False)

    bootstrap_rows: list[dict[str, object]] = []
    for condition in (*CONDITIONS, "all_conditions"):
        subset = image_table if condition == "all_conditions" else image_table.loc[image_table["condition"] == condition]
        reference = subset.loc[subset["method"] == "clean_global"].set_index(["condition", "sample_id"])
        for method in ("pooled", "per_degradation"):
            candidate = subset.loc[subset["method"] == method].set_index(["condition", "sample_id"])
            for metric in METRICS:
                difference = candidate[metric].loc[reference.index].to_numpy() - reference[metric].to_numpy()
                mean, low, high = bootstrap_difference(difference, iterations=args.bootstrap_iterations, seed=20260721 + len(metric) + len(method))
                bootstrap_rows.append({"condition": condition, "comparison": f"{method}_minus_clean_global", "metric": metric, "mean_difference": mean, "ci95_low": low, "ci95_high": high, "iterations": args.bootstrap_iterations, "unit": "image"})
    pd.DataFrame(bootstrap_rows).to_csv(output / "paired_bootstrap.csv", index=False)

    per_class = pd.read_csv(per_class_source)
    classes = per_class.loc[(per_class["split"] == "val") & per_class["method"].isin(["raw", "clean_global"])]
    pivot = classes.pivot(index=["condition", "class_id", "class_name"], columns="method", values="classwise_ece").reset_index()
    pivot["classwise_ece_delta_clean_minus_raw"] = pivot["clean_global"] - pivot["raw"]
    pivot.to_csv(output / "classwise_residual_calibration.csv", index=False)
    metadata = {"source_commit": "v0.4-temperature-scaling", "splits_evaluated": ["val"], "official_test_evaluated": False, "model_retrained": False, "bootstrap_iterations": args.bootstrap_iterations, "cache_integrity_verified": True}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
