"""Boundary-radius sensitivity and image-clustered bootstrap diagnostics.

Ground-truth boundaries are evaluation strata only; they are never used as a
model input or deployment-time uncertainty signal.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_boundary_residual import RegionStats, boundary_mask  # noqa: E402
from scripts.evaluate_temperature_scaling import CONDITIONS, load_yaml, sha256, validate_cache_payload  # noqa: E402


def bootstrap(values: np.ndarray, *, iterations: int, seed: int) -> tuple[float, float, float]:
    generator = np.random.default_rng(seed)
    count = len(values)
    means = values[generator.integers(0, count, size=(iterations, count))].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def clustered_bootstrap(table: pd.DataFrame, *, iterations: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    comparisons = (
        ("boundary_minus_interior", "clean_global", "boundary", "clean_global", "interior", ("nll", "ece", "aurc", "eaurc")),
        ("clean_global_minus_raw_boundary", "clean_global", "boundary", "raw", "boundary", ("nll", "ece")),
    )
    for radius in sorted(table["boundary_radius"].unique()):
        radius_rows = table.loc[table["boundary_radius"] == radius]
        for name, left_method, left_region, right_method, right_region, metrics in comparisons:
            left = radius_rows.loc[(radius_rows["method"] == left_method) & (radius_rows["region"] == left_region)].groupby("sample_id")[list(metrics)].mean()
            right = radius_rows.loc[(radius_rows["method"] == right_method) & (radius_rows["region"] == right_region)].groupby("sample_id")[list(metrics)].mean()
            if not left.index.equals(right.index):
                raise ValueError("Each bootstrap image cluster must retain all 13 conditions.")
            for metric in metrics:
                mean, low, high = bootstrap((left[metric] - right[metric]).to_numpy(), iterations=iterations, seed=20260721 + radius * 100 + len(metric))
                rows.append({"boundary_radius": radius, "comparison": name, "metric": metric, "mean_difference": mean, "ci95_low": low, "ci95_high": high, "iterations": iterations, "cluster_unit": "sample_id_with_all_13_conditions", "images": len(left)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test boundary residual calibration across radii with image-clustered bootstrap.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "temperature_scaling.yaml")
    parser.add_argument("--radii", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()
    config = load_yaml(args.config.resolve())
    experiment = config["experiment"]
    if list(experiment["splits"]) != ["calibration", "val"]:
        raise ValueError("Official TEST is locked.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temperature = float(json.loads((ROOT / experiment["output_dir"] / "temperatures.json").read_text(encoding="utf-8"))["clean_global"])
    expected_checkpoint = sha256(ROOT / experiment["checkpoint"])
    expected_degradation = sha256(ROOT / experiment["degradation_config"])
    cache_root = ROOT / experiment["output_dir"] / "cache" / "val"
    aggregate: dict[tuple[str, int, str, str], RegionStats] = {}
    per_image: list[dict[str, object]] = []
    for condition in CONDITIONS:
        payload = torch.load(cache_root / f"{condition}.pt", map_location="cpu", weights_only=False)
        validate_cache_payload(payload, split="val", condition=condition, checkpoint_sha256=expected_checkpoint, degradation_config_sha256=expected_degradation)
        for start in range(0, len(payload["labels"]), 4):
            labels = payload["labels"][start:start + 4].to(device)
            valid = labels.ge(0) & labels.lt(8)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = functional.interpolate(payload["logits"][start:start + 4].to(device), size=labels.shape[-2:], mode="bilinear", align_corners=False)
                probabilities = {"raw": torch.softmax(logits, dim=1), "clean_global": torch.softmax(logits / temperature, dim=1)}
            for radius in args.radii:
                regions = {"boundary": valid & boundary_mask(labels, radius), "interior": valid & ~boundary_mask(labels, radius)}
                for method, values in probabilities.items():
                    for region, mask in regions.items():
                        key = (condition, radius, method, region)
                        aggregate.setdefault(key, RegionStats(int(config["metrics"]["ece_bins"]))).update(values, labels, mask)
                        for offset, sample_id in enumerate(payload["sample_id"][start:start + len(labels)]):
                            image_stats = RegionStats(int(config["metrics"]["ece_bins"]))
                            image_stats.update(values[offset:offset + 1], labels[offset:offset + 1], mask[offset:offset + 1])
                            per_image.append({"condition": condition, "sample_id": sample_id, "boundary_radius": radius, "method": method, "region": region, **image_stats.result()})
        print(f"processed {condition}")
    aggregate_rows = [{"condition": condition, "boundary_radius": radius, "method": method, "region": region, **stats.result()} for (condition, radius, method, region), stats in aggregate.items()]
    output = ROOT / "outputs" / "residual_calibration_analysis"
    pd.DataFrame(aggregate_rows).to_csv(output / "boundary_radius_metrics.csv", index=False)
    per_image_table = pd.DataFrame(per_image)
    per_image_table.to_csv(output / "boundary_radius_per_image.csv", index=False)
    clustered_bootstrap(per_image_table, iterations=args.bootstrap_iterations).to_csv(output / "boundary_radius_clustered_bootstrap.csv", index=False)
    (output / "boundary_radius_metadata.json").write_text(json.dumps({"radii": args.radii, "bootstrap_iterations": args.bootstrap_iterations, "cluster_unit": "sample_id_with_all_13_conditions", "gt_boundary_diagnostic_only": True, "official_test_evaluated": False}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
