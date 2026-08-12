"""Render the seed-aggregated SADR condition-gain figure."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["condition"]: float(row["miou"]) for row in csv.DictReader(handle) if row["variant"] == "SADR"}


def main() -> None:
    files = [
        ROOT / "outputs/sadr_long/confirmation_evaluation/condition_metrics.csv",
        ROOT / "outputs/sadr_seed2/confirmation_evaluation/condition_metrics.csv",
        ROOT / "outputs/sadr_seed3/confirmation_evaluation/condition_metrics.csv",
    ]
    baselines = {}
    with files[0].open(newline="", encoding="utf-8") as handle:
        baselines = {row["condition"]: float(row["miou"]) for row in csv.DictReader(handle) if row["variant"] == "UIIS-F4"}
    gains = [read(path) for path in files]
    conditions = ["color_s1", "color_s2", "color_s3", "turbidity_s1", "turbidity_s2", "turbidity_s3", "lowlight_s1", "lowlight_s2", "lowlight_s3", "blur_s1", "blur_s2", "blur_s3"]
    values = np.asarray([[100.0 * (run[name] - baselines[name]) for name in conditions] for run in gains])
    mean, sd = values.mean(0), values.std(0, ddof=1)
    labels = [name.replace("_", "\n") for name in conditions]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.errorbar(np.arange(len(conditions)), mean, yerr=sd, fmt="o", capsize=3, color="#087f8c")
    ax.set_xticks(np.arange(len(conditions)), labels)
    ax.set_ylabel("SADR − UIIS-F4 mIoU (percentage points)")
    ax.set_title("Three-seed robustness across fixed synthetic degradations")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output = ROOT / "outputs/sadr_long/condition_gain_plot.png"
    output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(output, dpi=180, bbox_inches="tight"); print(output)


if __name__ == "__main__": main()
