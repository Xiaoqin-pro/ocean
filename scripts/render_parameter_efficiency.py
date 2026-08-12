"""Render a compact parameter-efficiency Pareto figure from locked summaries."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    confirmation = json.loads((ROOT / "outputs/parameter_efficiency/confirmation_evaluation/summary.json").read_text(encoding="utf-8"))
    suim = json.loads((ROOT / "outputs/parameter_efficiency/suim_official_evaluation/summary.json").read_text(encoding="utf-8"))
    variants = ["frozen", "sadr", "head", "last_block", "full"]
    trainable = {"frozen": 0, "sadr": 11012, "head": 396808, "last_block": 800000, "full": 3716200}
    labels = {"frozen": "Frozen F4", "sadr": "SADR", "head": "Head-only", "last_block": "Last block", "full": "Full FT"}
    colors = {"frozen": "#777777", "sadr": "#0072B2", "head": "#D55E00", "last_block": "#CC79A7", "full": "#009E73"}
    x = [100.0 * trainable[name] / 3716200 for name in variants]
    y_confirmation = [confirmation["gain_vs_frozen_pp"][name] for name in variants]
    y_suim = [suim["gain_vs_frozen_pp"][name] for name in variants]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for axis, y, title in zip(axes, (y_confirmation, y_suim), ("UIIS confirmation", "SUIM official"), strict=True):
        for name, xv, yv in zip(variants, x, y, strict=True):
            axis.scatter(xv, yv, s=70, color=colors[name], label=labels[name], zorder=3)
            axis.annotate(labels[name], (xv, yv), xytext=(5, 5), textcoords="offset points", fontsize=8)
        axis.axhline(0, color="#444444", linewidth=0.8)
        axis.set_xscale("symlog", linthresh=0.01)
        axis.set_xlabel("Trainable expert fraction (%)")
        axis.set_ylabel("mIoU change vs frozen (pp)")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    output = ROOT / "outputs/parameter_efficiency/parameter_pareto.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
