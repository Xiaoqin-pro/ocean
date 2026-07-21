"""Plot the immutable training history of a finished baseline run."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "segformer_b0_suim_baseline.yaml")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    experiment_dir = PROJECT_ROOT / config["experiment"]["output_dir"]
    history = pd.read_csv(experiment_dir / "logs" / "train_history.csv")
    validation = history.dropna(subset=["val_miou"])
    best = validation.loc[validation["val_miou"].idxmax()]
    output_dir = experiment_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(history["epoch"], history["train_loss"], color="#2c7fb8", label="train loss")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(validation["epoch"], validation["val_miou"], marker="o", color="#238b45", label="validation mIoU")
    axes[1].axvline(best["epoch"], color="#d95f0e", linestyle="--", label=f"best epoch {int(best['epoch'])}")
    axes[1].scatter([best["epoch"]], [best["val_miou"]], color="#d95f0e", zorder=3)
    axes[1].annotate(f"{best['val_miou']:.4f}", (best["epoch"], best["val_miou"]), xytext=(-45, 12), textcoords="offset points")
    axes[1].set_ylabel("mIoU")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    axes[2].plot(history["epoch"], [config["training"]["learning_rate"]] * len(history), color="#756bb1", label="learning rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning rate")
    axes[2].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    figure.suptitle("SegFormer-B0 SUIM baseline training history")
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(figure)
    pd.DataFrame([{"best_epoch": int(best["epoch"]), "best_val_miou": float(best["val_miou"]), "epoch_100_val_miou": float(validation.iloc[-1]["val_miou"])}]).to_csv(output_dir / "training_summary.csv", index=False)


if __name__ == "__main__":
    main()
