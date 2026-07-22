"""Run the frozen SUIM v2 checkpoint under fixed, image-only degradations.

Only the formal validation and calibration splits are accepted.  This script never
loads the official test CSV, never updates model weights, and writes all generated
artefacts below the configured output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.label_mapping import CLASS_NAMES, ID2LABEL, LABEL2ID, index_mask_to_rgb  # noqa: E402
from datasets.suim_dataset import IMAGENET_MEAN, IMAGENET_STD, SUIMDataset, build_eval_transform  # noqa: E402
from degradations.registry import Condition, build_image_degradation, load_conditions  # noqa: E402
from metrics.segmentation import confusion_matrix, metrics_from_confusion_matrix  # noqa: E402
from scripts.evaluate_baseline import CalibrationAccumulator, json_safe, resize_logits  # noqa: E402
from utils.reproducibility import set_seed  # noqa: E402


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass
class ErrorDetectionAccumulator:
    confidences: list[np.ndarray] = field(default_factory=list)
    errors: list[np.ndarray] = field(default_factory=list)

    def update(self, probabilities: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> None:
        prediction = probabilities.argmax(dim=1)
        confidence = probabilities.max(dim=1).values
        valid = labels.ne(ignore_index)
        self.confidences.append(confidence[valid].detach().float().cpu().numpy())
        self.errors.append(prediction[valid].ne(labels[valid]).detach().cpu().numpy())

    def metrics(self) -> dict[str, float]:
        confidence = np.concatenate(self.confidences)
        errors = np.concatenate(self.errors).astype(bool)
        uncertainty = 1.0 - confidence
        auroc = float("nan") if errors.min() == errors.max() else float(roc_auc_score(errors, uncertainty))
        order = np.argsort(uncertainty)  # retain most confident pixels first
        cumulative_risk = np.cumsum(errors[order], dtype=np.float64) / np.arange(1, len(errors) + 1)
        coverage = np.arange(1, len(errors) + 1, dtype=np.float64) / len(errors)
        return {
            "error_auroc": auroc,
            "aurc": float(np.trapezoid(cumulative_risk, coverage)),
            "mean_confidence": float(confidence.mean()),
            "mean_correct_confidence": float(confidence[~errors].mean()) if (~errors).any() else float("nan"),
            "mean_wrong_confidence": float(confidence[errors].mean()) if errors.any() else float("nan"),
        }


@torch.no_grad()
def evaluate_condition_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    classes: int,
    ignore_index: int,
) -> dict[str, Any]:
    model.eval()
    matrix = torch.zeros((classes, classes), dtype=torch.long, device=device)
    loss_sum, valid_pixel_count = 0.0, 0
    calibration = CalibrationAccumulator(classes)
    error_detection = ErrorDetectionAccumulator()
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = resize_logits(model(pixel_values=pixels).logits, labels)
            loss = functional.cross_entropy(logits, labels, ignore_index=ignore_index)
            probabilities = torch.softmax(logits, dim=1)
        prediction = logits.argmax(dim=1)
        valid_pixels = int(labels.ne(ignore_index).sum().item())
        loss_sum += float(loss.item()) * valid_pixels
        valid_pixel_count += valid_pixels
        matrix += confusion_matrix(prediction, labels, num_classes=classes, ignore_index=ignore_index).to(device)
        calibration.update(probabilities, labels, ignore_index)
        error_detection.update(probabilities, labels, ignore_index)
    result = metrics_from_confusion_matrix(matrix.cpu())
    result["loss"] = loss_sum / valid_pixel_count if valid_pixel_count else float("nan")
    result.update(calibration.metrics())
    result.update(error_detection.metrics())
    return result


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(image * np.asarray(IMAGENET_STD) + np.asarray(IMAGENET_MEAN), 0, 1)


@torch.no_grad()
def save_examples(
    model: torch.nn.Module,
    dataset: SUIMDataset,
    condition: Condition,
    device: torch.device,
    *,
    amp: bool,
    output_dir: Path,
    count: int,
) -> None:
    if count <= 0:
        return
    destination = output_dir / condition.name
    destination.mkdir(parents=True, exist_ok=True)
    model.eval()
    for index in range(min(count, len(dataset))):
        item = dataset[index]
        pixels = item["pixel_values"].unsqueeze(0).to(device)
        labels = item["labels"]
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(pixel_values=pixels).logits
            logits = functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            probabilities = torch.softmax(logits, dim=1)[0]
        prediction = probabilities.argmax(dim=0).cpu().numpy()
        confidence = probabilities.max(dim=0).values.cpu().numpy()
        figure, axes = plt.subplots(1, 4, figsize=(16, 4))
        panels = [
            (denormalize(item["pixel_values"]), "Degraded image", None),
            (index_mask_to_rgb(labels.numpy()), "Target", None),
            (index_mask_to_rgb(prediction), "Prediction", None),
            (confidence, "Confidence", "magma"),
        ]
        for axis, (panel, title, cmap) in zip(axes, panels):
            axis.imshow(panel, cmap=cmap)
            axis.set_title(title)
            axis.axis("off")
        figure.suptitle(f"{condition.name} | {item['sample_id']}")
        figure.tight_layout()
        figure.savefig(destination / f"{index + 1:02d}_{item['sample_id']}.png", dpi=150, bbox_inches="tight")
        plt.close(figure)


def save_plots(metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    clean = metrics.loc[metrics["condition"] == "clean"]
    for metric_name in ("miou", "ece", "nll", "aurc"):
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
        for axis, split in zip(axes, ("val", "calibration")):
            split_rows = metrics.loc[metrics["split"] == split]
            clean_value = float(clean.loc[clean["split"] == split, metric_name].iloc[0])
            for degradation_type in ("color_attenuation", "turbidity", "lowlight", "blur"):
                rows = split_rows.loc[split_rows["degradation_type"] == degradation_type].sort_values("severity")
                series = pd.concat([pd.DataFrame({"severity": [0], metric_name: [clean_value]}), rows[["severity", metric_name]]])
                axis.plot(series["severity"], series[metric_name], marker="o", label=degradation_type)
            axis.set_title(split)
            axis.set_xlabel("Severity")
            axis.set_ylabel(metric_name)
            axis.set_xticks([0, 1, 2, 3])
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / f"{metric_name}_vs_severity.png", dpi=180)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen SUIM v2 best.pt under fixed degradations.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "degradation_pilot.yaml")
    parser.add_argument("--examples-per-condition", type=int)
    args = parser.parse_args()
    pilot_config_path = args.config.resolve()
    pilot = load_yaml(pilot_config_path)
    experiment = pilot["experiment"]
    allowed_splits = ["val", "calibration"]
    if list(experiment["splits"]) != allowed_splits:
        raise ValueError("The degradation pilot is intentionally restricted to [val, calibration]; official TEST is locked.")
    if not torch.cuda.is_available():
        raise RuntimeError("This evaluator is configured for CUDA.")

    baseline_config_path = PROJECT_ROOT / experiment["baseline_config"]
    baseline = load_yaml(baseline_config_path)
    data, model_config, training, loss_config = baseline["data"], baseline["model"], baseline["training"], baseline["loss"]
    checkpoint_path = PROJECT_ROOT / experiment["checkpoint"]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Frozen baseline checkpoint not found: {checkpoint_path}")
    conditions = load_conditions(pilot_config_path)
    set_seed(int(experiment["seed"]))
    device = torch.device("cuda")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_config["pretrained_model"], num_labels=data["num_classes"], id2label=ID2LABEL,
        label2id=LABEL2ID, ignore_mismatched_sizes=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    output_dir = PROJECT_ROOT / experiment["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pilot_config_path, output_dir / "degradation_config.yaml")
    split_dir = PROJECT_ROOT / data["split_dir"]
    examples_per_condition = int(args.examples_per_condition if args.examples_per_condition is not None else experiment.get("examples_per_condition", 0))
    raw_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for condition in conditions:
        degradation = build_image_degradation(condition)
        for split_name in allowed_splits:
            dataset = SUIMDataset(
                split_dir / f"{split_name}.csv",
                transform=build_eval_transform(data["image_size"]),
                image_degradation=degradation,
            )
            loader = DataLoader(dataset, batch_size=data["batch_size"], shuffle=False, num_workers=data["num_workers"], pin_memory=True)
            metrics = evaluate_condition_split(
                model, loader, device, amp=bool(training["amp"]), classes=data["num_classes"], ignore_index=loss_config["ignore_index"],
            )
            row = {
                "condition": condition.name,
                "degradation_type": condition.degradation_type,
                "severity": condition.severity,
                "split": split_name,
                **{key: metrics[key] for key in (
                    "miou", "pixel_accuracy", "mean_accuracy", "mean_dice", "loss", "nll", "brier_score", "ece",
                    "error_auroc", "aurc", "mean_confidence", "mean_correct_confidence", "mean_wrong_confidence",
                )},
            }
            raw_rows.append(row)
            for class_id, class_name in enumerate(CLASS_NAMES):
                per_class_rows.append({
                    "condition": condition.name,
                    "degradation_type": condition.degradation_type,
                    "severity": condition.severity,
                    "split": split_name,
                    "class_id": class_id,
                    "class_name": class_name,
                    "iou": metrics["per_class_iou"][class_id],
                    "dice": metrics["per_class_dice"][class_id],
                    "accuracy": metrics["per_class_accuracy"][class_id],
                    "classwise_ece": metrics["classwise_ece"][class_id],
                })
            print(f"{condition.name} {split_name}: mIoU={metrics['miou']:.4f} ECE={metrics['ece']:.4f} AURC={metrics['aurc']:.4f}")
            if split_name == "val":
                save_examples(model, dataset, condition, device, amp=bool(training["amp"]), output_dir=output_dir / "examples", count=examples_per_condition)

    raw_metrics = pd.DataFrame(raw_rows)
    raw_metrics.to_csv(output_dir / "raw_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(output_dir / "per_class_metrics.csv", index=False)
    save_plots(raw_metrics, output_dir)
    metadata = {
        "git_commit": git_revision(),
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "baseline_config": str(baseline_config_path.relative_to(PROJECT_ROOT)),
        "baseline_config_sha256": sha256_file(baseline_config_path),
        "degradation_config": str(pilot_config_path.relative_to(PROJECT_ROOT)),
        "degradation_config_sha256": sha256_file(pilot_config_path),
        "splits_evaluated": allowed_splits,
        "official_test_evaluated": False,
        "condition_count": len(conditions),
        "model_retrained": False,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(metadata), handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
