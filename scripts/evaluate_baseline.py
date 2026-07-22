"""Evaluate the frozen SegFormer-B0 baseline without touching the official test split.

Outputs are written below the experiment directory and include aggregate metrics,
per-class metrics, confusion matrices, and deterministic validation examples.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.label_mapping import CLASS_NAMES, ID2LABEL, LABEL2ID, index_mask_to_rgb  # noqa: E402
from datasets.suim_dataset import IMAGENET_MEAN, IMAGENET_STD, SUIMDataset, build_eval_transform  # noqa: E402
from metrics.segmentation import confusion_matrix, metrics_from_confusion_matrix  # noqa: E402
from utils.reproducibility import set_seed  # noqa: E402


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resize_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)


@dataclass
class CalibrationAccumulator:
    classes: int
    bins: int = 15

    def __post_init__(self) -> None:
        self.count = 0
        self.nll_sum = 0.0
        self.brier_sum = 0.0
        self.bin_count = np.zeros(self.bins, dtype=np.int64)
        self.bin_confidence = np.zeros(self.bins, dtype=np.float64)
        self.bin_correct = np.zeros(self.bins, dtype=np.float64)
        self.class_bin_count = np.zeros((self.classes, self.bins), dtype=np.int64)
        self.class_bin_probability = np.zeros((self.classes, self.bins), dtype=np.float64)
        self.class_bin_event = np.zeros((self.classes, self.bins), dtype=np.float64)

    def update(self, probabilities: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> None:
        probs = probabilities.permute(0, 2, 3, 1).reshape(-1, self.classes).detach().float().cpu().numpy()
        target = labels.reshape(-1).detach().cpu().numpy()
        valid = (target != ignore_index) & (target >= 0) & (target < self.classes)
        probs, target = probs[valid], target[valid]
        if not len(target):
            return
        prediction = probs.argmax(axis=1)
        confidence = probs.max(axis=1)
        self.count += len(target)
        self.nll_sum += float(-np.log(np.clip(probs[np.arange(len(target)), target], 1e-12, 1.0)).sum())
        one_hot = np.eye(self.classes, dtype=np.float32)[target]
        self.brier_sum += float(np.square(probs - one_hot).sum(axis=1).sum())
        bin_index = np.minimum((confidence * self.bins).astype(int), self.bins - 1)
        for bin_id in range(self.bins):
            selected = bin_index == bin_id
            if selected.any():
                self.bin_count[bin_id] += int(selected.sum())
                self.bin_confidence[bin_id] += float(confidence[selected].sum())
                self.bin_correct[bin_id] += float((prediction[selected] == target[selected]).sum())
        for class_id in range(self.classes):
            class_probability = probs[:, class_id]
            class_bin = np.minimum((class_probability * self.bins).astype(int), self.bins - 1)
            class_event = target == class_id
            for bin_id in range(self.bins):
                selected = class_bin == bin_id
                if selected.any():
                    self.class_bin_count[class_id, bin_id] += int(selected.sum())
                    self.class_bin_probability[class_id, bin_id] += float(class_probability[selected].sum())
                    self.class_bin_event[class_id, bin_id] += float(class_event[selected].sum())

    def metrics(self) -> dict[str, Any]:
        if not self.count:
            return {"nll": float("nan"), "brier_score": float("nan"), "ece": float("nan"), "classwise_ece": []}
        nonempty = self.bin_count > 0
        confidence = np.zeros(self.bins)
        accuracy = np.zeros(self.bins)
        confidence[nonempty] = self.bin_confidence[nonempty] / self.bin_count[nonempty]
        accuracy[nonempty] = self.bin_correct[nonempty] / self.bin_count[nonempty]
        ece = float(np.sum((self.bin_count / self.count) * np.abs(accuracy - confidence)))
        classwise: list[float] = []
        for class_id in range(self.classes):
            counts = self.class_bin_count[class_id]
            populated = counts > 0
            probability = np.zeros(self.bins)
            event = np.zeros(self.bins)
            probability[populated] = self.class_bin_probability[class_id, populated] / counts[populated]
            event[populated] = self.class_bin_event[class_id, populated] / counts[populated]
            classwise.append(float(np.sum((counts / self.count) * np.abs(event - probability))))
        return {
            "nll": self.nll_sum / self.count,
            "brier_score": self.brier_sum / self.count,
            "ece": ece,
            "classwise_ece": classwise,
            "ece_bins": self.bins,
        }


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module, dataset: SUIMDataset, loader: DataLoader, device: torch.device,
    amp: bool, classes: int, ignore_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    matrix = torch.zeros((classes, classes), dtype=torch.long, device=device)
    loss_sum, valid_pixel_count = 0.0, 0
    calibration = CalibrationAccumulator(classes)
    per_image: list[dict[str, Any]] = []
    for batch in loader:
        pixels, labels = batch["pixel_values"].to(device), batch["labels"].to(device)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = resize_logits(model(pixel_values=pixels).logits, labels)
            loss = functional.cross_entropy(logits, labels, ignore_index=ignore_index)
            probabilities = torch.softmax(logits, dim=1)
        prediction = logits.argmax(dim=1)
        valid = labels.ne(ignore_index)
        pixels_in_batch = int(valid.sum().item())
        loss_sum += float(loss.item()) * pixels_in_batch
        valid_pixel_count += pixels_in_batch
        matrix += confusion_matrix(prediction, labels, num_classes=classes, ignore_index=ignore_index).to(device)
        calibration.update(probabilities, labels, ignore_index)
        for image_id, image_prediction, image_target in zip(batch["sample_id"], prediction, labels):
            image_matrix = confusion_matrix(image_prediction, image_target, num_classes=classes, ignore_index=ignore_index)
            per_image.append({"sample_id": str(image_id), "miou": metrics_from_confusion_matrix(image_matrix)["miou"]})
    result = metrics_from_confusion_matrix(matrix.cpu())
    result["loss"] = loss_sum / valid_pixel_count if valid_pixel_count else float("nan")
    result.update(calibration.metrics())
    return result, per_image


def json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.cpu().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def save_metrics(metrics: dict[str, Any], output_dir: Path, split_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{split_name}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(metrics), handle, ensure_ascii=False, indent=2)
    table = pd.DataFrame({
        "class_id": range(len(CLASS_NAMES)),
        "class_name": CLASS_NAMES,
        "iou": metrics["per_class_iou"],
        "dice": metrics["per_class_dice"],
        "accuracy": metrics["per_class_accuracy"],
        "classwise_ece": metrics["classwise_ece"],
    })
    table.to_csv(output_dir / f"{split_name}_per_class_metrics.csv", index=False)


def save_confusion_matrix(metrics: dict[str, Any], output_path: Path, title: str) -> None:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046)
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_xlabel("Prediction")
    axis.set_ylabel("Ground truth")
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center", fontsize=7)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip((image * np.asarray(IMAGENET_STD)) + np.asarray(IMAGENET_MEAN), 0, 1)


@torch.no_grad()
def save_validation_examples(
    model: torch.nn.Module, dataset: SUIMDataset, sample_scores: list[dict[str, Any]], device: torch.device,
    amp: bool, output_dir: Path, examples_per_group: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(sample_scores).sort_values(["miou", "sample_id"], kind="stable").reset_index(drop=True)
    groups = {
        "worst": scores.head(examples_per_group),
        "middle": scores.iloc[max(0, len(scores) // 2 - examples_per_group // 2): len(scores) // 2 + (examples_per_group + 1) // 2],
        "best": scores.tail(examples_per_group).iloc[::-1],
    }
    lookup = {str(row.sample_id): index for index, row in dataset.samples.iterrows()}
    model.eval()
    for group, rows in groups.items():
        for rank, row in enumerate(rows.itertuples(index=False), start=1):
            item = dataset[lookup[row.sample_id]]
            pixels = item["pixel_values"].unsqueeze(0).to(device)
            labels = item["labels"]
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(pixel_values=pixels).logits
                logits = functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                probabilities = torch.softmax(logits, dim=1)[0]
            prediction = probabilities.argmax(dim=0).cpu().numpy()
            target = labels.cpu().numpy()
            confidence = probabilities.max(dim=0).values.cpu().numpy()
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=0).cpu().numpy()
            correct = prediction == target
            correct_map = np.zeros((*correct.shape, 3), dtype=np.uint8)
            correct_map[correct] = (44, 160, 44)
            correct_map[~correct] = (210, 45, 45)
            figure, axes = plt.subplots(1, 6, figsize=(24, 4.4))
            panels = [
                (denormalize(item["pixel_values"]), "Image", None),
                (index_mask_to_rgb(target), "Target", None),
                (index_mask_to_rgb(prediction), "Prediction", None),
                (correct_map, "Correct / error", None),
                (confidence, "Confidence", "magma"),
                (entropy, "Entropy", "viridis"),
            ]
            for axis, (panel, name, cmap) in zip(axes, panels):
                axis.imshow(panel, cmap=cmap)
                axis.set_title(name)
                axis.axis("off")
            figure.suptitle(f"{group} #{rank:02d} | {row.sample_id} | per-image mIoU={row.miou:.4f}", y=1.02)
            figure.tight_layout()
            figure.savefig(output_dir / f"{group}_{rank:02d}_{row.sample_id}.png", dpi=150, bbox_inches="tight")
            plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen SUIM baseline on validation and calibration only.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "segformer_b0_suim_baseline.yaml")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--examples-per-group", type=int, default=10)
    args = parser.parse_args()
    config = load_config(args.config)
    data, model_config, training, loss_config = config["data"], config["model"], config["training"], config["loss"]
    if not torch.cuda.is_available():
        raise RuntimeError("This evaluator is configured for CUDA.")
    set_seed(config["experiment"]["seed"])
    device = torch.device("cuda")
    experiment_dir = PROJECT_ROOT / config["experiment"]["output_dir"]
    checkpoint_path = args.checkpoint or experiment_dir / "checkpoints" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_config["pretrained_model"], num_labels=data["num_classes"], id2label=ID2LABEL,
        label2id=LABEL2ID, ignore_mismatched_sizes=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    split_dir = PROJECT_ROOT / data["split_dir"]
    output_dir = experiment_dir / "evaluation"
    metadata = {
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_best_miou": checkpoint["best_miou"],
        "splits_evaluated": ["val", "calibration"],
        "official_test_evaluated": False,
        "image_size": data["image_size"],
        "num_classes": data["num_classes"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("val", "calibration"):
        dataset = SUIMDataset(split_dir / f"{split_name}.csv", transform=build_eval_transform(data["image_size"]))
        loader = DataLoader(dataset, batch_size=data["batch_size"], shuffle=False, num_workers=data["num_workers"], pin_memory=True)
        metrics, sample_scores = evaluate_split(model, dataset, loader, device, bool(training["amp"]), data["num_classes"], loss_config["ignore_index"])
        save_metrics(metrics, output_dir, split_name)
        save_confusion_matrix(metrics, output_dir / f"{split_name}_confusion_matrix.png", f"{split_name} confusion matrix")
        print(f"{split_name}: loss={metrics['loss']:.4f}, mIoU={metrics['miou']:.4f}, pixel_accuracy={metrics['pixel_accuracy']:.4f}, ECE={metrics['ece']:.4f}")
        if split_name == "val":
            pd.DataFrame(sample_scores).sort_values("miou").to_csv(output_dir / "val_per_image_miou.csv", index=False)
            save_validation_examples(model, dataset, sample_scores, device, bool(training["amp"]), output_dir / "val_examples", args.examples_per_group)
    with (output_dir / "evaluation_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
