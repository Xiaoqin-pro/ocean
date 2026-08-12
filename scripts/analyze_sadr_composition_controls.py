"""Audit compositionality across degradation families and trivial nulls."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.suim_dataset import SUIMDataset, build_eval_transform  # noqa: E402
from degradations.registry import build_image_degradation, load_conditions  # noqa: E402
from scripts.evaluate_sadr import SADRModel  # noqa: E402
from scripts.train_sadr import FORMAT, build_model  # noqa: E402


FAMILIES = ("color", "turbidity", "lowlight", "blur")
SEVERITIES = (1, 2, 3)


def compose_degradations(registry: dict[str, object], names: tuple[str, ...]):
    transforms = [build_image_degradation(registry[name]) for name in names]

    def apply(image: np.ndarray, sample_id: str) -> np.ndarray:
        result = image.copy()
        for transform in transforms:
            result = transform(result, sample_id)
        return result

    return apply


def correction(base: torch.nn.Module, front: torch.nn.Module, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        raw = base(pixel_values=pixels).logits.float()
        restored, residual = front(pixels)
        corrected = base(pixel_values=restored).logits.float()
    return corrected - raw, residual.float(), raw, corrected


def update_confusion(target: torch.Tensor, logits: torch.Tensor, confusion: torch.Tensor) -> None:
    prediction = torch.nn.functional.interpolate(logits.float(), size=target.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
    valid = target.ne(255)
    indices = target[valid] * 8 + prediction[valid]
    confusion += torch.bincount(indices, minlength=64).reshape(8, 8)


def confusion_miou(confusion: torch.Tensor) -> float:
    matrix = confusion.cpu().numpy().astype(np.float64)
    diagonal = np.diag(matrix)
    denominator = matrix.sum(0) + matrix.sum(1) - diagonal
    valid = denominator > 0
    return float(np.divide(diagonal, denominator, out=np.zeros_like(diagonal), where=valid)[valid].mean())


def relation_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left_flat = left.flatten(1)
    right_flat = right.flatten(1)
    cosine = torch.nn.functional.cosine_similarity(left_flat, right_flat, dim=1)
    relative_error = (left_flat - right_flat).norm(dim=1) / (left_flat.norm(dim=1) + right_flat.norm(dim=1) + 1e-6)
    return {"cosine": float(cosine.mean()), "relative_error": float(relative_error.mean())}


def pixel_order_metrics(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    first_flat = first.flatten(1)
    second_flat = second.flatten(1)
    cosine = torch.nn.functional.cosine_similarity(first_flat, second_flat, dim=1)
    mae = (first_flat - second_flat).abs().mean(dim=1)
    scale = first_flat.abs().mean(dim=1) + second_flat.abs().mean(dim=1) + 1e-6
    return {"cosine": float(cosine.mean()), "mean_abs_difference": float(mae.mean()), "relative_l1": float((mae / scale).mean())}


def build_variant(config_path: Path, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload = torch.load(ROOT / str(config["experiment"]["output_dir"]) / "formal/checkpoints/final.pt", map_location=device, weights_only=False)
    if payload.get("checkpoint_format") != FORMAT or payload.get("epoch") != int(config["training"]["epochs"]):
        raise ValueError(f"Invalid checkpoint for {config_path}.")
    if any(bool(payload.get(key, True)) for key in ("calibration_evaluated", "confirmation_evaluated", "official_suim_test_evaluated")):
        raise ValueError(f"Checkpoint records prohibited evaluation access: {config_path}")
    base, front = build_model(config, device)
    front.load_state_dict(payload["model_state_dict"])
    base.eval(); front.eval()
    return base, front


def relation_specs() -> list[tuple[str, str, str, str, str]]:
    specs = []
    for severity in SEVERITIES:
        for left_index, left_family in enumerate(FAMILIES):
            for right_family in FAMILIES[left_index + 1 :]:
                left = f"{left_family}_s{severity}"
                right = f"{right_family}_s{severity}"
                specs.append((f"{left}+{right}", left, right, left, right))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sadr_long.yaml")
    parser.add_argument("--split", choices=("confirmation", "calibration"), default="confirmation")
    parser.add_argument("--max-images", type=int, default=128)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/sadr_long/composition_controls.json")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    data = config["data"]
    csv_path = ROOT / str(data[f"{args.split}_csv"])
    frame = pd.read_csv(csv_path)
    expected = {"confirmation": 511, "calibration": 508}[args.split]
    if len(frame) != expected or csv_path.name != f"{args.split}.csv":
        raise PermissionError(f"Composition controls require the frozen {args.split} split with {expected} images.")
    if args.max_images <= 0 or args.max_images > len(frame):
        raise ValueError(f"max-images must be in [1, {len(frame)}].")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device("cuda")
    registry = {condition.name: condition for condition in load_conditions(ROOT / str(data["degradation_registry"]))}
    transform = build_eval_transform(int(data["evaluation_image_size"]))
    base, front = build_variant(args.config.resolve(), device)
    rows = []
    for label, left_name, right_name, _, _ in relation_specs():
        names = {
            "A": (left_name,),
            "B": (right_name,),
            "AB": (left_name, right_name),
            "BA": (right_name, left_name),
        }
        loaders = {
            key: DataLoader(
                SUIMDataset(csv_path, transform=transform, image_degradation=compose_degradations(registry, members)),
                batch_size=8, shuffle=False, num_workers=0, pin_memory=True,
            ) for key, members in names.items()
        }
        iterators = {key: iter(loader) for key, loader in loaders.items()}
        accum = {key: [] for key in ("add", "order", "same_a", "same_b", "shuffled_add", "pixel_order")}
        raw_ab_confusion = torch.zeros(8, 8, dtype=torch.int64, device=device)
        sadr_ab_confusion = torch.zeros(8, 8, dtype=torch.int64, device=device)
        raw_ba_confusion = torch.zeros(8, 8, dtype=torch.int64, device=device)
        sadr_ba_confusion = torch.zeros(8, 8, dtype=torch.int64, device=device)
        seen = 0
        while seen < args.max_images:
            batches = {key: next(iterator) for key, iterator in iterators.items()}
            take = min(8, args.max_images - seen)
            pixels = {key: batches[key]["pixel_values"][:take].to(device, non_blocking=True) for key in batches}
            targets = {key: batches[key]["labels"][:take].to(device, non_blocking=True) for key in batches}
            corrections = {key: correction(base, front, value) for key, value in pixels.items()}
            delta_a = corrections["A"][0]
            delta_b = corrections["B"][0]
            delta_ab = corrections["AB"][0]
            shuffled_b = delta_b.roll(shifts=1, dims=0) if take > 1 else delta_b
            accum["add"].append(relation_metrics(delta_ab, delta_a + delta_b))
            accum["order"].append(relation_metrics(delta_ab, corrections["BA"][0]))
            accum["same_a"].append(relation_metrics(delta_ab, delta_a))
            accum["same_b"].append(relation_metrics(delta_ab, delta_b))
            accum["shuffled_add"].append(relation_metrics(delta_ab, delta_a + shuffled_b))
            accum["pixel_order"].append(pixel_order_metrics(pixels["AB"], pixels["BA"]))
            update_confusion(targets["AB"], corrections["AB"][2], raw_ab_confusion)
            update_confusion(targets["AB"], corrections["AB"][3], sadr_ab_confusion)
            update_confusion(targets["BA"], corrections["BA"][2], raw_ba_confusion)
            update_confusion(targets["BA"], corrections["BA"][3], sadr_ba_confusion)
            seen += take
        summary = {}
        for key, values in accum.items():
            metric_names = tuple(values[0].keys())
            summary[key] = {metric: float(np.mean([row[metric] for row in values])) for metric in metric_names}
        rows.append({"relation": label, "images": seen, "metrics": summary, "miou": {"raw_ab": confusion_miou(raw_ab_confusion), "sadr_ab": confusion_miou(sadr_ab_confusion), "gain_ab_pp": 100.0 * (confusion_miou(sadr_ab_confusion) - confusion_miou(raw_ab_confusion)), "raw_ba": confusion_miou(raw_ba_confusion), "sadr_ba": confusion_miou(sadr_ba_confusion), "gain_ba_pp": 100.0 * (confusion_miou(sadr_ba_confusion) - confusion_miou(raw_ba_confusion))}})
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
    payload = {
        "split": args.split,
        "images": args.max_images,
        "relations": rows,
        "relation_count": len(rows),
        "note": "Matched-severity family-pair audit. Semantic metrics are post-freeze diagnostics; operator-order metrics quantify pixel-level commutativity.",
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
