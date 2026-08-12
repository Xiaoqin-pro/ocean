"""Evaluate label-free per-image selectors between frozen F4 and SADR.

The selectors use only the two models' predictions on the current image. They
are not fitted on calibration labels: entropy, mean top-1 margin, mean maximum
probability, and conservative conjunctions choose either the raw or restored
logits per image. This is an audit of whether semantic self-consistency can
repair SADR's family-selective transfer without a learned quality gate.
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.suim_dataset import SUIMDataset, build_eval_transform  # noqa: E402
from degradations.registry import build_image_degradation, load_conditions  # noqa: E402
from scripts.evaluate_dts_seg_gate0 import mean_iou  # noqa: E402
from scripts.train_sadr import build_model  # noqa: E402


SELECTORS = ("raw", "sadr", "entropy", "margin", "confidence", "entropy_and_margin")


def selector_mask(raw: torch.Tensor, adapted: torch.Tensor, name: str) -> torch.Tensor:
    raw_prob = raw.float().softmax(dim=1)
    adapted_prob = adapted.float().softmax(dim=1)
    raw_entropy = -(raw_prob.clamp_min(1e-7) * raw_prob.clamp_min(1e-7).log()).sum(1).mean(dim=(-2, -1))
    adapted_entropy = -(adapted_prob.clamp_min(1e-7) * adapted_prob.clamp_min(1e-7).log()).sum(1).mean(dim=(-2, -1))
    raw_top = raw_prob.topk(2, dim=1).values
    adapted_top = adapted_prob.topk(2, dim=1).values
    raw_margin = (raw_top[:, 0] - raw_top[:, 1]).mean(dim=(-2, -1))
    adapted_margin = (adapted_top[:, 0] - adapted_top[:, 1]).mean(dim=(-2, -1))
    raw_confidence = raw_prob.max(1).values.mean(dim=(-2, -1))
    adapted_confidence = adapted_prob.max(1).values.mean(dim=(-2, -1))
    if name == "raw":
        return torch.zeros(raw.shape[0], dtype=torch.bool, device=raw.device)
    if name == "sadr":
        return torch.ones(raw.shape[0], dtype=torch.bool, device=raw.device)
    if name == "entropy":
        return adapted_entropy < raw_entropy
    if name == "margin":
        return adapted_margin > raw_margin
    if name == "confidence":
        return adapted_confidence > raw_confidence
    if name == "entropy_and_margin":
        return (adapted_entropy < raw_entropy) & (adapted_margin > raw_margin)
    raise ValueError(f"Unknown selector: {name}")


def evaluate_condition(model: tuple[torch.nn.Module, torch.nn.Module], csv_path: Path, condition: object, image_size: int, device: torch.device) -> list[dict[str, object]]:
    base, front = model
    dataset = SUIMDataset(csv_path, transform=build_eval_transform(image_size), image_degradation=build_image_degradation(condition))
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    confusion = {name: torch.zeros(8, 8, dtype=torch.int64, device=device) for name in SELECTORS}
    selected = {name: 0 for name in SELECTORS}
    images = 0
    with torch.no_grad():
        for batch in loader:
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                raw = base(pixel_values=pixels).logits
                restored, _ = front(pixels)
                adapted = base(pixel_values=restored).logits
            for name in SELECTORS:
                choose_adapted = selector_mask(raw, adapted, name)
                logits = torch.where(choose_adapted[:, None, None, None], adapted, raw)
                prediction = functional.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
                valid = labels.ne(255)
                indices = labels[valid] * 8 + prediction[valid]
                confusion[name] += torch.bincount(indices, minlength=64).reshape(8, 8)
                selected[name] += int(choose_adapted.sum())
            images += int(pixels.shape[0])
    rows = []
    for name in SELECTORS:
        miou, per_class = mean_iou(confusion[name].cpu().numpy())
        rows.append({"condition": condition.name, "degradation_type": condition.degradation_type, "severity": int(condition.severity), "selector": name, "miou": miou, "selected_fraction": float(selected[name] / max(images, 1)), "images": images, "per_class_iou": per_class})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sadr_long.yaml")
    parser.add_argument("--split", choices=("calibration", "confirmation"), default="calibration")
    parser.add_argument("--allow-confirmation", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.split == "confirmation" and not args.allow_confirmation:
        raise PermissionError("Confirmation self-gate audit is locked; pass --allow-confirmation explicitly.")
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8")); data = config["data"]
    csv_entry = data.get(f"{args.split}_csv") or str(Path(data["confirmation_csv"]).with_name("calibration.csv"))
    csv_path = ROOT / str(csv_entry); expected = {"calibration": 508, "confirmation": 511}[args.split]
    if csv_path.name != f"{args.split}.csv" or len(pd.read_csv(csv_path)) != expected:
        raise PermissionError(f"Expected the frozen {args.split} split with {expected} images.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device("cuda")
    base, front = build_model(config, device)
    payload = torch.load(ROOT / str(config["experiment"]["output_dir"]) / "formal/checkpoints/final.pt", map_location=device, weights_only=False)
    if any(bool(payload.get(key, True)) for key in ("calibration_evaluated", "confirmation_evaluated", "official_suim_test_evaluated")):
        raise ValueError("Checkpoint records prohibited evaluation access.")
    front.load_state_dict(payload["model_state_dict"]); base.eval(); front.eval()
    conditions = load_conditions(ROOT / str(data["degradation_registry"]))
    rows = []
    for condition in conditions:
        rows.extend(evaluate_condition((base, front), csv_path, condition, int(data["evaluation_image_size"]), device))
        print(f"{args.split}/{condition.name}: done", flush=True)
    frame = pd.DataFrame(rows)
    summary_rows = []
    for selector, group in frame.groupby("selector", sort=False):
        summary_rows.append({"selector": selector, "mean_miou": float(group.miou.mean()), "selected_fraction": float(group.selected_fraction.mean()), "images": expected * len(conditions)})
    output = args.output or (ROOT / str(config["experiment"]["output_dir"]) / f"{args.split}_self_gate")
    output.resolve().mkdir(parents=True, exist_ok=True)
    frame.to_json(output / "condition_metrics.json", orient="records", indent=2)
    summary = {"split": args.split, "images": expected, "conditions": len(conditions), "selectors": summary_rows, "note": "Unfitted prediction-only selectors; calibration is an audit and confirmation is a post-freeze diagnostic."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
