"""Diagnostic evaluation on unseen compositions of registered degradations."""
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
from scripts.evaluate_sadr import SADRModel  # noqa: E402
from scripts.train_sadr import FORMAT, build_model  # noqa: E402


def compose_degradations(registry: dict[str, object], names: tuple[str, ...]):
    transforms = [build_image_degradation(registry[name]) for name in names]

    def apply(image: np.ndarray, sample_id: str) -> np.ndarray:
        result = image.copy()
        for transform in transforms:
            result = transform(result, sample_id)
        return result

    return apply


def evaluate_callable(model: torch.nn.Module, csv_path: Path, image_degradation, image_size: int, device: torch.device) -> float:
    dataset = SUIMDataset(csv_path, transform=build_eval_transform(image_size), image_degradation=image_degradation)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    confusion = torch.zeros(8, 8, dtype=torch.int64, device=device)
    with torch.no_grad():
        for batch in loader:
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(pixel_values=pixels).logits
            prediction = functional.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
            valid = labels.ne(255)
            indices = labels[valid] * 8 + prediction[valid]
            confusion += torch.bincount(indices, minlength=64).reshape(8, 8)
    return mean_iou(confusion.cpu().numpy())[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sadr_long.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/sadr_long/unseen_compositions.json")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8")); data = config["data"]
    csv_path = ROOT / str(data["confirmation_csv"])
    if csv_path.name != "confirmation.csv" or len(pd.read_csv(csv_path)) != 511:
        raise PermissionError("Unseen diagnostic accepts only the frozen 511-image confirmation role.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device("cuda")
    conditions = {condition.name: condition for condition in load_conditions(ROOT / str(data["degradation_registry"]))}
    checkpoint = ROOT / str(config["experiment"]["output_dir"]) / "formal/checkpoints/final.pt"
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("checkpoint_format") != FORMAT or payload.get("epoch") != int(config["training"]["epochs"]):
        raise ValueError("Invalid SADR checkpoint.")
    if any(bool(payload.get(key, True)) for key in ("calibration_evaluated", "confirmation_evaluated", "official_suim_test_evaluated")):
        raise ValueError("SADR checkpoint records prohibited evaluation access.")
    base, front = build_model(config, device); front.load_state_dict(payload["model_state_dict"]); front.eval(); base.eval(); sadr = SADRModel(base, front).eval()
    compositions = {
        "color_s2+turbidity_s2": ("color_s2", "turbidity_s2"),
        "turbidity_s2+color_s2": ("turbidity_s2", "color_s2"),
        "lowlight_s2+blur_s2": ("lowlight_s2", "blur_s2"),
        "blur_s2+lowlight_s2": ("blur_s2", "lowlight_s2"),
        "color_s3+lowlight_s2": ("color_s3", "lowlight_s2"),
        "lowlight_s2+color_s3": ("lowlight_s2", "color_s3"),
        "turbidity_s3+blur_s2": ("turbidity_s3", "blur_s2"),
        "blur_s2+turbidity_s3": ("blur_s2", "turbidity_s3"),
    }
    rows = []
    for name, members in compositions.items():
        degradation = compose_degradations(conditions, members)
        baseline = evaluate_callable(base, csv_path, degradation, int(data["evaluation_image_size"]), device)
        adapted = evaluate_callable(sadr, csv_path, degradation, int(data["evaluation_image_size"]), device)
        row = {"condition": name, "baseline_miou": baseline, "sadr_miou": adapted, "gain_pp": 100.0 * (adapted - baseline)}
        rows.append(row); print(json.dumps(row, sort_keys=True), flush=True)
    summary = {"split": "confirmation_diagnostic_unseen_compositions", "images": 511, "rows": rows, "mean_gain_pp": float(np.mean([row["gain_pp"] for row in rows]))}
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True); args.output.resolve().write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mean_gain_pp": summary["mean_gain_pp"], "conditions": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
