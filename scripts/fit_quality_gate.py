"""Fit a calibration-only quality gate between UIIS-F4 and SADR."""
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.suim_dataset import SUIMDataset, build_eval_transform  # noqa: E402
from degradations.registry import build_image_degradation, load_conditions  # noqa: E402
from scripts.train_sadr import build_model  # noqa: E402


def quality_features(pixels: torch.Tensor) -> torch.Tensor:
    mean = pixels.mean(dim=(-2, -1)); std = pixels.std(dim=(-2, -1))
    gray = pixels.mean(1, keepdim=True); gray_mean = gray.mean(dim=(-2, -1)); gray_std = gray.std(dim=(-2, -1))
    dx = gray[..., :, 1:] - gray[..., :, :-1]; dy = gray[..., 1:, :] - gray[..., :-1, :]
    return torch.cat([mean, std, gray_mean, gray_std, dx.abs().mean(dim=(-2, -1)), dy.abs().mean(dim=(-2, -1))], dim=1)


def per_image_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = functional.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
    losses = functional.cross_entropy(logits, labels, ignore_index=255, reduction="none")
    valid = labels.ne(255)
    return (losses * valid).sum((-2, -1)) / valid.sum((-2, -1)).clamp_min(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/sadr_fast.yaml"); args = parser.parse_args(); config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8")); data = config["data"]
    csv_path = ROOT / "data/uiis_processed/splits/uiis_alpha010_confirmation/calibration.csv"; frame = pd.read_csv(csv_path)
    if len(frame) != 508: raise PermissionError("Quality-gate fitting accepts only the frozen 508-image calibration split.")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required.")
    device = torch.device("cuda"); base, front = build_model(config, device); payload = torch.load(ROOT / str(config["experiment"]["output_dir"]) / "formal/checkpoints/final.pt", map_location=device, weights_only=False); front.load_state_dict(payload["model_state_dict"]); base.eval(); front.eval(); conditions = load_conditions(ROOT / str(data["degradation_registry"]))
    features: list[np.ndarray] = []; targets: list[np.ndarray] = []; deltas: list[np.ndarray] = []
    for condition in conditions:
        dataset = SUIMDataset(csv_path, transform=build_eval_transform(int(data["evaluation_image_size"])), image_degradation=build_image_degradation(condition)); loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
        for batch in loader:
            pixels = batch["pixel_values"].to(device, non_blocking=True); labels = batch["labels"].to(device, non_blocking=True)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                base_logits = base(pixel_values=pixels).logits; restored, _ = front(pixels); sadr_logits = base(pixel_values=restored).logits
            base_loss = per_image_ce(base_logits, labels); sadr_loss = per_image_ce(sadr_logits, labels); delta = base_loss - sadr_loss
            features.extend(quality_features(pixels).float().cpu().numpy()); deltas.extend(delta.float().cpu().numpy()); targets.extend((delta > 0.0).long().cpu().numpy()); del pixels, labels, base_logits, restored, sadr_logits
        print(f"fit/{condition.name}: samples={len(features)}", flush=True)
    x = np.asarray(features, dtype=np.float32); y = np.asarray(targets, dtype=np.int64); d = np.asarray(deltas, dtype=np.float32); scaler = StandardScaler().fit(x); clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=0).fit(scaler.transform(x), y); probability = clf.predict_proba(scaler.transform(x))[:, 1]; selected_loss = np.where(probability >= 0.5, d * 0.0, d * 0.0)
    output = ROOT / str(config["experiment"]["output_dir"]) / "quality_gate"; output.mkdir(parents=True, exist_ok=True); gate = {"format": "quality_gate_v1", "feature_mean": scaler.mean_.tolist(), "feature_scale": scaler.scale_.tolist(), "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]), "calibration_images": 508, "conditions": len(conditions), "positive_rate": float(y.mean()), "calibration_logloss_delta_mean": float(d.mean()), "calibration_accuracy": float((clf.predict(scaler.transform(x)) == y).mean())}; (output / "gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8"); np.savez(output / "calibration_gate_data.npz", features=x, target=y, delta=d, probability=probability); print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__": main()
