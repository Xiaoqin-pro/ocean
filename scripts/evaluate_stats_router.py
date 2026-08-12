"""Evaluate a raw-image-statistics domain router over frozen source F4 experts."""
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
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.suim_dataset import IMAGENET_MEAN, IMAGENET_STD, SUIMDataset, build_eval_transform  # noqa: E402
from degradations.registry import build_image_degradation, load_conditions  # noqa: E402
from scripts.train_dts_seg_gate0 import build_model as build_suim_f4  # noqa: E402
from scripts.train_uiis_scdi_replication import build_models as build_uiis_f4  # noqa: E402


def image_features(array: np.ndarray) -> np.ndarray:
    small = np.asarray(Image.fromarray(array.astype(np.uint8)).resize((16, 16), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    normalized = (small - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
    pooled = normalized.reshape(16, 16, 3).transpose(2, 0, 1)
    return np.concatenate((pooled.mean((1, 2)), pooled.std((1, 2)), pooled[:, ::4, ::4].reshape(-1))).astype(np.float32)


def train_router(config: dict) -> LogisticRegression:
    frames = []
    for domain, csv_name in enumerate(("suim_train_csv", "uiis_train_csv")):
        frame = pd.read_csv(ROOT / str(config["data"][csv_name])); frame["domain"] = domain; frames.append(frame)
    combined = pd.concat(frames, ignore_index=True); values = []; labels = []
    for row in combined.itertuples(index=False):
        with Image.open(ROOT / str(row.image_path)) as image: values.append(image_features(np.asarray(image.convert("RGB")))); labels.append(int(row.domain))
    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)); classifier.fit(np.asarray(values), np.asarray(labels)); return classifier


def confusion(labels: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    valid = labels.ne(255); indices = labels[valid] * 8 + prediction[valid]; return torch.bincount(indices, minlength=64).reshape(8, 8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/dcpt_fast.yaml"); parser.add_argument("--split", choices=("suim_official", "confirmation"), default="suim_official"); parser.add_argument("--allow-confirmation", action="store_true"); args = parser.parse_args(); config = yaml.safe_load(args.config.read_text(encoding="utf-8")); data = config["data"]
    if args.split == "confirmation" and not args.allow_confirmation: raise PermissionError("Confirmation is locked.")
    csv_path = ROOT / str(data["suim_test_csv"] if args.split == "suim_official" else data["uiis_confirmation_csv"]); expected = 110 if args.split == "suim_official" else 511
    if len(pd.read_csv(csv_path)) != expected: raise PermissionError("Unexpected frozen split size.")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required.")
    classifier = train_router(config); device = torch.device("cuda"); suim_cfg = yaml.safe_load((ROOT / "configs/dts_seg_gate0.yaml").read_text(encoding="utf-8")); uiis_cfg = yaml.safe_load((ROOT / "configs/uiis_scdi_replication.yaml").read_text(encoding="utf-8")); suim_payload = torch.load(ROOT / "outputs/dts_seg_gate0/formal/F4/checkpoints/final.pt", map_location=device, weights_only=False); uiis_payload = torch.load(ROOT / "outputs/uiis_scdi_replication/formal/F4/checkpoints/final.pt", map_location=device, weights_only=False); suim = build_suim_f4(suim_cfg, device); uiis = build_uiis_f4(uiis_cfg, "F4", device); suim.load_state_dict(suim_payload["model_state_dict"]); uiis.load_state_dict(uiis_payload["model_state_dict"]); suim.eval(); uiis.eval()
    conditions = load_conditions(ROOT / str(data["degradation_registry"])); rows = []
    for condition in conditions:
        dataset = SUIMDataset(csv_path, transform=build_eval_transform(int(data.get("evaluation_image_size", 384))), image_degradation=build_image_degradation(condition)); loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True); matrices = {name: torch.zeros(8, 8, dtype=torch.int64, device=device) for name in ("SUIM", "UIIS", "stats")}; selected = []
        with torch.no_grad():
            for batch in loader:
                pixels = batch["pixel_values"].to(device); labels = batch["labels"].to(device); clean = pixels.detach().cpu().numpy(); clean = clean * np.asarray(IMAGENET_STD, dtype=np.float32)[None, :, None, None] + np.asarray(IMAGENET_MEAN, dtype=np.float32)[None, :, None, None]; clean = np.clip(clean.transpose(0, 2, 3, 1) * 255.0, 0, 255).astype(np.uint8); predicted_domain = torch.from_numpy(classifier.predict(np.asarray([image_features(value) for value in clean]))).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16): first = suim(pixel_values=pixels).logits; second = uiis(pixel_values=pixels).logits
                first = functional.interpolate(first.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False).argmax(1); second = functional.interpolate(second.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False).argmax(1); matrices["SUIM"] += confusion(labels, first); matrices["UIIS"] += confusion(labels, second); prediction = torch.where(predicted_domain[:, None, None].eq(0), first, second); matrices["stats"] += confusion(labels, prediction); selected.extend(predicted_domain.cpu().tolist())
        row = {"condition": condition.name, "selected_suim_fraction": float(np.mean(np.asarray(selected) == 0))};
        for name, matrix in matrices.items():
            array = matrix.cpu().numpy(); diagonal = array.diagonal(); denominator = array.sum(0) + array.sum(1) - diagonal; valid = denominator > 0; row[f"{name}_miou"] = float((diagonal / denominator.clip(min=1))[valid].mean())
        rows.append(row); print(json.dumps(row, sort_keys=True), flush=True)
    frame = pd.DataFrame(rows); output = ROOT / str(config["experiment"]["output_dir"]) / f"stats_router_{args.split}"; output.mkdir(parents=True, exist_ok=True); frame.to_csv(output / "metrics.csv", index=False); summary = {"split": args.split, "rows": len(frame), "mean_miou": {key: float(frame[key].mean()) for key in ("SUIM_miou", "UIIS_miou", "stats_miou")}, "mean_selected_suim_fraction": float(frame.selected_suim_fraction.mean())}; (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8"); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
