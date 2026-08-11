"""Evaluate unsupervised confidence routing between frozen SUIM and UIIS experts."""
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
from scripts.train_dts_seg_gate0 import build_model as build_suim_f4  # noqa: E402
from scripts.train_uiis_scdi_replication import build_models as build_uiis_f4  # noqa: E402


def _confusion(labels: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    valid = labels.ne(255); indices = labels[valid] * 8 + prediction[valid]; return torch.bincount(indices, minlength=64).reshape(8, 8)


def evaluate_pair(first: torch.nn.Module, second: torch.nn.Module, csv_path: Path, condition, image_size: int, device: torch.device) -> dict[str, object]:
    dataset = SUIMDataset(csv_path, transform=build_eval_transform(image_size), image_degradation=build_image_degradation(condition)); loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True); matrices = {name: torch.zeros(8, 8, dtype=torch.int64, device=device) for name in ("SUIM", "UIIS", "confidence")}; selected = []
    with torch.no_grad():
        for batch in loader:
            pixels = batch["pixel_values"].to(device); labels = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits_first = first(pixel_values=pixels).logits; logits_second = second(pixel_values=pixels).logits
            logits_first = functional.interpolate(logits_first.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False); logits_second = functional.interpolate(logits_second.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
            pred_first = logits_first.argmax(1); pred_second = logits_second.argmax(1); matrices["SUIM"] += _confusion(labels, pred_first); matrices["UIIS"] += _confusion(labels, pred_second)
            prob_first = logits_first.softmax(1); prob_second = logits_second.softmax(1); confidence_first = prob_first.max(1).values.mean((1, 2)); confidence_second = prob_second.max(1).values.mean((1, 2)); choose_first = confidence_first >= confidence_second; selected.extend(choose_first.cpu().tolist()); prediction = torch.where(choose_first[:, None, None], pred_first, pred_second); matrices["confidence"] += _confusion(labels, prediction)
    result: dict[str, object] = {"condition": condition.name, "severity": int(condition.severity), "selected_suim_fraction": float(np.mean(selected))}
    for name, matrix in matrices.items():
        array = matrix.cpu().numpy(); diagonal = array.diagonal(); denominator = array.sum(0) + array.sum(1) - diagonal; valid = denominator > 0; result[f"{name}_miou"] = float((diagonal / denominator.clip(min=1))[valid].mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/dcpt_fast.yaml"); parser.add_argument("--split", choices=("suim_official", "confirmation"), default="suim_official"); parser.add_argument("--allow-confirmation", action="store_true"); args = parser.parse_args(); config = yaml.safe_load(args.config.read_text(encoding="utf-8")); data = config["data"]
    if args.split == "confirmation" and not args.allow_confirmation: raise PermissionError("Confirmation is locked.")
    csv_path = ROOT / str(data["suim_test_csv"] if args.split == "suim_official" else data["uiis_confirmation_csv"]); expected = 110 if args.split == "suim_official" else 511
    if len(pd.read_csv(csv_path)) != expected: raise PermissionError("Unexpected frozen split size.")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required.")
    device = torch.device("cuda"); suim_cfg = yaml.safe_load((ROOT / "configs/dts_seg_gate0.yaml").read_text(encoding="utf-8")); uiis_cfg = yaml.safe_load((ROOT / "configs/uiis_scdi_replication.yaml").read_text(encoding="utf-8")); suim_payload = torch.load(ROOT / "outputs/dts_seg_gate0/formal/F4/checkpoints/final.pt", map_location=device, weights_only=False); uiis_payload = torch.load(ROOT / "outputs/uiis_scdi_replication/formal/F4/checkpoints/final.pt", map_location=device, weights_only=False); suim = build_suim_f4(suim_cfg, device); uiis = build_uiis_f4(uiis_cfg, "F4", device); suim.load_state_dict(suim_payload["model_state_dict"]); uiis.load_state_dict(uiis_payload["model_state_dict"]); suim.eval(); uiis.eval()
    conditions = load_conditions(ROOT / str(data["degradation_registry"])); output = ROOT / str(config["experiment"]["output_dir"]) / f"confidence_router_{args.split}"; output.mkdir(parents=True, exist_ok=True); rows = []
    for condition in conditions:
        row = evaluate_pair(suim, uiis, csv_path, condition, int(data.get("evaluation_image_size", 384)), device); rows.append(row); print(json.dumps(row, sort_keys=True), flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(output / "metrics.csv", index=False); summary = {"split": args.split, "rows": len(frame), "mean_miou": {key: float(frame[key].mean()) for key in ("SUIM_miou", "UIIS_miou", "confidence_miou")}, "mean_selected_suim_fraction": float(frame.selected_suim_fraction.mean())}; (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8"); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
