"""Evaluate UIIS UTD against F4 on calibration or locked confirmation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as functional
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.suim_dataset import SUIMDataset, build_eval_transform  # noqa: E402
from degradations.registry import build_image_degradation, load_conditions  # noqa: E402
from scripts.train_uiis_scdi_replication import build_models  # noqa: E402
from scripts.train_uiis_spt import build_model  # noqa: E402
from scripts.train_uiis_utd import FORMAT  # noqa: E402


def evaluate(model: torch.nn.Module, csv_path: Path, condition, image_size: int, device: torch.device) -> dict[str, object]:
    dataset = SUIMDataset(csv_path, transform=build_eval_transform(image_size), image_degradation=build_image_degradation(condition)); loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True); confusion = torch.zeros(8, 8, dtype=torch.int64, device=device)
    with torch.no_grad():
        for batch in loader:
            pixels = batch["pixel_values"].to(device); labels = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16): logits = model(pixels).logits
            prediction = functional.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False).argmax(1); valid = labels.ne(255); indices = labels[valid] * 8 + prediction[valid]; confusion += torch.bincount(indices, minlength=64).reshape(8, 8)
    matrix = confusion.cpu().numpy(); diagonal = matrix.diagonal(); denominator = matrix.sum(0) + matrix.sum(1) - diagonal; valid = denominator > 0; per_class = diagonal / denominator.clip(min=1)
    return {"condition": condition.name, "severity": int(condition.severity), "degradation_type": condition.degradation_type, "miou": float(per_class[valid].mean()), "per_class_iou": [float(value) if valid[index] else None for index, value in enumerate(per_class)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/uiis_utd.yaml"); parser.add_argument("--split", choices=("calibration", "confirmation"), default="calibration"); parser.add_argument("--allow-confirmation", action="store_true"); args = parser.parse_args(); config = yaml.safe_load(args.config.read_text(encoding="utf-8")); data = config["data"]
    if args.split == "confirmation" and not args.allow_confirmation: raise PermissionError("Confirmation is locked; pass --allow-confirmation after development freeze.")
    csv_path = ROOT / str(data[f"{args.split}_csv"]); expected = {"calibration": 508, "confirmation": 511}[args.split]
    if csv_path.name != f"{args.split}.csv" or len(pd.read_csv(csv_path)) != expected: raise PermissionError(f"Only frozen UIIS {args.split} is permitted.")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required.")
    device = torch.device("cuda"); conditions = [item for item in load_conditions(ROOT / str(data["degradation_registry"])) if item.name in set(config["evaluation"]["conditions"])]; output = ROOT / str(config["experiment"]["output_dir"]) / ("confirmation_evaluation" if args.split == "confirmation" else "calibration_evaluation"); output.mkdir(parents=True, exist_ok=True); rows = []
    f4_config = yaml.safe_load((ROOT / "configs/uiis_scdi_replication.yaml").read_text(encoding="utf-8")); f4_payload = torch.load(ROOT / "outputs/uiis_scdi_replication/formal/F4/checkpoints/final.pt", map_location=device, weights_only=False); f4 = build_models(f4_config, "F4", device); f4.load_state_dict(f4_payload["model_state_dict"]); f4.eval()
    utd_path = ROOT / str(config["experiment"]["output_dir"]) / "formal/checkpoints/final.pt"; utd_payload = torch.load(utd_path, map_location=device, weights_only=False); required = {"checkpoint_format": FORMAT, "variant": "UTD", "epoch": int(config["training"]["epochs"]), "smoke": False}
    for key, expected_value in required.items():
        if utd_payload.get(key) != expected_value: raise ValueError(f"Invalid UTD checkpoint field {key}.")
    utd = build_model(config, device); utd.load_state_dict(utd_payload["model_state_dict"]); utd.eval()
    for variant, model in (("F4", f4), ("UTD", utd)):
        for condition in conditions:
            row = {"variant": variant, **evaluate(model, csv_path, condition, int(data["image_size"]), device)}; rows.append(row); print(f"{args.split}/{variant}/{condition.name}: mIoU={row['miou']:.4f}", flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(output / "condition_metrics.csv", index=False); summary = {"split": args.split, "confirmation_evaluated": args.split == "confirmation", "official_suim_test_evaluated": False, "rows": len(frame), "mean_miou": {variant: float(frame[frame.variant == variant].miou.mean()) for variant in frame.variant.unique()}}; (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8"); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
