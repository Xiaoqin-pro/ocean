"""Evaluate a conventional LAB-CLAHE correction before UIIS-F4."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from degradations.registry import load_conditions  # noqa: E402
from scripts.evaluate_dts_seg_gate0 import evaluate_condition  # noqa: E402
from scripts.train_uiis_scdi_replication import build_models as build_uiis_f4  # noqa: E402

MEAN = torch.tensor((0.485, 0.456, 0.406))
STD = torch.tensor((0.229, 0.224, 0.225))


class CLAHEModel(torch.nn.Module):
    def __init__(self, base: torch.nn.Module) -> None:
        super().__init__(); self.base = base; self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def forward(self, pixel_values: torch.Tensor):
        mean = MEAN.to(pixel_values.device).view(1, 3, 1, 1); std = STD.to(pixel_values.device).view(1, 3, 1, 1)
        rgb = ((pixel_values * std + mean).clamp(0.0, 1.0) * 255.0).byte().permute(0, 2, 3, 1).cpu().numpy()
        enhanced = []
        for image in rgb:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB); lab[:, :, 0] = self.clahe.apply(lab[:, :, 0]); enhanced.append(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
        corrected = torch.from_numpy(np.stack(enhanced)).to(pixel_values.device, dtype=torch.float32) / 255.0
        corrected = (corrected.permute(0, 3, 1, 2) - mean) / std
        return self.base(pixel_values=corrected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--split", choices=("confirmation", "suim_official"), default="confirmation"); parser.add_argument("--allow-confirmation", action="store_true"); args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/uiis_scdi_replication.yaml").read_text(encoding="utf-8")); data = config["data"]
    if args.split == "confirmation":
        if not args.allow_confirmation: raise PermissionError("Confirmation is locked.")
        csv_path, expected = ROOT / str(data["confirmation_csv"]), 511
    else: csv_path, expected = ROOT / "data/suim_processed/splits/v2_scene_grouped_deduplicated/test.csv", 110
    if len(pd.read_csv(csv_path)) != expected: raise PermissionError("Unexpected frozen split.")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required.")
    device = torch.device("cuda"); base = build_uiis_f4(config, "F4", device).eval(); model = CLAHEModel(base).eval(); rows = []
    for condition in load_conditions(ROOT / "configs/degradation_pilot.yaml"):
        metric = evaluate_condition(model, csv_path, condition, 384, device); rows.append({"variant": "CLAHE+UIIS-F4", **metric}); print(f"{args.split}/CLAHE/{condition.name}: mIoU={metric['miou']:.4f}", flush=True)
    result = pd.DataFrame(rows); output = ROOT / "outputs/clahe_baseline" / f"{args.split}_evaluation"; output.mkdir(parents=True, exist_ok=True); result.to_csv(output / "condition_metrics.csv", index=False); summary = {"split": args.split, "confirmation_evaluated": args.split == "confirmation", "official_suim_test_evaluated": args.split == "suim_official", "rows": len(result), "images": expected, "mean_miou": float(result.miou.mean())}; (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8"); print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
