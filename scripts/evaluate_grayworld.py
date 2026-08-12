"""Evaluate a conventional gray-world correction before UIIS-F4."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


class GrayWorldModel(torch.nn.Module):
    def __init__(self, base: torch.nn.Module) -> None:
        super().__init__(); self.base = base

    def forward(self, pixel_values: torch.Tensor):
        mean = MEAN.to(pixel_values.device).view(1, 3, 1, 1); std = STD.to(pixel_values.device).view(1, 3, 1, 1)
        rgb = (pixel_values * std + mean).clamp(0.0, 1.0); channel_mean = rgb.mean(dim=(-2, -1), keepdim=True); gray = channel_mean.mean(1, keepdim=True); gain = (gray / channel_mean.clamp_min(1e-3)).clamp(0.5, 2.0); corrected = (rgb * gain).clamp(0.0, 1.0); corrected = (corrected - mean) / std
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
    device = torch.device("cuda"); base = build_uiis_f4(config, "F4", device).eval(); gray = GrayWorldModel(base).eval(); conditions = load_conditions(ROOT / "configs/degradation_pilot.yaml"); rows = []
    for condition in conditions:
        metric = evaluate_condition(gray, csv_path, condition, 384, device); rows.append({"variant": "GrayWorld+UIIS-F4", **metric}); print(f"{args.split}/GrayWorld/{condition.name}: mIoU={metric['miou']:.4f}", flush=True)
    result = pd.DataFrame(rows); output = ROOT / "outputs/grayworld_baseline" / f"{args.split}_evaluation"; output.mkdir(parents=True, exist_ok=True); result.to_csv(output / "condition_metrics.csv", index=False); summary = {"split": args.split, "confirmation_evaluated": args.split == "confirmation", "official_suim_test_evaluated": args.split == "suim_official", "rows": len(result), "images": expected, "mean_miou": float(result.miou.mean())}; (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8"); print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
