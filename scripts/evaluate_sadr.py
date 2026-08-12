"""Evaluate SADR with a frozen UIIS-F4 segmentation expert."""
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
from reliability.sadr_seg import CompositionalBasisSADRFrontEnd, FrequencySADRFrontEnd, RoutedSADRFrontEnd, SADRFrontEnd  # noqa: E402
from scripts.evaluate_dts_seg_gate0 import evaluate_condition  # noqa: E402
from scripts.train_sadr import FORMAT, build_model  # noqa: E402


class SADRModel(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, front: torch.nn.Module) -> None:
        super().__init__(); self.base = base; self.front = front

    def forward(self, pixel_values: torch.Tensor):
        restored = self.front(pixel_values)[0]
        return self.base(pixel_values=restored)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/sadr_fast.yaml"); parser.add_argument("--split", choices=("confirmation", "suim_official"), default="confirmation"); parser.add_argument("--allow-confirmation", action="store_true"); args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8")); data = config["data"]
    if args.split == "confirmation":
        if not args.allow_confirmation: raise PermissionError("Confirmation is locked.")
        csv_path, expected = ROOT / str(data["confirmation_csv"]), 511
    else: csv_path, expected = ROOT / "data/suim_processed/splits/v2_scene_grouped_deduplicated/test.csv", 110
    if len(pd.read_csv(csv_path)) != expected: raise PermissionError("Unexpected frozen split size.")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required.")
    device = torch.device("cuda"); conditions = [item for item in load_conditions(ROOT / str(data["degradation_registry"])) if item.name in set(config["evaluation"]["conditions"])]
    checkpoint = ROOT / str(config["experiment"]["output_dir"]) / "formal/checkpoints/final.pt"; payload = torch.load(checkpoint, map_location=device, weights_only=False)
    for key, expected_value in {"checkpoint_format": FORMAT, "variant": "SADR", "epoch": int(config["training"]["epochs"]), "smoke": False}.items():
        if payload.get(key) != expected_value: raise ValueError(f"Invalid {key} in SADR checkpoint.")
    if any(bool(payload.get(key, True)) for key in ("calibration_evaluated", "confirmation_evaluated", "official_suim_test_evaluated")): raise ValueError("SADR checkpoint records prohibited evaluation access.")
    base, front = build_model(config, device); front.load_state_dict(payload["model_state_dict"]); front.eval(); baseline = base.eval(); sadr = SADRModel(base, front).eval(); rows = []; image_size = int(data["evaluation_image_size"])
    for variant, model in (("UIIS-F4", baseline), ("SADR", sadr)):
        for condition in conditions:
            metric = evaluate_condition(model, csv_path, condition, image_size, device); rows.append({"variant": variant, **metric}); print(f"{args.split}/{variant}/{condition.name}: mIoU={metric['miou']:.4f}", flush=True)
        torch.cuda.empty_cache()
    result = pd.DataFrame(rows); output = ROOT / str(config["experiment"]["output_dir"]) / f"{args.split}_evaluation"; output.mkdir(parents=True, exist_ok=True); result.to_csv(output / "condition_metrics.csv", index=False); baseline_mean = float(result[result.variant == "UIIS-F4"].miou.mean()); sadr_mean = float(result[result.variant == "SADR"].miou.mean()); summary = {"split": args.split, "confirmation_evaluated": args.split == "confirmation", "official_suim_test_evaluated": args.split == "suim_official", "rows": len(result), "images": expected, "mean_miou": {"UIIS-F4": baseline_mean, "SADR": sadr_mean}, "sadr_minus_uiis_pp": 100.0 * (sadr_mean - baseline_mean)}; (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8"); print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
