"""Evaluate COSA-Seg on frozen UIIS confirmation and optional SUIM official test."""
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
from scripts.train_cosa import FORMAT, build_model  # noqa: E402
from scripts.train_uiis_scdi_replication import build_models as build_uiis_f4  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cosa_fast.yaml")
    parser.add_argument("--split", choices=("confirmation", "suim_official"), default="confirmation")
    parser.add_argument("--allow-confirmation", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8")); data = config["data"]
    if args.split == "confirmation":
        if not args.allow_confirmation: raise PermissionError("Confirmation is locked.")
        csv_path, expected = ROOT / str(data["uiis_confirmation_csv"]), 511
    else:
        csv_path, expected = ROOT / "data/suim_processed/splits/v2_scene_grouped_deduplicated/test.csv", 110
    if len(pd.read_csv(csv_path)) != expected: raise PermissionError("Unexpected frozen split size.")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required.")
    device = torch.device("cuda"); conditions = [item for item in load_conditions(ROOT / str(data["degradation_registry"])) if item.name in set(config["evaluation"]["conditions"])]
    checkpoint = ROOT / str(config["experiment"]["output_dir"]) / "formal/checkpoints/final.pt"; payload = torch.load(checkpoint, map_location=device, weights_only=False)
    for key, expected_value in {"checkpoint_format": FORMAT, "variant": "COSA-Seg", "epoch": int(config["training"]["epochs"]), "smoke": False}.items():
        if payload.get(key) != expected_value: raise ValueError(f"Invalid {key} in COSA checkpoint.")
    if any(bool(payload.get(key, True)) for key in ("calibration_evaluated", "confirmation_evaluated", "official_suim_test_evaluated")): raise ValueError("COSA checkpoint records prohibited evaluation access.")
    cosa = build_model(config, device); cosa.load_state_dict(payload["model_state_dict"]); cosa.eval()
    uiis_config = yaml.safe_load((ROOT / str(data["uiis_config"])).read_text(encoding="utf-8")); baseline = build_uiis_f4(uiis_config, "F4", device); baseline.eval()
    rows = []; image_size = int(data["evaluation_image_size"])
    for variant, model in (("UIIS-F4", baseline), ("COSA-Seg", cosa)):
        for condition in conditions:
            metric = evaluate_condition(model, csv_path, condition, image_size, device); rows.append({"variant": variant, **metric}); print(f"{args.split}/{variant}/{condition.name}: mIoU={metric['miou']:.4f}", flush=True)
        torch.cuda.empty_cache()
    result = pd.DataFrame(rows); output = ROOT / str(config["experiment"]["output_dir"]) / f"{args.split}_evaluation"; output.mkdir(parents=True, exist_ok=True); result.to_csv(output / "condition_metrics.csv", index=False)
    baseline_mean = float(result[result.variant == "UIIS-F4"].miou.mean()); cosa_mean = float(result[result.variant == "COSA-Seg"].miou.mean()); summary = {"split": args.split, "confirmation_evaluated": args.split == "confirmation", "official_suim_test_evaluated": args.split == "suim_official", "rows": len(result), "images": expected, "mean_miou": {"UIIS-F4": baseline_mean, "COSA-Seg": cosa_mean}, "cosa_minus_uiis_pp": 100.0 * (cosa_mean - baseline_mean)}; (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8"); print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
