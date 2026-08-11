"""Evaluate frozen PESC against UIIS F4 and optionally SUIM F4."""
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
from scripts.train_dts_seg_gate0 import build_model as build_suim_f4  # noqa: E402
from scripts.train_pesc import build_pesc_model  # noqa: E402
from scripts.train_uiis_scdi_replication import build_models as build_uiis_f4  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/pesc_fast.yaml"); parser.add_argument("--split", choices=("confirmation", "suim_official"), default="confirmation"); parser.add_argument("--allow-confirmation", action="store_true"); args = parser.parse_args(); config = yaml.safe_load(args.config.read_text(encoding="utf-8")); data = config["data"]
    if args.split == "confirmation" and not args.allow_confirmation: raise PermissionError("Confirmation is locked.")
    if args.split == "confirmation": csv_path, expected = ROOT / str(data["uiis_confirmation_csv"]), 511
    else: csv_path, expected = ROOT / str(data.get("suim_test_csv", "data/suim_processed/splits/v2_scene_grouped_deduplicated/test.csv")), 110
    if len(pd.read_csv(csv_path)) != expected: raise PermissionError("Unexpected frozen split size.")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required.")
    device = torch.device("cuda"); conditions = [item for item in load_conditions(ROOT / str(data["degradation_registry"])) if item.name in set(config["evaluation"]["conditions"])]
    pesc_payload = torch.load(ROOT / str(config["experiment"]["output_dir"]) / "formal/checkpoints/final.pt", map_location=device, weights_only=False); pesc = build_pesc_model(config, device); pesc.load_state_dict(pesc_payload["model_state_dict"]); pesc.eval()
    if args.split == "confirmation":
        base_config = yaml.safe_load((ROOT / "configs/uiis_scdi_replication.yaml").read_text(encoding="utf-8")); base_payload = torch.load(ROOT / "outputs/uiis_scdi_replication/formal/F4/checkpoints/final.pt", map_location=device, weights_only=False); baseline = build_uiis_f4(base_config, "F4", device); baseline.load_state_dict(base_payload["model_state_dict"])
    else:
        base_config = yaml.safe_load((ROOT / "configs/dts_seg_gate0.yaml").read_text(encoding="utf-8")); base_payload = torch.load(ROOT / "outputs/dts_seg_gate0/formal/F4/checkpoints/final.pt", map_location=device, weights_only=False); baseline = build_suim_f4(base_config, device); baseline.load_state_dict(base_payload["model_state_dict"])
    baseline.eval(); rows = []; image_size = int(data["evaluation_image_size"])
    for variant, model in (("SourceF4", baseline), ("PESC", pesc)):
        for condition in conditions:
            metric = evaluate_condition(model, csv_path, condition, image_size, device); rows.append({"variant": variant, **metric}); print(f"{args.split}/{variant}/{condition.name}: mIoU={metric['miou']:.4f}", flush=True)
    frame = pd.DataFrame(rows); output = ROOT / str(config["experiment"]["output_dir"]) / f"{args.split}_evaluation"; output.mkdir(parents=True, exist_ok=True); frame.to_csv(output / "condition_metrics.csv", index=False); summary = {"split": args.split, "confirmation_evaluated": args.split == "confirmation", "official_suim_test_evaluated": args.split == "suim_official", "rows": len(frame), "images": expected, "mean_miou": {variant: float(frame[frame.variant == variant].miou.mean()) for variant in frame.variant.unique()}}; (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8"); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
