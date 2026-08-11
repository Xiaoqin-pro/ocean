"""Evaluate BPT on the frozen SUIM development role."""
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
from scripts.evaluate_dts_seg_gate0 import atomic_json, evaluate_condition  # noqa: E402
from scripts.train_dts_seg_gate0 import build_model as build_f4  # noqa: E402
from scripts.train_suim_bpt import FORMAT, build_model as build_bpt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/bpt_seg_suim.yaml"); args = parser.parse_args(); config = yaml.safe_load(args.config.read_text(encoding="utf-8")); development = ROOT / str(config["data"]["development_csv"])
    if development.name != "risk_head_development.csv" or len(pd.read_csv(development)) != 231: raise PermissionError("Only SUIM development is allowed.")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required.")
    device = torch.device("cuda"); conditions = load_conditions(ROOT / str(config["data"]["degradation_registry"])); output = ROOT / str(config["experiment"]["output_dir"]) / "development_evaluation"; output.mkdir(parents=True, exist_ok=True); rows = []
    f4_cfg = yaml.safe_load((ROOT / "configs/dts_seg_gate0.yaml").read_text(encoding="utf-8")); f4_payload = torch.load(ROOT / "outputs/dts_seg_gate0/formal/F4/checkpoints/final.pt", map_location=device, weights_only=False); f4 = build_f4(f4_cfg, device); f4.load_state_dict(f4_payload["model_state_dict"]); f4.eval()
    bpt_path = ROOT / str(config["experiment"]["output_dir"]) / "formal/BPT/checkpoints/final.pt"; bpt_payload = torch.load(bpt_path, map_location=device, weights_only=False); required = {"checkpoint_format": FORMAT, "variant": "BPT", "epoch": int(config["training"]["epochs"]), "smoke": False}
    for key, expected in required.items():
        if bpt_payload.get(key) != expected: raise ValueError(f"Invalid BPT checkpoint field {key}.")
    bpt = build_bpt(config, device); bpt.load_state_dict(bpt_payload["model_state_dict"]); bpt.eval()
    for variant, model in (("F4", f4), ("BPT", bpt)):
        for condition in conditions:
            row = {"variant": variant, **evaluate_condition(model, development, condition, int(config["data"]["image_size"]), device)}; rows.append(row); print(f"{variant}/{condition.name}: mIoU={row['miou']:.4f}", flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(output / "condition_metrics.csv", index=False); summary = {"split": "risk_head_development", "official_suim_test_evaluated": False, "rows": len(frame), "mean_miou": {variant: float(frame[frame.variant == variant].miou.mean()) for variant in frame.variant.unique()}}; atomic_json(summary, output / "summary.json"); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
