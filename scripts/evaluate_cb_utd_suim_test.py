"""Final SUIM official-test evaluation for frozen CB-UTD."""
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
from scripts.train_suim_spt import build_model as build_cb_utd  # noqa: E402
from scripts.train_suim_utd import FORMAT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/cb_utd_suim_test.yaml"); args = parser.parse_args(); config = yaml.safe_load(args.config.read_text(encoding="utf-8")); data = config["data"]; test_csv = ROOT / str(data["test_csv"])
    if test_csv.name != "test.csv" or len(pd.read_csv(test_csv)) != 110: raise PermissionError("Only the frozen 110-scene SUIM official test role is permitted.")
    if not config["evaluation"].get("official_suim_test"): raise PermissionError("Official test flag must be explicit.")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required.")
    device = torch.device("cuda"); conditions = [item for item in load_conditions(ROOT / str(data["degradation_registry"])) if item.name in set(config["evaluation"]["conditions"])]
    output = ROOT / str(data.get("output_dir", config["experiment"]["output_dir"])) / "official_test_evaluation"; output.mkdir(parents=True, exist_ok=True); rows = []
    f4_cfg = yaml.safe_load((ROOT / "configs/dts_seg_gate0.yaml").read_text(encoding="utf-8")); f4_payload = torch.load(ROOT / str(config["model"]["comparator_checkpoint"]), map_location=device, weights_only=False); f4 = build_f4(f4_cfg, device); f4.load_state_dict(f4_payload["model_state_dict"]); f4.eval()
    cb_path = ROOT / str(config["model"]["checkpoint"]); cb_payload = torch.load(cb_path, map_location=device, weights_only=False)
    if cb_payload.get("checkpoint_format") != FORMAT or cb_payload.get("variant") != "UTD" or cb_payload.get("smoke"):
        raise ValueError("Invalid frozen CB-UTD checkpoint.")
    cb = build_cb_utd(config, device); cb.load_state_dict(cb_payload["model_state_dict"]); cb.eval()
    for variant, model in (("F4", f4), ("CB-UTD", cb)):
        for condition in conditions:
            row = {"variant": variant, **evaluate_condition(model, test_csv, condition, int(data["image_size"]), device)}; rows.append(row); print(f"official_test/{variant}/{condition.name}: mIoU={row['miou']:.4f}", flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(output / "condition_metrics.csv", index=False); summary = {"split": "suim_official_test", "official_suim_test_evaluated": True, "confirmation_evaluated": False, "rows": len(frame), "test_images": 110, "mean_miou": {variant: float(frame[frame.variant == variant].miou.mean()) for variant in frame.variant.unique()}}
    atomic_json(summary, output / "summary.json"); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
