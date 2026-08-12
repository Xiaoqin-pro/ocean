"""Evaluate frozen SGRE Gate-3 against F4."""
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
from scripts.evaluate_sdtc_seg_gate1 import decision_from as generic_decision  # noqa: E402
from scripts.train_scdi_seg_gate2 import build_model  # noqa: E402
from scripts.train_sgre_seg_gate3 import FORMAT  # noqa: E402


def decision_from(frame, config, names):
    renamed = frame.copy()
    renamed.loc[renamed.variant == "SGRE", "variant"] = "SDTC"
    result = generic_decision(renamed, config, names)
    return {key.replace("sdtc", "sgre"): value for key, value in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sgre_seg_gate3.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    development = ROOT / str(config["data"]["development_csv"])
    if development.name != "risk_head_development.csv" or len(pd.read_csv(development)) != 231:
        raise PermissionError("Gate-3 accepts only the frozen development role.")
    comparator = pd.read_csv(ROOT / str(config["gate"]["comparator"]))
    comparator = comparator[comparator.variant == str(config["gate"]["comparator_variant"])].copy()
    if len(comparator) != 13 or not torch.cuda.is_available():
        raise RuntimeError("Frozen comparator and CUDA are required.")
    device = torch.device("cuda")
    conditions = load_conditions(ROOT / str(config["data"]["degradation_registry"]))
    names = [condition.name for condition in conditions]
    checkpoint = ROOT / str(config["experiment"]["output_dir"]) / "formal/SGRE/checkpoints/final.pt"
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    for key, expected in {"checkpoint_format": FORMAT, "variant": "SGRE", "epoch": 8, "smoke": False}.items():
        if payload.get(key) != expected:
            raise ValueError(f"Invalid {key} in SGRE checkpoint.")
    if any(bool(payload.get(key, True)) for key in ("validation_evaluated", "calibration_evaluated", "official_suim_test_evaluated")):
        raise ValueError("Candidate records prohibited data access.")
    model = build_model(config, device)
    model.load_state_dict(payload["model_state_dict"]); model.eval()
    output = ROOT / str(config["experiment"]["output_dir"]) / "evaluation"
    rows = []
    for condition in conditions:
        path = output / "conditions" / f"SGRE_{condition.name}.json"
        if path.is_file():
            row = json.loads(path.read_text(encoding="utf-8"))
        else:
            row = {"variant": "SGRE", **evaluate_condition(model, development, condition, int(config["data"]["image_size"]), device)}
            atomic_json(row, path)
        rows.append(row); print(f"SGRE/{condition.name}: mIoU={row['miou']:.4f}", flush=True)
    frame = pd.concat([comparator, pd.DataFrame(rows)], ignore_index=True)
    output.mkdir(parents=True, exist_ok=True); frame.to_csv(output / "condition_metrics.csv", index=False)
    decision = decision_from(frame, config, names); atomic_json(decision, output / "gate3_decision.json")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
