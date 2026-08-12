"""Evaluate the frozen SCDI Gate-2 checkpoint against F4."""
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
from scripts.train_scdi_seg_gate2 import FORMAT, build_model  # noqa: E402


def decision_from(frame, config, condition_names):
    renamed = frame.copy()
    renamed.loc[renamed.variant == "SCDI", "variant"] = "SDTC"
    value = generic_decision(renamed, config, condition_names)
    return {key.replace("sdtc", "scdi"): item for key, item in value.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/scdi_seg_gate2.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    development = ROOT / str(config["data"]["development_csv"])
    if development.name != "risk_head_development.csv" or len(pd.read_csv(development)) != 231:
        raise PermissionError("Gate-2 accepts only the frozen development role.")
    comparator = pd.read_csv(ROOT / str(config["gate"]["comparator"]))
    comparator = comparator[comparator.variant == str(config["gate"]["comparator_variant"])].copy()
    if len(comparator) != 13:
        raise ValueError("Frozen F4 comparator must contain 13 conditions.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device("cuda")
    conditions = load_conditions(ROOT / str(config["data"]["degradation_registry"]))
    names = [item.name for item in conditions]
    checkpoint = ROOT / str(config["experiment"]["output_dir"]) / "formal/SCDI/checkpoints/final.pt"
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    required = {"checkpoint_format": FORMAT, "variant": "SCDI", "epoch": int(config["training"]["epochs"]), "smoke": False}
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"Invalid {key} in SCDI checkpoint.")
    if any(bool(payload.get(key, True)) for key in ("validation_evaluated", "calibration_evaluated", "official_suim_test_evaluated")):
        raise ValueError("Candidate checkpoint records prohibited access.")
    model = build_model(config, device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    output = ROOT / str(config["experiment"]["output_dir"]) / "evaluation"
    rows = []
    for condition in conditions:
        path = output / "conditions" / f"SCDI_{condition.name}.json"
        if path.is_file():
            row = json.loads(path.read_text(encoding="utf-8"))
        else:
            row = {"variant": "SCDI", **evaluate_condition(model, development, condition, int(config["data"]["image_size"]), device)}
            atomic_json(row, path)
        rows.append(row)
        print(f"SCDI/{condition.name}: mIoU={row['miou']:.4f}", flush=True)
    frame = pd.concat([comparator, pd.DataFrame(rows)], ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "condition_metrics.csv", index=False)
    decision = decision_from(frame, config, names)
    atomic_json(decision, output / "gate2_decision.json")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
