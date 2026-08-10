"""Evaluate the frozen SDTC Gate-1 checkpoint against the cached F4 comparator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from degradations.registry import load_conditions  # noqa: E402
from scripts.evaluate_dts_seg_gate0 import atomic_json, evaluate_condition  # noqa: E402
from scripts.train_sdtc_seg_gate1 import FORMAT, build_model  # noqa: E402


def decision_from(frame: pd.DataFrame, config: dict[str, Any], condition_names: list[str]) -> dict[str, Any]:
    required = {(variant, condition) for variant in ("F4", "SDTC") for condition in condition_names}
    observed = set(zip(frame.variant, frame.condition, strict=True))
    if len(frame) != 26 or observed != required:
        raise ValueError("Gate-1 requires F4 and SDTC over the frozen 13 conditions.")
    indexed = frame.set_index(["variant", "condition"])
    f4 = frame[frame.variant == "F4"]
    sdtc = frame[frame.variant == "SDTC"]
    mean_gain = float(100.0 * (sdtc.miou.mean() - f4.miou.mean()))
    severe_names = [name for name in condition_names if name.endswith("_s3")]
    severe_gain = float(100.0 * (
        indexed.loc[[('SDTC', name) for name in severe_names], "miou"].mean()
        - indexed.loc[[('F4', name) for name in severe_names], "miou"].mean()
    ))
    clean_change = float(100.0 * (indexed.loc[("SDTC", "clean"), "miou"] - indexed.loc[("F4", "clean"), "miou"]))
    severe_changes = {
        name: float(100.0 * (indexed.loc[("SDTC", name), "miou"] - indexed.loc[("F4", name), "miou"]))
        for name in severe_names
    }
    gate = config["gate"]
    magnitude = mean_gain >= float(gate["primary_min_miou_pp"]) or severe_gain >= float(gate["alternative_severe_mean_miou_pp_min"])
    clean_safe = clean_change >= -float(gate["clean_miou_decrease_pp_max"])
    nonnegative = sum(value >= 0.0 for value in severe_changes.values())
    family_direction = nonnegative >= int(gate["severe_families_nonnegative_min"])
    return {
        "decision": "PASS" if magnitude and clean_safe and family_direction else "FAIL",
        "sdtc_minus_f4_13_condition_mean_miou_pp": mean_gain,
        "sdtc_minus_f4_severity3_mean_miou_pp": severe_gain,
        "sdtc_minus_f4_clean_miou_pp": clean_change,
        "severity3_changes_pp": severe_changes,
        "severity3_nonnegative_families": int(nonnegative),
        "magnitude_gate": bool(magnitude),
        "clean_safety_gate": bool(clean_safe),
        "family_direction_gate": bool(family_direction),
        "formal_validation_evaluated": False,
        "calibration_evaluated": False,
        "official_suim_test_evaluated": False,
    }


def load_candidate(config: dict[str, Any], checkpoint: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    required = {"checkpoint_format": FORMAT, "variant": "SDTC", "epoch": int(config["training"]["epochs"]), "smoke": False}
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"Invalid {key} in {checkpoint}.")
    if any(bool(payload.get(key, True)) for key in ("validation_evaluated", "calibration_evaluated", "official_suim_test_evaluated")):
        raise ValueError("Candidate checkpoint records prohibited data access.")
    model = build_model(config, device)
    model.load_state_dict(payload["model_state_dict"])
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sdtc_seg_gate1.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    development = ROOT / str(config["data"]["development_csv"])
    if development.name != "risk_head_development.csv" or len(pd.read_csv(development)) != 231:
        raise PermissionError("Gate-1 evaluation accepts only the frozen 231-image development role.")
    comparator_path = ROOT / str(config["gate"]["comparator"])
    comparator = pd.read_csv(comparator_path)
    comparator = comparator[comparator.variant == str(config["gate"]["comparator_variant"])].copy()
    if len(comparator) != 13:
        raise ValueError("Frozen F4 comparator must contain 13 conditions.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device("cuda")
    conditions = load_conditions(ROOT / str(config["data"]["degradation_registry"]))
    condition_names = [item.name for item in conditions]
    if len(condition_names) != 13:
        raise ValueError("Expected the frozen 13-condition registry.")
    checkpoint = ROOT / str(config["experiment"]["output_dir"]) / "formal" / "SDTC" / "checkpoints" / "final.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Frozen SDTC final checkpoint is required: {checkpoint}")
    model = load_candidate(config, checkpoint, device)
    output = ROOT / str(config["experiment"]["output_dir"]) / "evaluation"
    rows = []
    for condition in conditions:
        path = output / "conditions" / f"SDTC_{condition.name}.json"
        if path.is_file():
            row = json.loads(path.read_text(encoding="utf-8"))
        else:
            row = {"variant": "SDTC", **evaluate_condition(model, development, condition, int(config["data"]["image_size"]), device)}
            atomic_json(row, path)
        rows.append(row)
        print(f"SDTC/{condition.name}: mIoU={row['miou']:.4f}", flush=True)
    frame = pd.concat([comparator, pd.DataFrame(rows)], ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "condition_metrics.csv", index=False)
    decision = decision_from(frame, config, condition_names)
    atomic_json(decision, output / "gate1_decision.json")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
