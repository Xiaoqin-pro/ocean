"""Evaluate both frozen DTS Gate-0 checkpoints and apply the fixed decision."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import yaml
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.label_mapping import ID2LABEL, LABEL2ID  # noqa: E402
from datasets.suim_dataset import SUIMDataset, build_eval_transform  # noqa: E402
from degradations.registry import build_image_degradation, load_conditions  # noqa: E402
from scripts.train_dts_seg_gate0 import FORMAT, VARIANTS  # noqa: E402


def atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=path.parent, delete=False, encoding="utf-8") as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def mean_iou(confusion: np.ndarray) -> tuple[float, list[float | None]]:
    diagonal = np.diag(confusion)
    denominator = confusion.sum(0) + confusion.sum(1) - diagonal
    valid = denominator > 0
    per_class = np.divide(diagonal, denominator, out=np.full(8, np.nan), where=valid)
    return float(per_class[valid].mean()), [None if not np.isfinite(value) else float(value) for value in per_class]


def decision_from(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    required = {(variant, condition) for variant in VARIANTS for condition in frame.condition.unique()}
    if len(frame) != 26 or set(zip(frame.variant, frame.condition, strict=True)) != required:
        raise ValueError("Gate-0 requires 2 variants x 13 unique conditions.")
    indexed = frame.set_index(["variant", "condition"])
    f4 = frame[frame.variant == "F4"]
    dts = frame[frame.variant == "DTS"]
    mean_gain = float(100.0 * (dts.miou.mean() - f4.miou.mean()))
    severe_names = [name for name in frame.condition.unique() if name.endswith("_s3")]
    severe_gain = float(100.0 * (
        indexed.loc[[('DTS', name) for name in severe_names], "miou"].mean()
        - indexed.loc[[('F4', name) for name in severe_names], "miou"].mean()
    ))
    clean_change = float(100.0 * (indexed.loc[("DTS", "clean"), "miou"] - indexed.loc[("F4", "clean"), "miou"]))
    severe_changes = {
        name: float(100.0 * (indexed.loc[("DTS", name), "miou"] - indexed.loc[("F4", name), "miou"]))
        for name in severe_names
    }
    gate = config["gate"]
    magnitude = mean_gain >= float(gate["primary_min"]) or severe_gain >= float(gate["alternative_severe_mean_miou_pp_min"])
    clean_safe = clean_change >= -float(gate["clean_miou_decrease_pp_max"])
    nonnegative = sum(value >= 0.0 for value in severe_changes.values())
    passed = magnitude and clean_safe and nonnegative >= int(gate["severe_families_nonnegative_min"])
    return {
        "decision": "PASS" if passed else "FAIL",
        "dts_minus_f4_13_condition_mean_miou_pp": mean_gain,
        "dts_minus_f4_severity3_mean_miou_pp": severe_gain,
        "dts_minus_f4_clean_miou_pp": clean_change,
        "severity3_changes_pp": severe_changes,
        "severity3_nonnegative_families": nonnegative,
        "magnitude_gate": bool(magnitude),
        "clean_safety_gate": bool(clean_safe),
        "family_direction_gate": bool(nonnegative >= int(gate["severe_families_nonnegative_min"])),
        "formal_validation_evaluated": False,
        "calibration_evaluated": False,
        "official_suim_test_evaluated": False,
    }


def load_model(config: dict[str, Any], checkpoint: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    required = {"checkpoint_format": FORMAT, "epoch": int(config["training"]["epochs"]), "smoke": False}
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"Invalid {key} in {checkpoint}.")
    if any(bool(payload.get(key, True)) for key in ("validation_evaluated", "calibration_evaluated", "official_suim_test_evaluated")):
        raise ValueError("A candidate checkpoint records prohibited data access.")
    model = SegformerForSemanticSegmentation.from_pretrained(
        config["model"]["pretrained_model"], num_labels=8, id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model.eval()


def evaluate_condition(model: torch.nn.Module, csv_path: Path, condition: Any, image_size: int, device: torch.device) -> dict[str, Any]:
    dataset = SUIMDataset(csv_path, transform=build_eval_transform(image_size), image_degradation=build_image_degradation(condition))
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    confusion = torch.zeros(8, 8, dtype=torch.int64, device=device)
    with torch.no_grad():
        for batch in loader:
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(pixel_values=pixels).logits
            prediction = functional.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
            valid = labels.ne(255)
            indices = labels[valid] * 8 + prediction[valid]
            confusion += torch.bincount(indices, minlength=64).reshape(8, 8)
    matrix = confusion.cpu().numpy()
    miou, per_class = mean_iou(matrix)
    return {"condition": condition.name, "degradation_type": condition.degradation_type, "severity": int(condition.severity), "miou": miou, "per_class_iou": per_class, "pixels": int(matrix.sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dts_seg_gate0.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    development = ROOT / str(config["data"]["development_csv"])
    if development.name != "risk_head_development.csv" or len(pd.read_csv(development)) != 231:
        raise PermissionError("Gate-0 evaluation accepts only the frozen 231-image development role.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device("cuda")
    conditions = load_conditions(ROOT / str(config["data"]["degradation_registry"]))
    if len(conditions) != 13:
        raise ValueError("Expected the frozen 13-condition registry.")
    output = ROOT / str(config["experiment"]["output_dir"]) / "evaluation"
    rows = []
    for variant in VARIANTS:
        checkpoint = ROOT / str(config["experiment"]["output_dir"]) / "formal" / variant / "checkpoints" / "final.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Both frozen final checkpoints are required: {checkpoint}")
        model = load_model(config, checkpoint, device)
        for condition in conditions:
            path = output / "conditions" / f"{variant}_{condition.name}.json"
            if path.is_file():
                row = json.loads(path.read_text(encoding="utf-8"))
            else:
                row = {"variant": variant, **evaluate_condition(model, development, condition, int(config["data"]["image_size"]), device)}
                atomic_json(row, path)
            rows.append(row)
            print(f"{variant}/{condition.name}: mIoU={row['miou']:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "condition_metrics.csv", index=False)
    decision = decision_from(frame, config)
    atomic_json(decision, output / "gate0_decision.json")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
