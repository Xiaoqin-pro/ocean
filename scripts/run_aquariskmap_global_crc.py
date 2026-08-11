"""Fixed Global-CRC stage for AquaRiskMap.

The command deliberately exposes only a preregistered model and phase.
``calibration`` builds curves and freezes coverage parameters; ``validation``
is refused until those parameters exist, then evaluates the frozen choices once.
Official SUIM TEST is never an accepted path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.suim_dataset import SUIMDataset, build_eval_transform  # noqa: E402
from degradations.registry import build_image_degradation, load_conditions  # noqa: E402
from metrics.uncertainty_ranking import SCORE_NAMES, uncertainty_scores  # noqa: E402
from reliability.aquariskmap import CONDITIONS, IGNORE_INDEX, AquaRiskMap, build_features, full_resolution_risk  # noqa: E402
from reliability.conformal_risk import select_crc_coverage  # noqa: E402
from reliability.selective_risk import coverage_grid, curve_summary  # noqa: E402
from scripts.cache_temperature_logits import sha256  # noqa: E402
from scripts.evaluate_aquariskmap_stage_a import load_risk_head  # noqa: E402
from scripts.evaluate_temperature_scaling import validate_cache_payload  # noqa: E402


ALLOWED_MODELS = ("segformer", "deeplab")
SCORES = (*SCORE_NAMES, "aquariskmap")
ALPHA = 0.10


def load_config(root: Path = ROOT) -> dict[str, Any]:
    return yaml.safe_load((root / "configs" / "aquariskmap_pilot.yaml").read_text(encoding="utf-8"))


def atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        table.to_csv(handle, index=False)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def context(root: Path, config: Mapping[str, Any], *, model_name: str, split: str) -> tuple[Path, Path, Path, Path, Path, Path]:
    if model_name not in ALLOWED_MODELS or split not in {"calibration", "val"}:
        raise ValueError("Only preregistered model names and calibration/val phases are permitted.")
    if not config["protocol"]["official_suim_test_locked"] or config["protocol"]["official_suim_test_evaluated"]:
        raise ValueError("Official SUIM TEST must remain locked.")
    split_csv = (root / str(config["protocol"][f"{split if split == 'calibration' else 'validation'}_split"])).resolve()
    if split_csv.name != f"{split}.csv" or not split_csv.is_file():
        raise ValueError("CRC may read only the fixed calibration or validation CSV.")
    temperature_config = root / ("configs/temperature_scaling.yaml" if model_name == "segformer" else "configs/deeplabv3_temperature_scaling.yaml")
    temperature_cfg = yaml.safe_load(temperature_config.read_text(encoding="utf-8"))
    cache_root = (root / str(temperature_cfg["experiment"]["output_dir"]) / "cache" / split).resolve()
    risk_checkpoint = (root / str(config["experiment"]["output_dir"]) / model_name / "checkpoints" / "last.pt").resolve()
    degradation = (root / str(config["degradations"]["config"])).resolve()
    base_checkpoint = (root / str(config["base_models"][model_name]["checkpoint"])).resolve()
    temperature = (root / str(temperature_cfg["experiment"]["output_dir"]) / "temperatures.json").resolve()
    baseline = (root / str(config["base_models"][model_name]["baseline_config"])).resolve()
    if not all(path.is_file() for path in (split_csv, risk_checkpoint, degradation, base_checkpoint, temperature, baseline)) or not cache_root.is_dir():
        raise FileNotFoundError("A frozen CRC input is missing.")
    return split_csv, cache_root, risk_checkpoint, degradation, base_checkpoint, temperature


def score_values(logits: torch.Tensor, pixels: torch.Tensor, labels: torch.Tensor, risk_head: AquaRiskMap, *, temperature: float) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    full_logits = functional.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
    risk = full_resolution_risk(risk_head(build_features(logits, pixels)), size=tuple(labels.shape[-2:]))[:, 0].sigmoid()
    valid = labels.ne(IGNORE_INDEX) & labels.ge(0) & labels.lt(full_logits.shape[1])
    return {**uncertainty_scores(full_logits, temperature=temperature), "aquariskmap": risk}, full_logits.argmax(dim=1).ne(labels) & valid


def build_curves(root: Path, config: Mapping[str, Any], *, model_name: str, split: str) -> Path:
    split_csv, cache_root, risk_checkpoint, degradation_path, base_checkpoint, temperature_path = context(root, config, model_name=model_name, split=split)
    conditions = load_conditions(degradation_path)
    if tuple(item.name for item in conditions) != CONDITIONS:
        raise ValueError("CRC requires the frozen 13-condition registry.")
    baseline = yaml.safe_load((root / str(config["base_models"][model_name]["baseline_config"])).read_text(encoding="utf-8"))
    ids = pd.read_csv(split_csv)["sample_id"].astype(str).tolist()
    expected_count = 146
    if len(ids) != expected_count or len(set(ids)) != expected_count:
        raise ValueError("CRC expects the fixed 146-image SUIM calibration or validation split.")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen AquaRiskMap CRC inference.")
    risk_head = load_risk_head(risk_checkpoint, model_name=model_name, device=device)
    temperature = float(json.loads(temperature_path.read_text(encoding="utf-8"))["clean_global"])
    grid = coverage_grid(0.01, 1.0, 0.01)
    actual = np.empty((len(SCORES), len(CONDITIONS), len(ids), len(grid)), dtype=np.float32)
    envelope = np.empty_like(actual)
    base_hash, degradation_hash = sha256(base_checkpoint), sha256(degradation_path)
    for condition_index, condition in enumerate(conditions):
        payload = torch.load(cache_root / f"{condition.name}.pt", map_location="cpu", weights_only=False)
        validate_cache_payload(payload, split=split, condition=condition.name, checkpoint_sha256=base_hash, degradation_config_sha256=degradation_hash)
        if [str(value) for value in payload["sample_id"]] != ids:
            raise ValueError("Frozen-logit cache IDs differ from the fixed CRC split.")
        dataset = SUIMDataset(split_csv, transform=build_eval_transform(int(baseline["data"]["image_size"])), image_degradation=build_image_degradation(condition))
        for start in range(0, len(ids), 4):
            samples = [dataset[index] for index in range(start, min(start + 4, len(ids)))]
            if [item["sample_id"] for item in samples] != ids[start:start + len(samples)]:
                raise ValueError("Dataset order differs from frozen CRC cache order.")
            labels_cpu = torch.stack([item["labels"] for item in samples])
            if not torch.equal(labels_cpu, payload["labels"][start:start + len(samples)]):
                raise ValueError("Dataset labels differ from frozen CRC cache labels.")
            pixels = torch.stack([item["pixel_values"] for item in samples]).to(device, non_blocking=True)
            labels = labels_cpu.to(device, non_blocking=True)
            logits = payload["logits"][start:start + len(samples)].to(device, dtype=torch.float32, non_blocking=True)
            with torch.no_grad():
                values, errors = score_values(logits, pixels, labels, risk_head, temperature=temperature)
            valid = labels.ne(IGNORE_INDEX)
            for local in range(len(samples)):
                chosen = valid[local]
                for score_index, score_name in enumerate(SCORES):
                    curve, upper = curve_summary(values[score_name][local][chosen].detach().cpu().numpy(), errors[local][chosen].detach().cpu().numpy(), grid)
                    actual[score_index, condition_index, start + local] = curve
                    envelope[score_index, condition_index, start + local] = upper
        print(f"built AquaRiskMap CRC {model_name}/{split}/{condition.name}", flush=True)
    output = root / str(config["experiment"]["output_dir"]) / "stage_a" / model_name / "crc"
    destination = output / f"{split}_curves.npz"
    atomic_npz(destination, actual_risk=actual, monotone_envelope=envelope, coverages=grid, scores=np.asarray(SCORES), conditions=np.asarray(CONDITIONS), sample_ids=np.asarray(ids))
    atomic_json({"model": model_name, "split": split, "samples": len(ids), "conditions": list(CONDITIONS), "scores": list(SCORES), "coverage_grid": grid.tolist(), "base_checkpoint_sha256": base_hash, "risk_checkpoint_sha256": sha256(risk_checkpoint), "degradation_config_sha256": degradation_hash, "calibration_evaluated": split == "calibration", "validation_evaluated": split == "val", "official_suim_test_evaluated": False}, output / f"{split}_curves_metadata.json")
    return destination


def fit_parameters(root: Path, config: Mapping[str, Any], *, model_name: str) -> Path:
    output = root / str(config["experiment"]["output_dir"]) / "stage_a" / model_name / "crc"
    with np.load(output / "calibration_curves.npz", allow_pickle=False) as curves:
        grid, scores = curves["coverages"].astype(np.float64), curves["scores"].astype(str).tolist()
        envelope = curves["monotone_envelope"].astype(np.float64)
    if scores != list(SCORES) or envelope.shape[:3] != (len(SCORES), len(CONDITIONS), 146):
        raise ValueError("Calibration curves do not satisfy the fixed AquaRiskMap CRC schema.")
    selections = {}
    for index, score in enumerate(scores):
        losses = envelope[index].mean(axis=0)  # one independent 13-condition curve per original image
        selections[score] = select_crc_coverage(losses, grid, ALPHA, bound=1.0).__dict__
    path = output / "calibration_parameters.json"
    atomic_json({"model": model_name, "alpha": ALPHA, "cluster_unit": "sample_id_with_all_13_conditions", "monotone_envelope": True, "global_crc_only": True, "selections": selections, "official_suim_test_evaluated": False}, path)
    return path


def evaluate_validation(root: Path, config: Mapping[str, Any], *, model_name: str) -> Path:
    output = root / str(config["experiment"]["output_dir"]) / "stage_a" / model_name / "crc"
    parameters_path = output / "calibration_parameters.json"
    if not parameters_path.is_file():
        raise ValueError("Validation CRC is locked until calibration parameters are frozen.")
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    with np.load(output / "val_curves.npz", allow_pickle=False) as curves:
        actual, grid = curves["actual_risk"].astype(np.float64), curves["coverages"].astype(np.float64)
        scores, conditions, sample_ids = curves["scores"].astype(str).tolist(), curves["conditions"].astype(str).tolist(), curves["sample_ids"].astype(str).tolist()
    rows = []
    for score_index, score in enumerate(scores):
        selection = parameters["selections"][score]
        index = int(selection["index"])
        selected = actual[score_index, :, :, index]
        rows.append({"model": model_name, "score": score, "coverage": float(grid[index]), "selective_risk": float(selected.mean()), "risk_excess": float(selected.mean() - ALPHA), "calibration_coverage": float(selection["coverage"]), "calibration_corrected_risk": float(selection["corrected_risk"]), "clusters": len(sample_ids), "conditions": len(conditions)})
    table = pd.DataFrame(rows)
    indexed = table.set_index("score")
    raw, aqua = indexed.loc["raw_msp"], indexed.loc["aquariskmap"]
    decision = {"coverage_improvement": float(aqua.coverage - raw.coverage), "selective_risk_difference": float(aqua.selective_risk - raw.selective_risk), "crc_gate_pass": bool(aqua.coverage - raw.coverage >= 0.03 and aqua.selective_risk <= raw.selective_risk), "official_suim_test_evaluated": False}
    atomic_csv(table, output / "validation_global_crc_metrics.csv")
    atomic_json(decision, output / "validation_global_crc_decision.json")
    return output / "validation_global_crc_metrics.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=ALLOWED_MODELS)
    parser.add_argument("--phase", required=True, choices=("calibration", "fit", "validation"))
    args = parser.parse_args()
    config = load_config(ROOT)
    if args.phase == "calibration":
        build_curves(ROOT, config, model_name=args.model, split="calibration")
    elif args.phase == "fit":
        fit_parameters(ROOT, config, model_name=args.model)
    else:
        output = ROOT / str(config["experiment"]["output_dir"]) / "stage_a" / args.model / "crc"
        if not (output / "calibration_parameters.json").is_file():
            raise ValueError("Cannot read validation before calibration CRC parameters are frozen.")
        # A prior interruption after the atomically written val curves must not
        # trigger a second validation-data read.  The curve builder itself only
        # writes its final NPZ after all 13 conditions have passed validation.
        if not (output / "val_curves.npz").is_file():
            build_curves(ROOT, config, model_name=args.model, split="val")
        evaluate_validation(ROOT, config, model_name=args.model)


if __name__ == "__main__":
    main()
