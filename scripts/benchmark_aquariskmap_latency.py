"""Measure preregistered AquaRiskMap added inference latency on inner development.

This is an engineering measurement only: it permits the fixed
``risk_head_development`` split, never validation, calibration, or TEST.
Fifty images warm up CUDA; the following fifty provide per-image medians.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.suim_dataset import SUIMDataset, build_eval_transform  # noqa: E402
from reliability.aquariskmap import AquaRiskMap, build_features, full_resolution_risk  # noqa: E402
from scripts.cache_temperature_logits import build_frozen_model, model_logits  # noqa: E402
from scripts.evaluate_aquariskmap_stage_a import load_risk_head  # noqa: E402


ALLOWED_MODELS = ("segformer", "deeplab")
WARMUP_IMAGES = 50
MEASURE_IMAGES = 50


def load_config(root: Path = ROOT) -> dict[str, Any]:
    return yaml.safe_load((root / "configs" / "aquariskmap_pilot.yaml").read_text(encoding="utf-8"))


def atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def resolve_context(root: Path, config: dict[str, Any], *, model_name: str) -> tuple[Path, Path, Path, Path]:
    if model_name not in ALLOWED_MODELS:
        raise ValueError("Only preregistered SegFormer and DeepLab models are allowed.")
    split = (root / str(config["protocol"]["risk_head_split"]) / "risk_head_development.csv").resolve()
    if not split.is_file() or split.name != "risk_head_development.csv":
        raise ValueError("Latency benchmark must use fixed risk_head_development only.")
    names = set(pd.read_csv(split)["sample_id"].astype(str))
    split_root = root / "data" / "suim_processed" / "splits" / "v2_scene_grouped_deduplicated"
    barred = set().union(*(set(pd.read_csv(split_root / f"{name}.csv")["sample_id"].astype(str)) for name in ("val", "calibration", "test")))
    if names & barred:
        raise ValueError("Latency split overlaps a protected evaluation partition.")
    baseline_path = (root / str(config["base_models"][model_name]["baseline_config"])).resolve()
    checkpoint_path = (root / str(config["base_models"][model_name]["checkpoint"])).resolve()
    risk_path = (root / str(config["experiment"]["output_dir"]) / model_name / "checkpoints" / "last.pt").resolve()
    if not all(path.is_file() for path in (baseline_path, checkpoint_path, risk_path)):
        raise FileNotFoundError("Frozen segmentation or risk-head checkpoint is missing.")
    return split, baseline_path, checkpoint_path, risk_path


def synchronized_seconds(callable_: Any) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    callable_()
    torch.cuda.synchronize()
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=ALLOWED_MODELS)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for latency measurement.")
    config = load_config(ROOT)
    split, baseline_path, checkpoint_path, risk_path = resolve_context(ROOT, config, model_name=args.model)
    baseline = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    dataset = SUIMDataset(split, transform=build_eval_transform(int(baseline["data"]["image_size"])))
    if len(dataset) < WARMUP_IMAGES + MEASURE_IMAGES:
        raise ValueError("Development split is too small for the fixed latency protocol.")
    device = torch.device("cuda")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base = build_frozen_model(baseline["model"], int(baseline["data"]["num_classes"]), checkpoint, device).eval()
    risk = load_risk_head(risk_path, model_name=args.model, device=device).eval()
    def run_all(image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=bool(baseline["training"]["amp"])):
            logits = model_logits(base, image)
        with torch.no_grad():
            value = full_resolution_risk(risk(build_features(logits, image)))
        return value
    for index in range(WARMUP_IMAGES):
        image = dataset[index]["pixel_values"].unsqueeze(0).to(device, non_blocking=True)
        run_all(image)
    base_seconds, added_seconds = [], []
    for index in range(WARMUP_IMAGES, WARMUP_IMAGES + MEASURE_IMAGES):
        image = dataset[index]["pixel_values"].unsqueeze(0).to(device, non_blocking=True)
        holder: dict[str, torch.Tensor] = {}
        def base_only() -> None:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=bool(baseline["training"]["amp"])):
                holder["logits"] = model_logits(base, image)
        base_seconds.append(synchronized_seconds(base_only))
        logits = holder["logits"]
        def risk_only() -> None:
            with torch.no_grad():
                full_resolution_risk(risk(build_features(logits, image)))
        added_seconds.append(synchronized_seconds(risk_only))
    base_median = float(np.median(base_seconds))
    added_median = float(np.median(added_seconds))
    result = {
        "model": args.model, "warmup_images": WARMUP_IMAGES, "measured_images": MEASURE_IMAGES,
        "base_median_ms": base_median * 1000.0, "added_median_ms": added_median * 1000.0,
        "added_fraction_of_base": added_median / base_median, "latency_gate_pass": bool(added_median / base_median < 0.10),
        "risk_head_parameters": risk.parameter_count(), "parameter_gate_pass": bool(risk.parameter_count() < 200_000),
        "split": "risk_head_development", "validation_evaluated": False, "calibration_evaluated": False,
        "official_suim_test_evaluated": False,
    }
    output = ROOT / str(config["experiment"]["output_dir"]) / "stage_a" / args.model
    atomic_json(result, output / "latency.json")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
