"""Build DARC-Seg per-image risk curves from frozen calibration/val logits."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metrics.uncertainty_ranking import uncertainty_scores  # noqa: E402
from reliability.selective_risk import coverage_grid, curve_summary  # noqa: E402
from scripts.analyze_boundary_residual import boundary_mask  # noqa: E402
from scripts.evaluate_temperature_scaling import validate_cache_payload  # noqa: E402


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def validate_protocol(config: dict[str, Any]) -> None:
    experiment, protocol = config["experiment"], config["protocol"]
    if experiment["fit_split"] != "calibration" or experiment["evaluation_split"] != "val" or not experiment["official_test_locked"]:
        raise ValueError("DARC risk curves permit calibration and validation caches only; TEST is locked.")
    if protocol["ground_truth_boundary_as_input"] or protocol["degradation_label_as_input"] or protocol["validation_used_for_fitting"] or protocol["official_test_evaluated"] or protocol["model_retrained"]:
        raise ValueError("DARC protocol flags must preserve the preregistered zero-training boundary.")


def _scores(logits: torch.Tensor, *, temperature: float, names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    all_scores = uncertainty_scores(logits, temperature=temperature)
    return {name: all_scores[name] for name in names}


def build_split(*, split: str, config: dict[str, Any], output: Path, batch_size: int) -> None:
    experiment, ranking, risk = config["experiment"], config["ranking"], config["risk"]
    conditions = tuple(experiment["conditions"])
    score_names = (ranking["primary_score"], *tuple(ranking["baselines"]))
    regions = ("full", "boundary", "interior")
    grid = coverage_grid(float(risk["coverage_grid_start"]), float(risk["coverage_grid_stop"]), float(risk["coverage_grid_step"]))
    checkpoint_hash = sha256(ROOT / experiment["checkpoint"])
    degradation_hash = sha256(ROOT / experiment["degradation_config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actual: np.ndarray | None = None
    envelopes: np.ndarray | None = None
    counts: np.ndarray | None = None
    sample_ids: list[str] | None = None
    for condition_index, condition in enumerate(conditions):
        payload = torch.load(ROOT / experiment["cache_dir"] / split / f"{condition}.pt", map_location="cpu", weights_only=False)
        validate_cache_payload(payload, split=split, condition=condition, checkpoint_sha256=checkpoint_hash, degradation_config_sha256=degradation_hash)
        if sample_ids is None:
            sample_ids = list(payload["sample_id"])
            shape = (len(score_names), len(regions), len(conditions), len(sample_ids), len(grid))
            actual, envelopes = np.empty(shape, dtype=np.float32), np.empty(shape, dtype=np.float32)
            counts = np.empty((len(regions), len(conditions), len(sample_ids)), dtype=np.int64)
        elif sample_ids != list(payload["sample_id"]):
            raise ValueError("All registered conditions must preserve the split CSV sample_id order.")
        for start in range(0, len(sample_ids), batch_size):
            logits = payload["logits"][start:start + batch_size].to(device, dtype=torch.float32, non_blocking=True)
            labels = payload["labels"][start:start + batch_size].to(device, dtype=torch.long, non_blocking=True)
            logits = functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            prediction = logits.argmax(dim=1)
            valid = labels.ge(0) & labels.lt(logits.shape[1])
            boundary = valid & boundary_mask(labels, 1)
            region_masks = (valid, boundary, valid & ~boundary)
            errors = prediction.ne(labels)
            score_values = _scores(logits, temperature=float(ranking["clean_temperature"]), names=score_names)
            for local in range(len(labels)):
                sample_index = start + local
                for region_index, mask in enumerate(region_masks):
                    chosen = mask[local]
                    if not bool(chosen.any()):
                        raise ValueError(f"{split}/{condition}/{sample_ids[sample_index]} has an empty {regions[region_index]} region.")
                    counts[region_index, condition_index, sample_index] = int(chosen.sum())
                    sample_errors = errors[local][chosen].detach().cpu().numpy()
                    for score_index, score_name in enumerate(score_names):
                        risks, envelope = curve_summary(score_values[score_name][local][chosen].detach().cpu().numpy(), sample_errors, grid)
                        actual[score_index, region_index, condition_index, sample_index] = risks
                        envelopes[score_index, region_index, condition_index, sample_index] = envelope
        print(f"built {split}/{condition} on {device.type}")
        del payload
        if device.type == "cuda":
            torch.cuda.empty_cache()
    assert actual is not None and envelopes is not None and counts is not None and sample_ids is not None
    _atomic_npz(output / "risk_curves" / f"{split}.npz", actual_risk=actual, monotone_envelope=envelopes, pixel_counts=counts, coverages=grid, score_names=np.asarray(score_names), region_names=np.asarray(regions), conditions=np.asarray(conditions), sample_ids=np.asarray(sample_ids))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen-logit DARC selective-risk curves; TEST is locked.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "darc_crc_pilot.yaml")
    parser.add_argument("--splits", nargs="+", default=["calibration", "val"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    validate_protocol(config)
    if set(args.splits) - {"calibration", "val"} or not args.splits:
        raise ValueError("Only calibration and val risk curves are allowed; TEST is locked.")
    output = ROOT / config["experiment"]["output_dir"]
    for split in args.splits:
        build_split(split=split, config=config, output=output, batch_size=args.batch_size)
    metadata = {"config": str(config_path.relative_to(ROOT)), "config_sha256": sha256(config_path), "checkpoint_sha256": sha256(ROOT / config["experiment"]["checkpoint"]), "degradation_config_sha256": sha256(ROOT / config["experiment"]["degradation_config"]), "splits": list(args.splits), "risk_unit": "per_image_then_sample_id_cluster", "ground_truth_boundary_as_input": False, "degradation_label_as_input": False, "validation_used_for_fitting": False, "official_test_evaluated": False, "model_retrained": False}
    (output / "risk_curves" / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
