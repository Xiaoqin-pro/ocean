"""Build fixed per-image raw-MSP selective-risk curves from UIIS DeepLab caches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
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
from scripts.evaluate_temperature_scaling import CONDITIONS, validate_cache_payload  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".npz", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def build_split(config: dict[str, Any], split: str) -> None:
    experiment, risk = config["experiment"], config["risk"]
    if split not in ("calibration", "confirmation"):
        raise ValueError("Only calibration and confirmation CRC curves are permitted; TEST is locked.")
    expected = int(experiment["samples"][split])
    output = ROOT / experiment["output_dir"] / "risk_curves"
    destination = output / f"{split}.npz"
    if destination.is_file():
        with np.load(destination, allow_pickle=False) as stored:
            if stored["actual_risk"].shape[:2] == (len(CONDITIONS), expected):
                print(f"reused complete {split} curves", flush=True)
                return
    cache_root = ROOT / experiment["cache_dir"] / split
    checkpoint_sha256 = sha256(ROOT / experiment["checkpoint"])
    degradation_sha256 = sha256(ROOT / experiment["degradation_config"])
    grid = coverage_grid(float(risk["coverage_grid_start"]), float(risk["coverage_grid_stop"]), float(risk["coverage_grid_step"]))
    size = int(risk["measurement_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actual = np.empty((len(CONDITIONS), expected, len(grid)), dtype=np.float32)
    envelope = np.empty_like(actual)
    counts = np.empty((len(CONDITIONS), expected), dtype=np.int64)
    sample_ids: list[str] | None = None
    for condition_index, condition in enumerate(CONDITIONS):
        payload = torch.load(cache_root / f"{condition}.pt", map_location="cpu", weights_only=False)
        validate_cache_payload(payload, split=split, condition=condition, checkpoint_sha256=checkpoint_sha256, degradation_config_sha256=degradation_sha256)
        current_ids = [str(value) for value in payload["sample_id"]]
        if len(current_ids) != expected:
            raise ValueError(f"Unexpected {split} sample count for {condition}: {len(current_ids)} != {expected}.")
        if sample_ids is None:
            sample_ids = current_ids
        elif sample_ids != current_ids:
            raise ValueError(f"{split} cache sample order differs across conditions.")
        for start in range(0, expected, 8):
            logits = payload["logits"][start:start + 8].to(device, dtype=torch.float32, non_blocking=True)
            labels = payload["labels"][start:start + 8].to(device, dtype=torch.float32, non_blocking=True).unsqueeze(1)
            logits = functional.interpolate(logits, size=(size, size), mode="bilinear", align_corners=False)
            labels = functional.interpolate(labels, size=(size, size), mode="nearest").squeeze(1).long()
            scores = uncertainty_scores(logits, temperature=1.0)["raw_msp"]
            errors = logits.argmax(dim=1).ne(labels)
            for local in range(len(labels)):
                index = start + local
                actual[condition_index, index], envelope[condition_index, index] = curve_summary(
                    scores[local].reshape(-1).detach().cpu().numpy(), errors[local].reshape(-1).detach().cpu().numpy(), grid,
                )
                counts[condition_index, index] = int(errors[local].numel())
        print(f"built {split}/{condition}", flush=True)
        del payload
        if device.type == "cuda":
            torch.cuda.empty_cache()
    assert sample_ids is not None
    atomic_npz(destination, actual_risk=actual, monotone_envelope=envelope, pixel_counts=counts, coverages=grid, conditions=np.asarray(CONDITIONS), sample_ids=np.asarray(sample_ids), measurement_size=np.asarray(size))
    metadata = {"split": split, "samples": expected, "measurement_size": size, "checkpoint_sha256": checkpoint_sha256, "degradation_config_sha256": degradation_sha256, "official_suim_test_evaluated": False, "model_retrained": False}
    (output / f"{split}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "uiis_deeplab_crc_sensitivity.yaml")
    parser.add_argument("--splits", nargs="+", default=["calibration", "confirmation"])
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not config["protocol"]["official_suim_test_evaluated"] is False or not config["experiment"]["official_suim_test_locked"]:
        raise ValueError("SUIM official TEST must remain locked.")
    for split in args.splits:
        build_split(config, split)


if __name__ == "__main__":
    main()
