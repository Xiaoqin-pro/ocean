"""Write deterministic AquaRiskMap feature caches for inner risk-head splits only.

The command deliberately accepts only ``--model`` and ``--split``.  In
particular, it has no arbitrary CSV, validation, calibration, or TEST option.
"""
from __future__ import annotations

import argparse
import datetime as datetime
import hashlib
import io
import json
import os
import sys
import tempfile
import zipfile
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
from reliability.aquariskmap import CONDITIONS, FEATURE_SCHEMA_VERSION, IMAGE_SIZE, build_features, validate_cache_payload  # noqa: E402
from scripts.cache_temperature_logits import build_frozen_model, model_logits  # noqa: E402


ALLOWED_MODELS = ("segformer", "deeplab")
ALLOWED_SPLITS = ("risk_head_train", "risk_head_development")


def load_config(root: Path = ROOT) -> dict[str, Any]:
    return yaml.safe_load((root / "configs" / "aquariskmap_pilot.yaml").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def resolve_context(root: Path, config: Mapping[str, Any], *, model_name: str, split: str) -> tuple[Path, Path, Path, Path]:
    if model_name not in ALLOWED_MODELS or split not in ALLOWED_SPLITS:
        raise ValueError("Only preregistered model and risk-head split names are permitted.")
    protocol = config["protocol"]
    split_root = (root / str(protocol["risk_head_split"])).resolve()
    split_path = (split_root / f"{split}.csv").resolve()
    if split_path.parent != split_root or not split_path.is_file():
        raise ValueError("Risk-head CSV must reside in the preregistered inner-split directory.")
    model = config["base_models"][model_name]
    baseline_path = (root / str(model["baseline_config"])).resolve()
    checkpoint_path = (root / str(model["checkpoint"])).resolve()
    degradation_path = (root / str(config["degradations"]["config"])).resolve()
    if not all(path.is_file() for path in (baseline_path, checkpoint_path, degradation_path)):
        raise FileNotFoundError("A frozen baseline, checkpoint, or degradation registry is missing.")
    return split_path, baseline_path, checkpoint_path, degradation_path


def validate_membership(root: Path, config: Mapping[str, Any], *, split: str, frame: pd.DataFrame) -> None:
    expected_path, _, _, _ = resolve_context(root, config, model_name="segformer", split=split)
    expected = pd.read_csv(expected_path)
    if set(frame["sample_id"]) != set(expected["sample_id"]) or len(frame) != len(expected):
        raise ValueError("Cache sample IDs do not exactly match the preregistered inner split.")
    split_dir = root / "data" / "suim_processed" / "splits" / "v2_scene_grouped_deduplicated"
    barred = set()
    for name in ("val", "calibration", "test"):
        barred.update(pd.read_csv(split_dir / f"{name}.csv")["sample_id"])
    if set(frame["sample_id"]).intersection(barred):
        raise ValueError("Risk-head cache membership overlaps validation, calibration, or official TEST.")


def prediction_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return functional.interpolate(logits.float(), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False).argmax(dim=1).to(torch.uint8)


def deterministic_npz_bytes(payload: Mapping[str, object]) -> bytes:
    """Serialize an NPZ with fixed metadata ordering and timestamps."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for key in sorted(payload):
            value = np.asarray(payload[key])
            buffer = io.BytesIO()
            np.save(buffer, value, allow_pickle=False)
            entry = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o600 << 16
            archive.writestr(entry, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue()


def atomic_npz(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = deterministic_npz_bytes(payload)
    with tempfile.NamedTemporaryFile("wb", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_npz_payload(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def cache_is_complete(path: Path, *, split: str, model_name: str, checkpoint_sha256: str, degradation_config_sha256: str, source_image_sha256: str, source_mask_sha256: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = load_npz_payload(path)
        validate_cache_payload(payload, split=split, model_name=model_name)
    except (OSError, ValueError, zipfile.BadZipFile):
        return False
    return all(
        payload[key].item() == expected
        for key, expected in {
            "checkpoint_sha256": checkpoint_sha256,
            "degradation_config_sha256": degradation_config_sha256,
            "source_image_sha256": source_image_sha256,
            "source_mask_sha256": source_mask_sha256,
        }.items()
    )


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_sample_payload(*, features: list[np.ndarray], predicted_class: list[np.ndarray], sample_id: str, scene_group_id: str, split: str, model_name: str, checkpoint_sha256: str, degradation_config_sha256: str, source_image_sha256: str, source_mask_sha256: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "features": np.stack(features).astype(np.float16, copy=False),
        "predicted_class": np.stack(predicted_class).astype(np.uint8, copy=False),
        "conditions": np.asarray(CONDITIONS),
        "sample_id": sample_id,
        "scene_group_id": scene_group_id,
        "split": split,
        "model_name": model_name,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "degradation_config_sha256": degradation_config_sha256,
        "source_image_sha256": source_image_sha256,
        "source_mask_sha256": source_mask_sha256,
        "official_suim_test_evaluated": False,
    }
    validate_cache_payload(payload, split=split, model_name=model_name)
    return payload


def cache_split(root: Path, *, model_name: str, split: str, limit: int | None, resume: bool) -> Path:
    config = load_config(root)
    split_path, baseline_path, checkpoint_path, degradation_path = resolve_context(root, config, model_name=model_name, split=split)
    frame = pd.read_csv(split_path).sort_values("sample_id").reset_index(drop=True)
    validate_membership(root, config, split=split, frame=frame)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive.")
        frame = frame.iloc[:limit].copy()
    conditions = load_conditions(degradation_path)
    if tuple(condition.name for condition in conditions) != CONDITIONS:
        raise ValueError("Degradation registry differs from the preregistered 13-condition order.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen-model cache generation.")
    baseline = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    checkpoint_sha256, degradation_sha256 = sha256(checkpoint_path), sha256(degradation_path)
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    model = build_frozen_model(baseline["model"], int(baseline["data"]["num_classes"]), checkpoint, torch.device("cuda"))
    cache_root = root / config["experiment"]["output_dir"] / "cache" / model_name / split
    cache_root.mkdir(parents=True, exist_ok=True)
    datasets = [SUIMDataset(split_path, transform=build_eval_transform(int(baseline["data"]["image_size"])), image_degradation=build_image_degradation(condition)) for condition in conditions]
    batch_size = int(config["training"]["base_scenes_per_batch"])
    manifest_rows: list[dict[str, object]] = []
    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        current = frame.iloc[start:stop].reset_index(drop=True)
        features_by_sample: list[list[np.ndarray]] = [[] for _ in range(len(current))]
        predictions_by_sample: list[list[np.ndarray]] = [[] for _ in range(len(current))]
        for condition, dataset in zip(conditions, datasets):
            batch = [dataset[index] for index in range(start, stop)]
            sample_ids = [item["sample_id"] for item in batch]
            if sample_ids != current["sample_id"].tolist():
                raise AssertionError("Dataset order differs from the frozen risk-head split.")
            pixels = torch.stack([item["pixel_values"] for item in batch]).to("cuda", non_blocking=True)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=bool(baseline["training"]["amp"])):
                logits = model_logits(model, pixels)
            features = build_features(logits, pixels).cpu().numpy().astype(np.float16)
            predictions = prediction_from_logits(logits).cpu().numpy().astype(np.uint8)
            for local in range(len(current)):
                features_by_sample[local].append(features[local])
                predictions_by_sample[local].append(predictions[local])
        for local, row in current.iterrows():
            image_hash = sha256(root / str(row["image_path"]))
            mask_hash = sha256(root / str(row["mask_path"]))
            destination = cache_root / f"{row['sample_id']}.npz"
            complete = cache_is_complete(destination, split=split, model_name=model_name, checkpoint_sha256=checkpoint_sha256, degradation_config_sha256=degradation_sha256, source_image_sha256=image_hash, source_mask_sha256=mask_hash)
            if complete and resume:
                manifest_rows.append({"sample_id": str(row["sample_id"]), "path": destination.name, "sha256": sha256(destination), "reused": True})
                continue
            if complete and not resume:
                raise FileExistsError(f"Complete cache exists: {destination}; use --resume.")
            payload = build_sample_payload(features=features_by_sample[local], predicted_class=predictions_by_sample[local], sample_id=str(row["sample_id"]), scene_group_id=str(row["scene_group_id"]), split=split, model_name=model_name, checkpoint_sha256=checkpoint_sha256, degradation_config_sha256=degradation_sha256, source_image_sha256=image_hash, source_mask_sha256=mask_hash)
            atomic_npz(destination, payload)
            manifest_rows.append({"sample_id": str(row["sample_id"]), "path": destination.name, "sha256": sha256(destination), "reused": False})
        print(f"cached {model_name}/{split}: {stop}/{len(frame)}", flush=True)
    manifest = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "model_name": model_name,
        "checkpoint_sha256": checkpoint_sha256,
        "degradation_config_sha256": degradation_sha256,
        "split": split,
        "split_csv_sha256": sha256(split_path),
        "samples": len(frame),
        "scene_groups": int(frame["scene_group_id"].nunique()),
        "conditions": list(CONDITIONS),
        "files": manifest_rows,
        "validation_evaluated": False,
        "calibration_evaluated": False,
        "official_suim_test_evaluated": False,
    }
    atomic_json(cache_root / "cache_manifest.json", manifest)
    return cache_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=ALLOWED_MODELS)
    parser.add_argument("--split", required=True, choices=ALLOWED_SPLITS)
    parser.add_argument("--limit", type=int, help="Technical integration subset: first N fixed sample IDs only.")
    parser.add_argument("--resume", action="store_true", help="Reuse only files passing full schema and hash validation.")
    args = parser.parse_args()
    cache_split(ROOT, model_name=args.model, split=args.split, limit=args.limit, resume=args.resume)


if __name__ == "__main__":
    main()
