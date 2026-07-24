"""Cache UIIS frozen logits for the preregistered alpha=0.10 CRC protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.label_mapping import ID2LABEL, LABEL2ID  # noqa: E402
from datasets.suim_dataset import SUIMDataset, build_eval_transform  # noqa: E402
from degradations.registry import build_image_degradation, load_conditions  # noqa: E402
from utils.reproducibility import set_seed  # noqa: E402


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def validate_protocol(config: dict[str, Any], requested_splits: list[str]) -> None:
    experiment, protocol = config["experiment"], config["protocol"]
    if experiment["fit_split"] != "calibration" or experiment["evaluation_split"] != "confirmation":
        raise ValueError("UIIS protocol requires calibration fitting and confirmation-only evaluation.")
    if not experiment["official_suim_test_locked"] or protocol["official_suim_test_evaluated"]:
        raise ValueError("The SUIM official TEST must remain locked.")
    if protocol["confirmation_used_for_fitting"] or protocol["model_retrained_after_protocol_freeze"]:
        raise ValueError("UIIS protocol forbids confirmation fitting and post-freeze retraining.")
    invalid = set(requested_splits) - {"calibration", "confirmation"}
    if invalid or not requested_splits:
        raise ValueError("Only calibration or confirmation caches are valid for UIIS CRC.")
    if "confirmation" in requested_splits and not protocol["confirmation_opened"]:
        raise ValueError("Confirmation is locked until calibration controller parameters are frozen.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "uiis_alpha010_crc.yaml")
    parser.add_argument("--splits", nargs="+", default=["calibration"])
    parser.add_argument("--conditions", nargs="+", help="Optional registered condition subset for a cache smoke test.")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    requested_splits = list(args.splits)
    validate_protocol(config, requested_splits)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen-logit caching.")
    experiment = config["experiment"]
    training_config_path = ROOT / experiment["training_config"]
    training_config = load_yaml(training_config_path)
    checkpoint_path = ROOT / experiment["checkpoint"]
    degradation_path = ROOT / experiment["degradation_config"]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_hash, degradation_hash = sha256(checkpoint_path), sha256(degradation_path)
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    if checkpoint.get("checkpoint_format") != "uiis_fixed_protocol_v1" or int(checkpoint.get("epoch", 0)) != int(training_config["training"]["epochs"]):
        raise ValueError("UIIS checkpoint must be the completed fixed-epoch protocol checkpoint.")
    if checkpoint.get("confirmation_evaluated") or checkpoint.get("official_suim_test_evaluated"):
        raise ValueError("Checkpoint provenance records an impermissible evaluation.")

    data, model_config, training = (training_config[key] for key in ("data", "model", "training"))
    set_seed(int(training_config["experiment"]["seed"]))
    device = torch.device("cuda")
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_config["pretrained_model"],
        num_labels=int(data["num_classes"]),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    conditions = load_conditions(degradation_path)
    names = [condition.name for condition in conditions]
    if names != list(experiment["conditions"]):
        raise ValueError("The UIIS cache must use exactly the frozen 13-condition registry.")
    if args.conditions:
        requested = set(args.conditions)
        conditions = [condition for condition in conditions if condition.name in requested]
        if {condition.name for condition in conditions} != requested:
            raise ValueError("Requested condition is absent from the frozen registry.")

    output = ROOT / experiment["cache_dir"]
    split_dir = ROOT / data["split_dir"]
    manifest: list[dict[str, Any]] = []
    for split in requested_splits:
        for condition in conditions:
            dataset = SUIMDataset(
                split_dir / f"{split}.csv",
                transform=build_eval_transform(int(data["image_size"])),
                image_degradation=build_image_degradation(condition),
            )
            loader = DataLoader(
                dataset,
                batch_size=int(data["batch_size"]),
                shuffle=False,
                num_workers=int(data["num_workers"]),
                pin_memory=True,
            )
            logits, labels, sample_ids = [], [], []
            with torch.no_grad():
                for batch in loader:
                    with torch.amp.autocast("cuda", enabled=bool(training["amp"])):
                        result = model(pixel_values=batch["pixel_values"].to(device, non_blocking=True)).logits
                    logits.append(result.cpu().to(torch.float16))
                    # Semantic labels are restricted to 0..7 by the UIIS
                    # admission gate.  Store them compactly, then restore
                    # long dtype only when computing losses/risk curves.
                    labels.append(batch["labels"].cpu().to(torch.uint8))
                    sample_ids.extend(str(sample_id) for sample_id in batch["sample_id"])
            payload = {
                "sample_id": sample_ids,
                "condition": condition.name,
                "degradation_type": condition.degradation_type,
                "severity": condition.severity,
                "split": split,
                "logits": torch.cat(logits),
                "labels": torch.cat(labels),
                "checkpoint_sha256": checkpoint_hash,
                "degradation_config_sha256": degradation_hash,
                "training_config_sha256": sha256(training_config_path),
                "image_size": int(data["image_size"]),
                "logits_shape": list(logits[0].shape[1:]),
                "dtype": "float16",
                "labels_dtype": "uint8",
                "official_suim_test_evaluated": False,
                "confirmation_evaluated": split == "confirmation",
            }
            path = output / split / f"{condition.name}.pt"
            atomic_torch_save(payload, path)
            manifest.append(
                {
                    "split": split,
                    "condition": condition.name,
                    "samples": len(sample_ids),
                    "path": str(path.relative_to(ROOT)),
                    "checkpoint_sha256": checkpoint_hash,
                    "degradation_config_sha256": degradation_hash,
                }
            )
            print(f"cached {split}/{condition.name}: {len(sample_ids)} samples")
    manifest_path = output / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(
            {
                "entries": manifest,
                "requested_splits": requested_splits,
                "confirmation_opened": "confirmation" in requested_splits,
                "official_suim_test_evaluated": False,
                "model_retrained": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)


if __name__ == "__main__":
    main()
