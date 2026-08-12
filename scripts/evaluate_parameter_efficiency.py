"""Evaluate parameter-efficiency baselines on the locked 13-condition roles."""
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
from scripts.evaluate_dts_seg_gate0 import evaluate_condition  # noqa: E402
from scripts.evaluate_sadr import SADRModel  # noqa: E402
from scripts.train_sadr import build_model as build_sadr_model  # noqa: E402
from scripts.train_uiis_scdi_replication import build_models as build_uiis_model  # noqa: E402


def validate_split(path: Path, expected: int, role: str) -> None:
    frame = pd.read_csv(path)
    if len(frame) != expected:
        raise PermissionError(f"{role} must contain exactly {expected} rows, found {len(frame)}")


def load_uiis_checkpoint(model: torch.nn.Module, checkpoint: Path, device: torch.device, expected_format: str) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("checkpoint_format") != expected_format or bool(payload.get("smoke", True)):
        raise ValueError(f"Invalid formal checkpoint: {checkpoint}")
    if any(bool(payload.get(key, True)) for key in ("confirmation_evaluated", "official_suim_test_evaluated")):
        raise ValueError(f"Checkpoint records prohibited evaluation access: {checkpoint}")
    model.load_state_dict(payload["model_state_dict"])
    return model.eval()


def load_models(uiis_config: dict, sadr_config: dict, device: torch.device) -> dict[str, torch.nn.Module]:
    source = build_uiis_model(uiis_config, "F4", device).eval()
    full = load_uiis_checkpoint(
        build_uiis_model(uiis_config, "F4", device),
        ROOT / "outputs/uiis_scdi_replication/formal/F4/checkpoints/final.pt",
        device,
        "uiis_scdi_replication_v1",
    )
    partial: dict[str, torch.nn.Module] = {}
    for scope in ("head", "last_block"):
        partial[scope] = load_uiis_checkpoint(
            build_uiis_model(uiis_config, "F4", device),
            ROOT / f"outputs/parameter_efficiency/formal/{scope}/checkpoints/final.pt",
            device,
            "parameter_efficiency_v1",
        )
    base, front = build_sadr_model(sadr_config, device)
    payload = torch.load(ROOT / "outputs/sadr_long/formal/checkpoints/final.pt", map_location=device, weights_only=False)
    if payload.get("checkpoint_format") != "sadr_v1" or payload.get("variant") != "SADR" or bool(payload.get("smoke", True)):
        raise ValueError("Invalid formal SADR checkpoint")
    if any(bool(payload.get(key, True)) for key in ("confirmation_evaluated", "official_suim_test_evaluated")):
        raise ValueError("SADR checkpoint records prohibited evaluation access")
    front.load_state_dict(payload["model_state_dict"])
    return {"frozen": source, "sadr": SADRModel(base.eval(), front.eval()).eval(), "head": partial["head"], "last_block": partial["last_block"], "full": full}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("confirmation", "suim_official"), default="confirmation")
    parser.add_argument("--allow-confirmation", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/parameter_efficiency.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    data = config["data"]
    if args.split == "confirmation":
        if not args.allow_confirmation:
            raise PermissionError("Confirmation evaluation is locked; pass --allow-confirmation explicitly.")
        csv_path, expected = ROOT / "data/uiis_processed/splits/uiis_alpha010_confirmation/confirmation.csv", 511
    else:
        csv_path, expected = ROOT / "data/suim_processed/splits/v2_scene_grouped_deduplicated/test.csv", 110
    validate_split(csv_path, expected, args.split)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    uiis_config = yaml.safe_load((ROOT / str(data["uiis_config"])).read_text(encoding="utf-8"))
    sadr_config = yaml.safe_load((ROOT / "configs/sadr_long.yaml").read_text(encoding="utf-8"))
    conditions = load_conditions(ROOT / str(data["degradation_registry"]))
    if len(conditions) != 13:
        raise ValueError("Expected exactly 13 degradation conditions")
    models = load_models(uiis_config, sadr_config, device)
    output = ROOT / "outputs/parameter_efficiency" / f"{args.split}_evaluation"
    rows = []
    for name, model in models.items():
        for condition in conditions:
            cache = output / "conditions" / f"{name}_{condition.name}.json"
            if cache.is_file():
                metric = json.loads(cache.read_text(encoding="utf-8"))
            else:
                metric = evaluate_condition(model, csv_path, condition, 384, device)
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(metric, indent=2) + "\n", encoding="utf-8")
            rows.append({"variant": name, **metric})
            print(f"{args.split}/{name}/{condition.name}: mIoU={metric['miou']:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "condition_metrics.csv", index=False)
    means = frame.groupby("variant", sort=False).miou.mean()
    frozen = float(means["frozen"])
    summary = {
        "split": args.split, "images": expected, "conditions": len(conditions),
        "confirmation_evaluated": args.split == "confirmation", "official_suim_test_evaluated": args.split == "suim_official",
        "mean_miou": {key: float(value) for key, value in means.items()},
        "gain_vs_frozen_pp": {key: float(100.0 * (value - frozen)) for key, value in means.items()},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
