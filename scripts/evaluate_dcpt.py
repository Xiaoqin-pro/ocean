"""Evaluate source-anchored DCPT with automatic and oracle routing."""
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
from scripts.train_dts_seg_gate0 import build_model as build_suim_f4  # noqa: E402
from scripts.train_uiis_scdi_replication import build_models as build_uiis_f4  # noqa: E402
from scripts.train_dcpt import build_model as build_dcpt  # noqa: E402


class ExpertWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, domain: int) -> None:
        super().__init__(); self.model = model; self.domain = domain

    def forward(self, pixel_values: torch.Tensor):
        return self.model.forward_expert(pixel_values, self.domain)


def _split_info(config: dict, split: str) -> tuple[Path, int, bool, int]:
    data = config["data"]
    if split == "suim_official":
        return ROOT / str(data["suim_test_csv"]), 110, True, 0
    if split == "calibration":
        return ROOT / str(data["uiis_calibration_csv"]), 508, False, 1
    if split == "confirmation":
        return ROOT / str(data["uiis_confirmation_csv"]), 511, False, 1
    raise ValueError(split)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/dcpt_fast.yaml"); parser.add_argument("--split", choices=("suim_official", "calibration", "confirmation"), default="suim_official"); parser.add_argument("--allow-confirmation", action="store_true"); args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")); csv_path, expected, official, oracle_domain = _split_info(config, args.split)
    if args.split == "confirmation" and not args.allow_confirmation: raise PermissionError("Confirmation is locked; pass --allow-confirmation after development freeze.")
    if len(pd.read_csv(csv_path)) != expected: raise PermissionError(f"Unexpected split size for {args.split}.")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required.")
    device = torch.device("cuda"); conditions = [item for item in load_conditions(ROOT / str(config["data"]["degradation_registry"])) if item.name in set(config["evaluation"]["conditions"])]
    output = ROOT / str(config["experiment"]["output_dir"]) / f"{args.split}_evaluation"; output.mkdir(parents=True, exist_ok=True); rows: list[dict[str, object]] = []
    dcpt_payload = torch.load(ROOT / str(config["experiment"]["output_dir"]) / "formal/checkpoints/final.pt", map_location=device, weights_only=False)
    if dcpt_payload.get("checkpoint_format") != "dcpt_seg_v1" or dcpt_payload.get("variant") != "DCPT" or dcpt_payload.get("smoke"): raise ValueError("Invalid DCPT checkpoint.")
    dcpt = build_dcpt(config, device); dcpt.load_state_dict(dcpt_payload["model_state_dict"]); dcpt.eval(); dcpt.hard_routing = True
    if official:
        suim_cfg = yaml.safe_load((ROOT / "configs/dts_seg_gate0.yaml").read_text(encoding="utf-8")); baseline_payload = torch.load(ROOT / "outputs/dts_seg_gate0/formal/F4/checkpoints/final.pt", map_location=device, weights_only=False); baseline = build_suim_f4(suim_cfg, device); baseline.load_state_dict(baseline_payload["model_state_dict"]); baseline.eval()
    else:
        uiis_cfg = yaml.safe_load((ROOT / "configs/uiis_scdi_replication.yaml").read_text(encoding="utf-8")); baseline_payload = torch.load(ROOT / "outputs/uiis_scdi_replication/formal/F4/checkpoints/final.pt", map_location=device, weights_only=False); baseline = build_uiis_f4(uiis_cfg, "F4", device); baseline.load_state_dict(baseline_payload["model_state_dict"]); baseline.eval()
    oracle = ExpertWrapper(dcpt, oracle_domain).to(device).eval()
    models = (("SourceF4", baseline), ("DCPT-hard-gated", dcpt), ("DCPT-oracle", oracle))
    image_size = int(config["data"].get("evaluation_image_size", 384))
    for variant, model in models:
        for condition in conditions:
            metric = evaluate_condition(model, csv_path, condition, image_size, device); row = {"variant": variant, **metric}; rows.append(row); print(f"{args.split}/{variant}/{condition.name}: mIoU={metric['miou']:.4f}", flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(output / "condition_metrics.csv", index=False); summary = {"split": args.split, "official_suim_test_evaluated": official, "confirmation_evaluated": args.split == "confirmation", "rows": len(frame), "images": expected, "mean_miou": {variant: float(frame[frame.variant == variant].miou.mean()) for variant in frame.variant.unique()}}; (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8"); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
