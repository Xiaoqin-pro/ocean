"""Evaluate the calibration-fitted quality-conditioned SADR gate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as functional
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from degradations.registry import load_conditions  # noqa: E402
from scripts.evaluate_dts_seg_gate0 import evaluate_condition  # noqa: E402
from scripts.fit_quality_gate import quality_features  # noqa: E402
from scripts.train_sadr import build_model  # noqa: E402


class QualityGateModel(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, front: torch.nn.Module, gate: dict) -> None:
        super().__init__(); self.base = base; self.front = front; self.mean = torch.tensor(gate["feature_mean"], dtype=torch.float32); self.scale = torch.tensor(gate["feature_scale"], dtype=torch.float32); self.coef = torch.tensor(gate["coef"], dtype=torch.float32); self.intercept = torch.tensor(float(gate["intercept"]), dtype=torch.float32)

    def forward(self, pixel_values: torch.Tensor):
        with torch.no_grad():
            base_logits = self.base(pixel_values=pixel_values).logits
        restored, _ = self.front(pixel_values)
        sadr_logits = self.base(pixel_values=restored).logits
        features = (quality_features(pixel_values).float() - self.mean.to(pixel_values.device)) / self.scale.to(pixel_values.device).clamp_min(1e-6)
        alpha = torch.sigmoid(features @ self.coef.to(pixel_values.device) + self.intercept.to(pixel_values.device)).view(-1, 1, 1, 1)
        return type("Output", (), {"logits": base_logits + alpha * (sadr_logits - base_logits)})()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/sadr_fast.yaml"); parser.add_argument("--split", choices=("confirmation", "suim_official"), default="confirmation"); parser.add_argument("--allow-confirmation", action="store_true"); parser.add_argument("--gate-dir", type=Path, default=None); args = parser.parse_args(); config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8")); data = config["data"]
    if args.split == "confirmation":
        if not args.allow_confirmation: raise PermissionError("Confirmation is locked.")
        csv_path, expected = ROOT / str(data["confirmation_csv"]), 511
    else: csv_path, expected = ROOT / "data/suim_processed/splits/v2_scene_grouped_deduplicated/test.csv", 110
    if len(pd.read_csv(csv_path)) != expected: raise PermissionError("Unexpected frozen split size.")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required.")
    device = torch.device("cuda"); base, front = build_model(config, device); payload = torch.load(ROOT / str(config["experiment"]["output_dir"]) / "formal/checkpoints/final.pt", map_location=device, weights_only=False); front.load_state_dict(payload["model_state_dict"]); gate_dir = args.gate_dir.resolve() if args.gate_dir is not None else ROOT / str(config["experiment"]["output_dir"]) / "quality_gate"; gate = json.loads((gate_dir / "gate.json").read_text(encoding="utf-8")); base.eval(); front.eval(); gated = QualityGateModel(base, front, gate).eval(); conditions = [item for item in load_conditions(ROOT / str(data["degradation_registry"])) if item.name in set(config["evaluation"]["conditions"])]
    rows = []; image_size = int(data["evaluation_image_size"])
    for condition in conditions:
        metric = evaluate_condition(gated, csv_path, condition, image_size, device); rows.append({"variant": "Q-SADR", **metric}); print(f"{args.split}/Q-SADR/{condition.name}: mIoU={metric['miou']:.4f}", flush=True)
    result = pd.DataFrame(rows); output = ROOT / str(config["experiment"]["output_dir"]) / f"{args.split}_quality_gate_evaluation_{gate.get('fit_split', 'calibration')}"; output.mkdir(parents=True, exist_ok=True); result.to_csv(output / "condition_metrics.csv", index=False); summary = {"split": args.split, "confirmation_evaluated": args.split == "confirmation", "official_suim_test_evaluated": args.split == "suim_official", "rows": len(result), "images": expected, "mean_miou": float(result.miou.mean()), "gate_source": f"{gate.get('fit_images', 'unknown')}-image {gate.get('fit_split', 'calibration')} split"}; (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8"); print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
