"""Run the DTH Gate-1 label-signal audit on the frozen method_train role only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.suim_dataset import build_eval_transform  # noqa: E402
from degradations.registry import Condition, load_conditions  # noqa: E402
from reliability.dth_horizon import (  # noqa: E402
    FAMILIES, HORIZON_NAMES, apply_coupled_degradation, boundary_mask, family_conditions,
    first_failure_horizons, status_counts, trajectory_metadata, validate_monotonic_operator_parameters,
)
from reliability.ft_reliability import assert_method_train_access  # noqa: E402
from scripts.train_ft_reliability_pilot import CHECKPOINT_FORMAT, PROTOCOL_COMMIT, _build_segformer, load_config  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=path.parent, delete=False, encoding="utf-8") as handle:
        temporary = Path(handle.name)
        json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", dir=path.parent, delete=False, encoding="utf-8", newline="") as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def load_dth_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def preflight(config: Mapping[str, Any], config_path: Path) -> tuple[Path, Path, Path, pd.DataFrame, list[Condition]]:
    scope = config["scope"]
    if any(bool(scope[key]) for key in (
        "training_allowed", "head_training_allowed", "method_development_read_allowed", "validation_read_allowed",
        "calibration_read_allowed", "official_suim_test_read_allowed", "external_data_read_allowed",
    )):
        raise PermissionError("DTH Gate-1 scope permits only frozen method_train inference.")
    data = config["data"]
    split = (ROOT / str(data["method_train_csv"])).resolve()
    allowed_directory = (ROOT / str(data["allowed_split_dir"])).resolve()
    assert_method_train_access(split, allowed_directory=allowed_directory)
    if sha256(split) != str(data["method_train_csv_sha256"]).upper():
        raise ValueError("The frozen method_train CSV hash changed.")
    frame = pd.read_csv(split).sort_values("sample_id").reset_index(drop=True)
    required = {"sample_id", "image_path", "mask_path"}
    if required.difference(frame.columns) or len(frame) != int(data["method_train_samples"]) or frame.sample_id.duplicated().any():
        raise ValueError("Gate-1 must receive exactly the frozen 936 unique method_train samples.")
    checkpoint = (ROOT / str(config["base_model"]["checkpoint"])).resolve()
    if not checkpoint.is_file() or sha256(checkpoint) != str(config["base_model"]["checkpoint_sha256"]).upper():
        raise ValueError("The frozen Variant-B checkpoint is missing or has changed.")
    degradation_path = (ROOT / str(config["degradations"]["registry_config"])).resolve()
    if sha256(degradation_path) != str(config["degradations"]["registry_config_sha256"]).upper():
        raise ValueError("The frozen degradation registry hash changed.")
    conditions = load_conditions(degradation_path)
    validate_monotonic_operator_parameters(conditions)
    for family in FAMILIES:
        family_conditions(conditions, family)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    return split, checkpoint, degradation_path, frame, conditions


def load_variant_b(config: Mapping[str, Any], checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required = {
        "checkpoint_format": str(config["base_model"]["checkpoint_format"]),
        "protocol_commit": str(config["base_model"]["protocol_commit"]),
        "variant": "B", "model_name": "segformer", "run_kind": "formal",
        "checkpoint_selection": "final_epoch", "epoch": int(config["base_model"]["final_epoch"]), "epoch_completed": True,
    }
    for key, expected in required.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"Variant-B checkpoint metadata mismatch: {key}")
    if any(bool(checkpoint.get(key, True)) for key in (
        "method_development_evaluated", "validation_evaluated", "calibration_evaluated", "official_suim_test_evaluated",
    )):
        raise PermissionError("Frozen Variant-B checkpoint records impermissible data access.")
    ft_config = load_config(ROOT)
    model = _build_segformer(ft_config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def _resize_logits(logits: torch.Tensor, size: int) -> torch.Tensor:
    return functional.interpolate(logits, size=(size, size), mode="bilinear", align_corners=False)


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return None
    return float(pd.Series(x).rank(method="average").corr(pd.Series(y).rank(method="average"), method="pearson"))


def _bin_counts(values: np.ndarray, horizons: np.ndarray, edges: list[float], *, score: str) -> list[dict[str, object]]:
    bins = np.digitize(values, np.asarray(edges, dtype=np.float64)[1:-1], right=False)
    rows: list[dict[str, object]] = []
    for index in range(len(edges) - 1):
        chosen = bins == index
        for code, name in HORIZON_NAMES.items():
            rows.append({"score": score, "bin": index, "left": edges[index], "right": edges[index + 1],
                         "status": name, "pixels": int(np.sum(chosen & (horizons == code)))})
    return rows


def _screen_decision(scene_table: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    criteria = config["screening_criteria"]
    family_rows: list[dict[str, Any]] = []
    recovery_excess = 0
    for family in FAMILIES:
        subset = scene_table[scene_table.family == family]
        totals = {status: int(subset[f"{status}_count"].sum()) for status in HORIZON_NAMES.values()}
        eligible = int(subset.eligible_count.sum())
        eligible_scenes = int((subset.eligible_count > 0).sum())
        event_scenes = int((subset[["H1_count", "H2_count", "H3_count"]].sum(axis=1) > 0).sum())
        h2_scenes, h3_scenes = int((subset.H2_count > 0).sum()), int((subset.H3_count > 0).sum())
        recovery_denominator = int(subset.first_failure_count.sum())
        recovery_rate = float(subset.post_failure_recovery_count.sum() / recovery_denominator) if recovery_denominator else 0.0
        max_status_rate = max(totals.values()) / eligible if eligible else 1.0
        recovery_excess += int(recovery_rate > float(criteria["recovery_rate_max"]))
        passes_local = (
            eligible_scenes >= int(criteria["eligible_scenes_min"])
            and event_scenes >= int(criteria["event_scenes_min"])
            and h2_scenes >= int(criteria["h2_event_scenes_min"])
            and h3_scenes >= int(criteria["h3_event_scenes_min"])
            and max_status_rate <= float(criteria["max_single_status_rate_max"])
        )
        family_rows.append({"family": family, "eligible_pixels": eligible, "eligible_scenes": eligible_scenes,
                            "event_scenes": event_scenes, "h2_event_scenes": h2_scenes, "h3_event_scenes": h3_scenes,
                            "max_status_rate": max_status_rate, "post_failure_recovery_rate": recovery_rate,
                            "passes_local_signal_requirements": passes_local})
    passed_families = sum(bool(row["passes_local_signal_requirements"]) for row in family_rows)
    recovery_ok = recovery_excess <= int(criteria["recovery_rate_excess_families_max"])
    return {"gate": "gate_1_label_signal_audit", "family_results": family_rows,
            "passed_family_count": passed_families, "families_required": int(criteria["families_required_to_pass"]),
            "recovery_excess_family_count": recovery_excess, "recovery_rule_passes": recovery_ok,
            "passes": bool(passed_families >= int(criteria["families_required_to_pass"]) and recovery_ok),
            "gate_2_started": False, "method_development_evaluated": False, "validation_evaluated": False,
            "calibration_evaluated": False, "official_suim_test_evaluated": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "dth_horizon_signal_screen.yaml")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("DTH Gate-1 requires CUDA for the frozen SegFormer inference audit.")
    config_path = args.config.resolve()
    config = load_dth_config(config_path)
    split, checkpoint_path, degradation_path, frame, conditions = preflight(config, config_path)
    device, image_size = torch.device("cuda"), int(config["data"]["image_size"])
    model = load_variant_b(config, checkpoint_path, device)
    transform = build_eval_transform(image_size)
    output = ROOT / str(config["experiment"]["output_dir"])
    scene_rows: list[dict[str, object]] = []
    stratified: defaultdict[tuple[str, int, str, str], int] = defaultdict(int)
    transition_totals: defaultdict[tuple[str, str], int] = defaultdict(int)
    score_bins: defaultdict[tuple[str, str, int, float, float, str], int] = defaultdict(int)
    correlation_rows: list[dict[str, object]] = []
    clean_condition = next(item for item in conditions if item.name == "clean")
    with torch.no_grad():
        for number, row in frame.iterrows():
            sample_id = str(row.sample_id)
            with Image.open(ROOT / str(row.image_path)) as handle:
                image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
            with Image.open(ROOT / str(row.mask_path)) as handle:
                raw_label = np.asarray(handle, dtype=np.uint8)
            clean = transform(image=apply_coupled_degradation(image, clean_condition, sample_id=sample_id, family="color"), mask=raw_label)
            labels = clean["mask"].to(device=device, dtype=torch.long)
            clean_pixels = clean["image"].unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", enabled=True):
                clean_logits = _resize_logits(model(pixel_values=clean_pixels).logits, image_size)
            clean_prediction = clean_logits.argmax(dim=1)[0]
            probabilities = clean_logits.float().softmax(dim=1)[0]
            msp = probabilities.max(dim=0).values
            normalized_entropy = -(probabilities * probabilities.clamp_min(torch.finfo(torch.float32).eps).log()).sum(dim=0) / np.log(8.0)
            top_gap = clean_logits.float().topk(2, dim=1).values[0, 0] - clean_logits.float().topk(2, dim=1).values[0, 1]
            boundary = boundary_mask(labels, radius=int(config["labels"]["boundary_radius"]))
            for family in FAMILIES:
                family_views: list[torch.Tensor] = []
                for condition in family_conditions(conditions, family):
                    transformed = transform(image=apply_coupled_degradation(image, condition, sample_id=sample_id, family=family), mask=raw_label)
                    if not torch.equal(transformed["mask"], clean["mask"]):
                        raise AssertionError("DTH trajectory label alignment changed across severities.")
                    family_views.append(transformed["image"])
                with torch.amp.autocast("cuda", enabled=True):
                    logits = _resize_logits(model(pixel_values=torch.stack(family_views).to(device)).logits, image_size)
                labels_out = first_failure_horizons(clean_prediction, logits[0].argmax(0), logits[1].argmax(0), logits[2].argmax(0), labels)
                horizon, eligible = labels_out.horizon, labels_out.eligible
                counts = status_counts(horizon, eligible)
                transitions = {name: int(value.sum().item()) for name, value in labels_out.transitions.items()}
                transition_totals.update({(family, name): value for name, value in transitions.items()})
                row_out: dict[str, object] = {"sample_id": sample_id, "family": family, "scene_group_id": str(row.get("scene_group_id", "")),
                                                "trajectory_seed": trajectory_metadata(sample_id, family, conditions)["trajectory_seed"],
                                                "trajectory_hash": trajectory_metadata(sample_id, family, conditions)["trajectory_hash"],
                                                "eligible_count": int(eligible.sum().item()), **{f"{key}_count": value for key, value in counts.items()},
                                                "first_failure_count": int(sum(counts[key] for key in ("H1", "H2", "H3"))),
                                                "post_failure_recovery_count": int(labels_out.post_failure_recovery.sum().item()),
                                                "s1_wrong_to_s2_correct_count": int(labels_out.s1_wrong_to_s2_correct.sum().item()),
                                                "s2_wrong_to_s3_correct_count": int(labels_out.s2_wrong_to_s3_correct.sum().item()), **transitions}
                scene_rows.append(row_out)
                regions = {"full": eligible, "boundary": eligible & boundary, "interior": eligible & ~boundary}
                for class_id in range(8):
                    for region_name, region_mask in regions.items():
                        for code, status in HORIZON_NAMES.items():
                            stratified[(family, class_id, region_name, status)] += int((labels.eq(class_id) & region_mask & horizon.eq(code)).sum().item())
                eligible_np, horizon_np = eligible.detach().cpu().numpy(), horizon.detach().cpu().numpy()
                for score_name, score in (("msp", msp), ("normalized_entropy", normalized_entropy), ("top_gap", top_gap)):
                    values = score.detach().cpu().numpy()[eligible_np]
                    targets = horizon_np[eligible_np]
                    correlation_rows.append({"sample_id": sample_id, "family": family, "score": score_name,
                                             "eligible_pixels": int(len(values)), "spearman_horizon": _spearman(values, targets)})
                    for bin_row in _bin_counts(values, targets, list(config["reporting"]["score_bins"][score_name]), score=score_name):
                        score_bins[(family, str(bin_row["score"]), int(bin_row["bin"]), float(bin_row["left"]), float(bin_row["right"]), str(bin_row["status"]))] += int(bin_row["pixels"])
            if (number + 1) % 20 == 0 or number + 1 == len(frame):
                print(json.dumps({"processed": int(number + 1), "total": int(len(frame))}), flush=True)
    scene_table = pd.DataFrame(scene_rows)
    strata_rows: list[dict[str, object]] = []
    for family in FAMILIES:
        for class_id in range(8):
            for region in ("full", "boundary", "interior"):
                denominator = sum(stratified[(family, class_id, region, status)] for status in HORIZON_NAMES.values())
                for status in HORIZON_NAMES.values():
                    count = stratified[(family, class_id, region, status)]
                    strata_rows.append({"family": family, "class_id": class_id, "region": region, "status": status,
                                        "eligible_pixels": denominator, "pixels": count, "rate": count / denominator if denominator else None})
    transitions_table = pd.DataFrame([{"family": family, "transition": transition, "pixels": count}
                                      for (family, transition), count in sorted(transition_totals.items())])
    score_bins_table = pd.DataFrame([{"family": family, "score": score, "bin": index, "left": left, "right": right, "status": status, "pixels": pixels}
                                     for (family, score, index, left, right, status), pixels in score_bins.items()])
    _atomic_csv(scene_table, output / "scene_summary.csv")
    _atomic_csv(pd.DataFrame(strata_rows), output / "stratified_summary.csv")
    _atomic_csv(transitions_table, output / "transition_summary.csv")
    _atomic_csv(pd.DataFrame(correlation_rows), output / "score_correlations.csv")
    _atomic_csv(score_bins_table, output / "score_bins.csv")
    decision = _screen_decision(scene_table, config)
    _atomic_json(decision, output / "screen_decision.json")
    output_files = {path.name: sha256(path) for path in output.glob("*.csv")} | {"screen_decision.json": sha256(output / "screen_decision.json")}
    _atomic_json({"schema": "dth_horizon_signal_screen_output_v1", "implementation_commit": _git_commit(),
                  "config_sha256": sha256(config_path), "method_train_csv_sha256": sha256(split),
                  "checkpoint_sha256": sha256(checkpoint_path), "degradation_registry_sha256": sha256(degradation_path),
                  "files_sha256": output_files, "method_development_evaluated": False, "validation_evaluated": False,
                  "calibration_evaluated": False, "official_suim_test_evaluated": False, "head_trained": False}, output / "manifest.json")


if __name__ == "__main__":
    main()
