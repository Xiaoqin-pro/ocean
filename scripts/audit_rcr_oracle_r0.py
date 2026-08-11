"""RCR Gate R0: frozen full-image repairability/oracle analysis on 231 development scenes."""
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
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.label_mapping import ID2LABEL, LABEL2ID  # noqa: E402
from datasets.suim_dataset import SUIMDataset, build_eval_transform  # noqa: E402
from degradations.registry import build_image_degradation, load_conditions  # noqa: E402
from metrics.segmentation import confusion_matrix, metrics_from_confusion_matrix  # noqa: E402
from reliability.rcr_oracle import equal_probability_ensemble, pixelwise_oracle_union, repair_statistics  # noqa: E402
from scripts.train_deeplabv3_mobilenetv3 import build_model as build_deeplab  # noqa: E402


CANDIDATES = ("segformer_to_deeplab", "deeplab_to_segformer", "segformer_to_ensemble", "deeplab_to_ensemble")
REGIONS = ("full", "boundary", "interior")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", dir=path.parent, delete=False, encoding="utf-8", newline="") as handle:
        temporary = Path(handle.name); frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=path.parent, delete=False, encoding="utf-8") as handle:
        temporary = Path(handle.name); json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True); handle.write("\n")
    os.replace(temporary, path)


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _boundary(labels: torch.Tensor) -> torch.Tensor:
    valid = labels.ne(255)
    value = labels.float().masked_fill(~valid, 0.0).unsqueeze(1)
    maximum = functional.max_pool2d(value, 7, 1, 3)
    minimum = -functional.max_pool2d(-value, 7, 1, 3)
    return maximum[:, 0].ne(minimum[:, 0]) & valid


def _resize(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)


def preflight(config: Mapping[str, Any]) -> tuple[Path, list[Any]]:
    scope = config["scope"]
    if not bool(scope["method_development_read_allowed"]) or any(bool(scope[key]) for key in (
        "training_allowed", "crop_refinement_allowed", "communication_simulation_allowed", "calibration_read_allowed",
        "validation_read_allowed", "official_suim_test_read_allowed", "external_data_read_allowed",
    )):
        raise PermissionError("RCR R0 permits only fixed development inference.")
    data = config["data"]
    development = (ROOT / str(data["development_csv"])).resolve()
    if development.name != "risk_head_development.csv" or not development.is_file() or sha256(development) != str(data["development_csv_sha256"]).upper():
        raise PermissionError("RCR R0 requires the exact frozen development CSV.")
    frame = pd.read_csv(development)
    if len(frame) != int(data["samples"]) or frame.sample_id.duplicated().any():
        raise ValueError("Frozen development membership is invalid.")
    for model in config["models"].values():
        for key in ("checkpoint", "config"):
            path = ROOT / str(model[key]); hash_key = f"{key}_sha256"
            if not path.is_file() or sha256(path) != str(model[hash_key]).upper():
                raise ValueError(f"Frozen model provenance changed: {key}")
    registry = ROOT / str(config["degradations"]["registry_config"])
    if not registry.is_file() or sha256(registry) != str(config["degradations"]["registry_config_sha256"]).upper():
        raise ValueError("Frozen degradation registry changed.")
    conditions = load_conditions(registry)
    if [item.name for item in conditions] != list(config["degradations"]["conditions"]):
        raise ValueError("RCR R0 condition registry differs from its frozen order.")
    return development, conditions


def _load_models(config: Mapping[str, Any], device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    seg_cfg = yaml.safe_load((ROOT / str(config["models"]["segformer"]["config"])).read_text(encoding="utf-8"))
    seg = SegformerForSemanticSegmentation.from_pretrained(seg_cfg["model"]["pretrained_model"], num_labels=8, id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True).to(device)
    seg_state = torch.load(ROOT / str(config["models"]["segformer"]["checkpoint"]), map_location=device, weights_only=False)
    seg.load_state_dict(seg_state["model_state_dict"]); seg.eval()
    deep = build_deeplab(8, None).to(device)
    deep_state = torch.load(ROOT / str(config["models"]["deeplab"]["checkpoint"]), map_location=device, weights_only=False)
    if deep_state.get("checkpoint_format") != "deeplabv3_mobilenetv3_suim_v1" or bool(deep_state.get("official_test_evaluated", True)):
        raise ValueError("DeepLab checkpoint is not the frozen formal baseline.")
    deep.load_state_dict(deep_state["model_state_dict"]); deep.eval()
    return seg, deep


def _region_metrics(prediction: torch.Tensor, labels: torch.Tensor, region: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor]:
    masked = labels.clone(); masked[~region] = 255
    matrix = confusion_matrix(prediction, masked, num_classes=8, ignore_index=255).cpu()
    return metrics_from_confusion_matrix(matrix), matrix


def _foreground_miou(metrics: Mapping[str, Any]) -> float:
    scores = np.asarray(metrics["per_class_iou"], dtype=float)[1:]
    return float(np.nanmean(scores))


def _candidate_predictions(seg: torch.Tensor, deep: torch.Tensor, ensemble: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {
        "segformer_to_deeplab": (seg, deep), "deeplab_to_segformer": (deep, seg),
        "segformer_to_ensemble": (seg, ensemble), "deeplab_to_ensemble": (deep, ensemble),
    }


def _bootstrap(scene: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []; rng = np.random.default_rng(seed)
    for candidate in CANDIDATES:
        for region in REGIONS:
            table = scene[(scene.candidate == candidate) & (scene.region == region)]
            pivot = table.groupby("sample_id")[["oracle_delta_miou", "repair_rate", "damage_rate", "net_correction_mass"]].mean()
            if len(pivot) != 231:
                raise ValueError("RCR scene cluster is incomplete.")
            for metric in pivot:
                values = pivot[metric].to_numpy(float)
                draws = values[rng.integers(0, len(values), size=(iterations, len(values)))].mean(axis=1)
                rows.append({"candidate": candidate, "region": region, "metric": metric, "clusters": len(values),
                             "iterations": iterations, "mean": float(values.mean()), "ci95_low": float(np.quantile(draws, .025)),
                             "ci95_high": float(np.quantile(draws, .975))})
    return pd.DataFrame(rows)


def _gate(aggregate: pd.DataFrame, bootstrap: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    criteria = config["gate_r0"]; results = []
    for candidate in CANDIDATES:
        full = aggregate[(aggregate.candidate == candidate) & (aggregate.region == "full")]
        boundary = aggregate[(aggregate.candidate == candidate) & (aggregate.region == "boundary")]
        family = full[full.degradation_type != "clean"].groupby("degradation_type").oracle_delta_miou.mean()
        boot = bootstrap[(bootstrap.candidate == candidate) & (bootstrap.region == "full") & (bootstrap.metric == "oracle_delta_miou")].iloc[0]
        checks = {
            "repair_rate": float(full.repair_rate.mean()) >= float(criteria["repair_rate_min"]),
            "oracle_macro_miou": float(full.oracle_delta_miou.mean()) >= float(criteria["oracle_macro_miou_improvement_min"]),
            "foreground_macro": float(full.foreground_oracle_delta_miou.mean()) >= float(criteria["foreground_macro_miou_improvement_min"]),
            "boundary": float(boundary.oracle_delta_miou.mean()) >= float(criteria["boundary_oracle_macro_miou_improvement_min"]),
            "families": int((family > 0).sum()) >= int(criteria["positive_family_count_min"]),
            "bootstrap": float(boot.ci95_low) > float(criteria["scene_bootstrap_ci95_lower_must_exceed"]),
        }
        results.append({"candidate": candidate, "checks": checks, "passes": all(checks.values()),
                        "full_macro_oracle_delta_miou": float(full.oracle_delta_miou.mean()),
                        "full_repair_rate": float(full.repair_rate.mean()), "boundary_oracle_delta_miou": float(boundary.oracle_delta_miou.mean()),
                        "positive_families": int((family > 0).sum()), "scene_bootstrap_ci95_low": float(boot.ci95_low)})
    passing = [item for item in results if item["passes"]]
    selected = max(passing, key=lambda item: item["full_macro_oracle_delta_miou"])["candidate"] if passing else None
    return {"gate": "R0_full_image_repairability_oracle", "candidate_results": results, "passes": bool(passing),
            "r1_authorized": bool(passing), "selected_candidate_for_future_r1": selected,
            "calibration_evaluated": False, "validation_evaluated": False, "official_suim_test_evaluated": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs" / "rcr_oracle_r0.yaml"); args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("RCR R0 requires CUDA inference.")
    config_path = args.config.resolve(); config = load_config(config_path); development, conditions = preflight(config)
    device = torch.device("cuda"); segformer, deeplab = _load_models(config, device); image_size = int(config["data"]["image_size"])
    output = ROOT / str(config["experiment"]["output_dir"]); scene_rows: list[dict[str, Any]] = []; aggregate_matrices: defaultdict[tuple[str, str, str], dict[str, torch.Tensor]] = defaultdict(lambda: {"local": torch.zeros((8, 8), dtype=torch.long), "remote": torch.zeros((8, 8), dtype=torch.long), "oracle": torch.zeros((8, 8), dtype=torch.long)})
    with torch.no_grad():
        for condition in conditions:
            dataset = SUIMDataset(development, transform=build_eval_transform(image_size), image_degradation=build_image_degradation(condition))
            loader = DataLoader(dataset, batch_size=int(config["data"]["batch_size"]), shuffle=False, num_workers=0, pin_memory=True)
            for batch in loader:
                labels = batch["labels"].to(device); pixels = batch["pixel_values"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=True):
                    seg_logits = _resize(segformer(pixel_values=pixels).logits, labels)
                    deep_logits = _resize(deeplab(pixels)["out"], labels)
                seg_pred, deep_pred = seg_logits.argmax(1), deep_logits.argmax(1)
                ensemble = equal_probability_ensemble(seg_logits, deep_logits)
                valid = labels.ne(255); boundary = valid & _boundary(labels); regions = {"full": valid, "boundary": boundary, "interior": valid & ~boundary}
                for candidate, (local, remote) in _candidate_predictions(seg_pred, deep_pred, ensemble).items():
                    oracle = pixelwise_oracle_union(local, remote, labels)
                    for region_name, region in regions.items():
                        for index, sample_id in enumerate(batch["sample_id"]):
                            stats = repair_statistics(local[index], remote[index], labels[index], region[index])
                            local_metrics, _ = _region_metrics(local[index], labels[index], region[index]); remote_metrics, _ = _region_metrics(remote[index], labels[index], region[index]); oracle_metrics, _ = _region_metrics(oracle[index], labels[index], region[index])
                            scene_rows.append({"sample_id": str(sample_id), "candidate": candidate, "condition": condition.name, "degradation_type": condition.degradation_type, "severity": condition.severity, "region": region_name, **stats,
                                               "local_miou": local_metrics["miou"], "remote_miou": remote_metrics["miou"], "oracle_miou": oracle_metrics["miou"],
                                               "oracle_delta_miou": oracle_metrics["miou"] - local_metrics["miou"], "foreground_local_miou": _foreground_miou(local_metrics), "foreground_oracle_miou": _foreground_miou(oracle_metrics), "foreground_oracle_delta_miou": _foreground_miou(oracle_metrics) - _foreground_miou(local_metrics)})
                        key = (candidate, condition.name, region_name)
                        for name, pred in (("local", local), ("remote", remote), ("oracle", oracle)):
                            aggregate_matrices[key][name] += confusion_matrix(pred, labels.masked_fill(~region, 255), num_classes=8, ignore_index=255).cpu()
            print(json.dumps({"condition": condition.name, "completed": True}), flush=True)
    scene = pd.DataFrame(scene_rows)
    if len(scene) != len(conditions) * int(config["data"]["samples"]) * len(CANDIDATES) * len(REGIONS) or scene.duplicated(["sample_id", "candidate", "condition", "region"]).any(): raise AssertionError("R0 scene table is incomplete or duplicated.")
    aggregate_rows: list[dict[str, Any]] = []; per_class: list[dict[str, Any]] = []
    for (candidate, condition, region), matrices in aggregate_matrices.items():
        local, remote, oracle = (metrics_from_confusion_matrix(matrices[name]) for name in ("local", "remote", "oracle"))
        subset = scene[(scene.candidate == candidate) & (scene.condition == condition) & (scene.region == region)]
        source = subset.iloc[0]
        aggregate_rows.append({"candidate": candidate, "condition": condition, "degradation_type": source.degradation_type, "severity": int(source.severity), "region": region,
                               "repair_rate": float(subset.repaired_pixels.sum() / max(subset.local_wrong_pixels.sum(), 1)),
                               "damage_rate": float(subset.damaged_pixels.sum() / max(subset.local_correct_pixels.sum(), 1)),
                               "net_correction_mass": float((subset.repaired_pixels.sum() - subset.damaged_pixels.sum()) / max(subset.valid_pixels.sum(), 1)),
                               "local_miou": local["miou"], "remote_miou": remote["miou"], "oracle_miou": oracle["miou"], "oracle_delta_miou": oracle["miou"] - local["miou"], "foreground_local_miou": _foreground_miou(local), "foreground_oracle_miou": _foreground_miou(oracle), "foreground_oracle_delta_miou": _foreground_miou(oracle) - _foreground_miou(local)})
        for class_id, (liou, riou, oiou) in enumerate(zip(local["per_class_iou"], remote["per_class_iou"], oracle["per_class_iou"])):
            per_class.append({"candidate": candidate, "condition": condition, "region": region, "class_id": class_id, "local_iou": liou, "remote_iou": riou, "oracle_iou": oiou, "oracle_delta_iou": oiou - liou})
    aggregate = pd.DataFrame(aggregate_rows); bootstrap = _bootstrap(scene, int(config["metrics"]["bootstrap_replicates"]), int(config["experiment"]["seed"])); decision = _gate(aggregate, bootstrap, config)
    _atomic_csv(scene, output / "per_scene_metrics.csv"); _atomic_csv(aggregate, output / "aggregate_metrics.csv"); _atomic_csv(pd.DataFrame(per_class), output / "per_class_metrics.csv"); _atomic_csv(bootstrap, output / "scene_cluster_bootstrap.csv"); _atomic_json(decision, output / "screen_decision.json")
    files = {path.name: sha256(path) for path in output.glob("*.csv")} | {"screen_decision.json": sha256(output / "screen_decision.json")}
    _atomic_json({"schema": "rcr_oracle_r0_output_v1", "implementation_commit": _commit(), "config_sha256": sha256(config_path), "development_csv_sha256": sha256(development), "files_sha256": files, "model_sha256": {name: sha256(ROOT / str(value["checkpoint"])) for name, value in config["models"].items()}, "model_retrained": False, "crop_refinement": False, "communication_simulation": False, "calibration_evaluated": False, "validation_evaluated": False, "official_suim_test_evaluated": False}, output / "manifest.json")


if __name__ == "__main__": main()
