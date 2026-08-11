"""Train source-anchored Domain-Conditioned Prototype Transport (DCPT)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional
import yaml
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.label_mapping import ID2LABEL, LABEL2ID  # noqa: E402
from reliability.dcpt_seg import DomainConditionedSPT, gate_accuracy, shared_prototype_alignment_loss  # noqa: E402
from reliability.sdtc_seg import clean_identity_loss  # noqa: E402
from reliability.spt_seg import semantic_prototype_transport_loss  # noqa: E402
from reliability.utd_seg import semantic_trajectory_distillation_loss  # noqa: E402
from scripts.train_dts_seg_gate0 import atomic_torch_save, sha256  # noqa: E402
from scripts.train_joint_cb_utd import CombinedTrajectoryDataset  # noqa: E402

FORMAT = "dcpt_seg_v1"


class EpochSubset(Dataset[dict[str, object]]):
    def __init__(self, source: CombinedTrajectoryDataset, indices: list[int]) -> None:
        self.source = source
        self.indices = indices

    def set_epoch(self, epoch: int) -> None:
        self.source.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.source[self.indices[index]]


def _load_source(config: dict, domain: int, device: torch.device) -> torch.nn.Module:
    base = SegformerForSemanticSegmentation.from_pretrained(
        config["model"]["pretrained_model"], num_labels=8, id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True,
    ).to(device)
    path = ROOT / str(config["model"]["suim_source_checkpoint"] if domain == 0 else config["model"]["uiis_source_checkpoint"])
    payload = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in payload:
        raise ValueError(f"Source checkpoint lacks model_state_dict: {path}")
    if domain == 0:
        required = {"variant": "B", "model_name": "segformer", "epoch": 100, "run_kind": "formal", "epoch_completed": True}
        for key, expected in required.items():
            if payload.get(key) != expected:
                raise ValueError(f"SUIM source has invalid {key}.")
        if any(bool(payload.get(key, True)) for key in ("validation_evaluated", "calibration_evaluated", "official_suim_test_evaluated")):
            raise ValueError("SUIM source records prohibited evaluation access.")
    else:
        if payload.get("checkpoint_format") != "uiis_scdi_replication_v1" or payload.get("variant") != "F4" or payload.get("epoch") != 8 or payload.get("smoke"):
            raise ValueError("UIIS source must be the frozen train-only F4 checkpoint.")
        if payload.get("confirmation_evaluated") or payload.get("official_suim_test_evaluated"):
            raise ValueError("UIIS source records prohibited evaluation access.")
    base.load_state_dict(payload["model_state_dict"])
    return base


def build_model(config: dict, device: torch.device) -> DomainConditionedSPT:
    suim = _load_source(config, 0, device)
    uiis = _load_source(config, 1, device)
    model = DomainConditionedSPT(
        (suim, uiis), tuple(config["model"]["hidden_sizes"]), int(config["model"]["num_classes"]), int(config["model"]["rectifier_reduction"])
    ).to(device)
    return model


def _keep_sources_eval(model: DomainConditionedSPT) -> None:
    for expert in model.experts:
        expert.base_model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dcpt.yaml")
    parser.add_argument("--smoke-batches", type=int, default=0)
    args = parser.parse_args()
    config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.smoke_batches < 0:
        raise ValueError("smoke-batches must be non-negative")
    seed = int(config["experiment"]["seed"]); random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("DCPT requires CUDA.")
    device = torch.device("cuda"); data_config = config["data"]; training = config["training"]
    combined = CombinedTrajectoryDataset(ROOT / str(data_config["suim_train_csv"]), ROOT / str(data_config["uiis_train_csv"]), ROOT / str(data_config["degradation_registry"]), int(data_config["image_size"]))
    domain_indices = {domain: [index for index, value in enumerate(combined.frame.domain.tolist()) if value == domain_name] for domain, domain_name in ((0, "suim"), (1, "uiis"))}
    datasets = {domain: EpochSubset(combined, indices) for domain, indices in domain_indices.items()}
    loaders = {domain: DataLoader(dataset, batch_size=int(training["batch_size"]), shuffle=True, num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(seed + domain)) for domain, dataset in datasets.items()}
    model = build_model(config, device); model.train(); _keep_sources_eval(model)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(training["amp"])); epochs = 1 if args.smoke_batches else int(training["epochs"]); output = ROOT / str(config["experiment"]["output_dir"]) / ("smoke" if args.smoke_batches else "formal"); history: list[dict[str, float]] = []
    model.train(); _keep_sources_eval(model)
    for epoch in range(1, epochs + 1):
        totals = {key: 0.0 for key in ("ce", "prototype", "shared", "distillation", "identity", "anchor", "gate", "gate_accuracy", "pairs")}; batches = 0
        for domain in (0, 1):
            datasets[domain].set_epoch(epoch)
            for batch_index, batch in enumerate(loaders[domain]):
                if args.smoke_batches and batch_index >= args.smoke_batches: break
                labels = batch["labels"].to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True)
                clean_reference = None; teacher_logits = None; total_loss = labels.float().sum() * 0.0; gate_term = labels.float().sum() * 0.0; gate_acc = labels.float().sum() * 0.0
                clean_result = model.forward_expert(batch["clean"].to(device, non_blocking=True), domain)
                with torch.no_grad():
                    other_clean = model.forward_expert(batch["clean"].to(device, non_blocking=True), 1 - domain)
                first, second = (clean_result, other_clean) if domain == 0 else (other_clean, clean_result)
                gate_logits = model.gate_from_outputs(first, second); domain_target = torch.full((labels.shape[0],), domain, device=device, dtype=torch.long); gate_term = functional.cross_entropy(gate_logits, domain_target); gate_acc = gate_accuracy(gate_logits, domain_target)
                clean_reference = tuple(item.detach() for item in clean_result.raw_features); teacher_logits = clean_result.logits.detach(); clean_identity = clean_identity_loss(clean_result.raw_features, clean_result.residuals)
                clean_full_logits = functional.interpolate(clean_result.logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
                clean_ce = functional.cross_entropy(clean_full_logits, labels, ignore_index=255) / 4.0
                clean_shared, clean_shared_pairs = shared_prototype_alignment_loss(clean_result.corrected_features, labels, model.shared_prototypes, num_classes=int(config["model"]["num_classes"]), minimum_pixels=int(training["prototype_minimum_pixels"]))
                clean_anchor = sum(item.float().pow(2).mean() for item in clean_result.residuals)
                total_loss = total_loss + clean_ce + float(training["shared_weight"]) * clean_shared + float(training["anchor_weight"]) * clean_anchor
                totals["ce"] += float(clean_ce.detach()); totals["shared"] += float(clean_shared.detach()); totals["anchor"] += float(clean_anchor.detach()); totals["pairs"] += float(clean_shared_pairs)
                del clean_full_logits
                for key in ("s1", "s2", "s3"):
                    pixels = batch[key].to(device, non_blocking=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(training["amp"])):
                        result = model.forward_expert(pixels, domain); logits = result.logits; full_logits = functional.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False); ce = functional.cross_entropy(full_logits, labels, ignore_index=255) / 4.0
                        prototype, pairs = semantic_prototype_transport_loss(clean_reference, result.corrected_features, labels, num_classes=int(config["model"]["num_classes"]), minimum_pixels=int(training["prototype_minimum_pixels"]), margin=float(training["prototype_margin"]))
                        shared, shared_pairs = shared_prototype_alignment_loss(result.corrected_features, labels, model.shared_prototypes, num_classes=int(config["model"]["num_classes"]), minimum_pixels=int(training["prototype_minimum_pixels"]))
                        distillation = semantic_trajectory_distillation_loss(teacher_logits, logits, labels, temperature=float(training["distillation_temperature"]), confidence_power=float(training["confidence_power"]), class_balance=True)
                        anchor = sum(item.float().pow(2).mean() for item in result.residuals)
                        view_loss = ce + float(training["prototype_weight"]) * prototype + float(training["shared_weight"]) * shared + float(training["distillation_weight"]) * (int(key[-1]) / 3.0) * distillation + float(training["anchor_weight"]) * anchor
                    total_loss = total_loss + view_loss
                    totals["ce"] += float(ce.detach()); totals["prototype"] += float(prototype.detach()); totals["shared"] += float(shared.detach()); totals["distillation"] += float(distillation.detach()); totals["anchor"] += float(anchor.detach()); totals["pairs"] += float(pairs + shared_pairs); del pixels, result, logits, full_logits, view_loss
                total_loss = total_loss + float(training["clean_identity_weight"]) * clean_identity + float(training["gate_weight"]) * gate_term
                scaler.scale(total_loss).backward(); scaler.step(optimizer); scaler.update(); batches += 1; totals["identity"] += float(clean_identity.detach()); totals["gate"] += float(gate_term.detach()); totals["gate_accuracy"] += float(gate_acc.detach())
                if batch_index % 50 == 0: print(f"DCPT epoch={epoch} domain={domain} batch={batch_index}/{len(loaders[domain])} ce={totals['ce']/max(batches,1):.4f} gate={totals['gate']/max(batches,1):.4f} gate_acc={totals['gate_accuracy']/max(batches,1):.3f}", flush=True)
                del clean_result, other_clean, clean_identity, total_loss
        if batches == 0: raise RuntimeError("No training batch completed.")
        row = {"epoch": float(epoch), "batches": float(batches), **{key: value / batches for key, value in totals.items()}}; history.append(row); print(json.dumps(row, sort_keys=True), flush=True)
        payload = {"checkpoint_format": FORMAT, "variant": "DCPT", "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(), "history": history, "config_sha256": sha256(config_path), "smoke": bool(args.smoke_batches), "calibration_evaluated": False, "confirmation_evaluated": False, "official_suim_test_evaluated": False}
        atomic_torch_save(payload, output / "checkpoints" / "last.pt")
    if not args.smoke_batches: atomic_torch_save(payload, output / "checkpoints" / "final.pt")


if __name__ == "__main__":
    main()
