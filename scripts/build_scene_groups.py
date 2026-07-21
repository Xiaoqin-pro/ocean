"""Build auditable scene groups from exact hashes plus manually confirmed near duplicates.

Candidate generation never changes a split by itself. Only review rows marked
``same_scene`` are unioned with exact-SHA groups when this script is rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phash(path: Path) -> int:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    coefficients = cv2.dct(gray)[:8, :8]
    threshold = np.median(coefficients.ravel()[1:])
    value = 0
    for bit in (coefficients > threshold).ravel():
        value = (value << 1) | int(bit)
    return value


def dhash(path: Path) -> int:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.uint8)
    value = 0
    for bit in (gray[:, 1:] > gray[:, :-1]).ravel():
        value = (value << 1) | int(bit)
    return value


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parent[max(first_root, second_root)] = min(first_root, second_root)


def image_embeddings(manifest: pd.DataFrame, batch_size: int) -> np.ndarray:
    from transformers import SegformerModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegformerModel.from_pretrained("nvidia/mit-b0").to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(manifest), batch_size):
            arrays: list[np.ndarray] = []
            for relative_path in manifest.iloc[start:start + batch_size]["image_path"]:
                with Image.open(PROJECT_ROOT / relative_path) as image:
                    arrays.append(np.asarray(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0)
            pixels = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
            pixels = ((pixels - mean) / std).to(device)
            output = model(pixel_values=pixels).last_hidden_state.mean(dim=(2, 3))
            vectors.append(torch.nn.functional.normalize(output, dim=1).cpu().numpy())
    return np.concatenate(vectors, axis=0)


def preserve_reviews(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_file():
        return {}
    previous = pd.read_csv(path)
    return {str(row.pair_key): row._asdict() for row in previous.itertuples(index=False)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate near-duplicate candidates and scene-group IDs.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "suim_processed" / "manifest.csv")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "data" / "suim_processed" / "quality_reports")
    parser.add_argument("--phash-threshold", type=int, default=14)
    parser.add_argument("--dhash-threshold", type=int, default=14)
    parser.add_argument("--with-segformer-embeddings", action="store_true")
    parser.add_argument("--embedding-threshold", type=float, default=0.96)
    parser.add_argument("--embedding-neighbors", type=int, default=6)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--max-candidates", type=int, default=300)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest).sort_values(["partition", "sample_id"]).reset_index(drop=True)
    manifest["image_sha256"] = [sha256_file(PROJECT_ROOT / path) for path in manifest["image_path"]]
    manifest["phash"] = [phash(PROJECT_ROOT / path) for path in manifest["image_path"]]
    manifest["dhash"] = [dhash(PROJECT_ROOT / path) for path in manifest["image_path"]]
    candidates: dict[str, dict[str, object]] = {}

    def add_pair(left_index: int, right_index: int, *, source: str, cosine: float | None = None) -> None:
        if left_index == right_index or manifest.at[left_index, "image_sha256"] == manifest.at[right_index, "image_sha256"]:
            return
        left, right = sorted((left_index, right_index))
        first, second = manifest.iloc[left], manifest.iloc[right]
        key = f"{first.sample_id}__{second.sample_id}"
        row = candidates.setdefault(key, {
            "pair_key": key, "sample_id_a": first.sample_id, "partition_a": first.partition, "image_path_a": first.image_path,
            "sample_id_b": second.sample_id, "partition_b": second.partition, "image_path_b": second.image_path,
            "phash_distance": int((int(first.phash) ^ int(second.phash)).bit_count()),
            "dhash_distance": int((int(first.dhash) ^ int(second.dhash)).bit_count()),
            "embedding_cosine": np.nan, "candidate_sources": "", "review_decision": "pending", "reviewer": "", "review_date": "", "notes": "",
        })
        sources = set(filter(None, str(row["candidate_sources"]).split(";")))
        sources.add(source)
        row["candidate_sources"] = ";".join(sorted(sources))
        if cosine is not None:
            row["embedding_cosine"] = max(float(cosine), float(row["embedding_cosine"]) if pd.notna(row["embedding_cosine"]) else -1.0)

    for left in range(len(manifest)):
        for right in range(left + 1, len(manifest)):
            p_distance = int((int(manifest.at[left, "phash"]) ^ int(manifest.at[right, "phash"])).bit_count())
            d_distance = int((int(manifest.at[left, "dhash"]) ^ int(manifest.at[right, "dhash"])).bit_count())
            if p_distance <= args.phash_threshold or d_distance <= args.dhash_threshold:
                add_pair(left, right, source="perceptual_hash")
    if args.with_segformer_embeddings:
        vectors = image_embeddings(manifest, args.embedding_batch_size)
        neighbors = NearestNeighbors(n_neighbors=min(args.embedding_neighbors + 1, len(manifest)), metric="cosine").fit(vectors)
        distances, indices = neighbors.kneighbors(vectors)
        for left, (row_distances, row_indices) in enumerate(zip(distances, indices)):
            for distance, right in zip(row_distances[1:], row_indices[1:]):
                cosine = 1.0 - float(distance)
                if cosine >= args.embedding_threshold:
                    add_pair(left, int(right), source="segformer_embedding", cosine=cosine)
    review_path = args.report_dir / "near_duplicate_review.csv"
    previous = preserve_reviews(review_path)
    ordered = sorted(candidates.values(), key=lambda row: (-(row["embedding_cosine"] if pd.notna(row["embedding_cosine"]) else -1.0), row["phash_distance"], row["dhash_distance"], row["pair_key"]))[:args.max_candidates]
    for row in ordered:
        if row["pair_key"] in previous:
            for key in ("review_decision", "reviewer", "review_date", "notes"):
                row[key] = previous[row["pair_key"]].get(key, row[key])
    args.report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ordered).to_csv(review_path, index=False)
    union_find = UnionFind(manifest["sample_id"].tolist())
    for _, group in manifest.groupby("image_sha256"):
        members = group["sample_id"].tolist()
        for member in members[1:]:
            union_find.union(members[0], member)
    confirmed = [row for row in ordered if row["review_decision"] == "same_scene"]
    for row in confirmed:
        union_find.union(str(row["sample_id_a"]), str(row["sample_id_b"]))
    manifest["scene_group_id"] = [f"scene:{union_find.find(sample_id)}" for sample_id in manifest["sample_id"]]
    manifest["scene_group_source"] = "exact_sha"
    confirmed_ids = {str(row["sample_id_a"]) for row in confirmed} | {str(row["sample_id_b"]) for row in confirmed}
    manifest.loc[manifest["sample_id"].isin(confirmed_ids), "scene_group_source"] = "exact_sha_plus_reviewed_near_duplicate"
    manifest[["sample_id", "partition", "image_path", "image_sha256", "phash", "dhash", "scene_group_id", "scene_group_source"]].to_csv(args.report_dir / "scene_group_members.csv", index=False)
    summary = {
        "exact_sha_groups": int(manifest["image_sha256"].nunique()), "scene_groups": int(manifest["scene_group_id"].nunique()),
        "candidate_count": int(len(ordered)), "pending_candidates": int(sum(row["review_decision"] == "pending" for row in ordered)),
        "confirmed_same_scene_pairs": int(len(confirmed)), "embeddings_used": bool(args.with_segformer_embeddings),
        "protocol_status": "pending_manual_near_duplicate_review" if any(row["review_decision"] == "pending" for row in ordered) else "review_complete",
    }
    (args.report_dir / "scene_group_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
