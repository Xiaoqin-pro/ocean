"""Build conservative UIIS scene groups before creating the confirmation split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def phash(path: Path) -> int:
    """Return the same 64-bit perceptual hash used by the UIIS/SUIM audit."""
    with Image.open(path) as image:
        grayscale = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    coefficients = cv2.dct(grayscale)[:8, :8]
    threshold = np.median(coefficients.ravel()[1:])
    value = 0
    for bit in (coefficients > threshold).ravel():
        value = (value << 1) | int(bit)
    return value


def build_scene_groups(frame: pd.DataFrame, root: Path, phash_threshold: int = 2) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required = {"sample_id", "image_path", "image_sha256"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Admission manifest missing columns: {sorted(missing)}")
    if not 0 <= phash_threshold <= 64:
        raise ValueError("pHash threshold must lie in [0, 64].")
    ordered = frame.sort_values("sample_id", kind="stable").reset_index(drop=True).copy()
    ids = ordered["sample_id"].tolist()
    union_find = UnionFind(ids)
    edges: list[dict[str, object]] = []

    for _, group in ordered.groupby("image_sha256", sort=True):
        group_ids = group["sample_id"].tolist()
        for sample_id in group_ids[1:]:
            union_find.union(group_ids[0], sample_id)
            edges.append({"sample_id_a": group_ids[0], "sample_id_b": sample_id, "distance": 0, "evidence": "exact_sha256"})

    hashes = [phash(root / path) for path in ordered["image_path"]]
    for left_index, left_hash in enumerate(hashes):
        for right_index in range(left_index + 1, len(hashes)):
            distance = int(left_hash ^ hashes[right_index]).bit_count()
            if distance <= phash_threshold:
                left_id, right_id = ids[left_index], ids[right_index]
                union_find.union(left_id, right_id)
                edges.append(
                    {
                        "sample_id_a": left_id,
                        "sample_id_b": right_id,
                        "distance": distance,
                        "evidence": f"phash_le_{phash_threshold}",
                    }
                )

    roots = {sample_id: union_find.find(sample_id) for sample_id in ids}
    ordered["scene_group_id"] = [f"uiis_scene:{roots[sample_id]}" for sample_id in ids]
    edge_frame = pd.DataFrame(edges, columns=["sample_id_a", "sample_id_b", "distance", "evidence"])
    counts = ordered["scene_group_id"].value_counts()
    summary = {
        "protocol_status": "automatic_conservative_scene_groups_complete",
        "admitted_images": int(len(ordered)),
        "scene_group_count": int(len(counts)),
        "multi_image_scene_group_count": int((counts > 1).sum()),
        "largest_scene_group": int(counts.max()),
        "exact_sha256_edges": int((edge_frame["evidence"] == "exact_sha256").sum()) if len(edge_frame) else 0,
        "phash_threshold": phash_threshold,
        "phash_edges": int((edge_frame["evidence"] == f"phash_le_{phash_threshold}").sum()) if len(edge_frame) else 0,
        "policy": "Exact SHA-256 and pHash<=2 pairs are bound into a common split; they are not used to select model settings.",
    }
    return ordered, edge_frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "uiis_processed" / "admission" / "admitted_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "uiis_processed" / "scene_groups")
    parser.add_argument("--phash-threshold", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    members, edges, summary = build_scene_groups(pd.read_csv(args.manifest), ROOT, args.phash_threshold)
    members.to_csv(args.output_dir / "scene_group_members.csv", index=False)
    edges.to_csv(args.output_dir / "automatic_scene_edges.csv", index=False)
    (args.output_dir / "scene_group_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
