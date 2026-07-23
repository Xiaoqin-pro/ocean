import pandas as pd

from scripts.build_uiis_scene_groups import build_scene_groups


def test_scene_grouping_binds_exact_and_near_hash_pairs(monkeypatch, tmp_path):
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "image_path": ["a.jpg", "b.jpg", "c.jpg"],
            "image_sha256": ["sha-a", "sha-b", "sha-b"],
        }
    )
    hashes = {"a.jpg": 0b111100, "b.jpg": 0b0011, "c.jpg": 0b0011}
    monkeypatch.setattr("scripts.build_uiis_scene_groups.phash", lambda path: hashes[path.name])
    members, edges, summary = build_scene_groups(frame, tmp_path, phash_threshold=2)
    assert members.loc[members["sample_id"].isin(["b", "c"]), "scene_group_id"].nunique() == 1
    assert summary["admitted_images"] == 3
    assert summary["scene_group_count"] == 2
    assert set(edges["evidence"]) == {"exact_sha256", "phash_le_2"}
