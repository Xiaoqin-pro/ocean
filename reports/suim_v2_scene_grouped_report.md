# SUIM v2 reviewed scene-grouped protocol

Status: training-ready data protocol, 2026-07-22.

- Near-duplicate candidate pool: 72 complete high-precision candidates (`pHash <= 4`, `dHash <= 4`, or SegFormer feature cosine `>= 0.995`).
- Human review: 71 `same_scene`, 1 `different_scene`, 0 `pending`.
- Exact SHA groups: 1,622; reviewed scene groups: 1,557.
- Development exclusions: 66. This includes unresolved size repairs, exact duplicate/conflicting-label cases, and reviewed development scenes connected to official TEST.
- Formal split: train 1,167 / validation 146 / calibration 146 / immutable official TEST 110.
- Balance selection: 300 group-aware candidates, selected seed `20260851`, objective `0.015494853063111231`.
- Guarantees verified before training: no exact-SHA group leakage, no reviewed scene-group leakage across development splits, no development-to-official-TEST reviewed scene overlap, paths exist, masks contain only labels 0--7, and all eight classes occur in each development split.

Split CSV SHA-256 values are recorded in `data/suim_processed/quality_reports/v2_scene_grouped_summary.json`. Review decisions and exclusions are stored alongside it in the quality reports directory.
