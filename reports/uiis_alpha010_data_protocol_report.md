# UIIS alpha=0.10 confirmation: data protocol and speed benchmark

## Scope

This report records preparation only for the preregistered independent UIIS
confirmation. It contains no formal UIIS model result.

- SUIM official TEST was not evaluated.
- It was used only to screen UIIS/SUIM duplicate and near-duplicate scenes.
- The UIIS confirmation split was not loaded during the training speed benchmark.

## Semantic conversion

UIIS provides COCO instance annotations. All 4,628 images were deterministically
rasterized to the existing eight-class semantic label space:

| UIIS class | Semantic ID |
| --- | ---: |
| background | 0 |
| human divers | 1 |
| aquatic plants | 2 |
| wrecks / ruins | 3 |
| robots | 4 |
| reefs | 5 |
| fish | 6 |
| sea-floor | 7 |

- Images converted: 4,628
- Source annotations: 28,410
- Empty masks: 0
- Image/mask dimension mismatches: 0
- Deterministic visual checks: 20 image-mask overlays; no obvious semantic or
  spatial misalignment was found.

## Independence screen

The screen compared every UIIS image against all 1,569 images in the frozen
SUIM protocol, including the locked TEST only for duplication audit.

| Decision | UIIS images |
| --- | ---: |
| Exact-SHA or pHash <= 2 automatic SUIM overlap exclusion | 1,220 |
| pHash = 4 visual-review same-scene exclusion | 13 |
| UIIS-internal exact duplicate exclusion | 5 |
| Admitted to UIIS confirmation protocol | 3,390 |

All 13 pHash = 4 pairs were reviewed as the same scene or adjacent frame.
There are no pending manual review rows.

Within the admitted set, exact SHA and pHash <= 2 edges were bound into
conservative scene groups before splitting:

- Scene groups: 2,508
- Multi-image groups: 737
- Largest group: 5 images
- Development-to-development scene-group leakage: 0 by construction

## Fixed split

The frozen UIIS alpha=0.10 confirmation v1 split was chosen from 300
group-aware, label-balance candidates with seed 20260779.

| Partition | Images |
| --- | ---: |
| train | 2,371 |
| calibration | 508 |
| confirmation | 511 |

Calibration is reserved for the preregistered CRC fit. Confirmation remains
unseen by the model-selection and CRC-fitting steps.

## Non-formal RTX 4060 speed benchmark

Configuration: SegFormer-B0, 384x384, AMP, batch size 2, 150 train steps.
The benchmark writes only to outputs/uiis_speed_benchmark/; its checkpoint
and loss are not formal experiment artifacts.

| Measurement | Value |
| --- | ---: |
| Mean step time | 0.15725 s |
| Mean data-loading time | 0.01409 s |
| Peak allocated GPU memory | 388.85 MiB |
| Estimated epoch time | 186.42 s (3.11 min) |
| 60-epoch pure-training estimate | 3.11 h |
| Practical 60-epoch estimate with saving and variation | about 3.5-4 h |

The formal UIIS trainer uses fixed epochs (no confirmation access), and atomically
replaces last.pt after each completed epoch. The checkpoint includes model,
optimizer, scheduler, AMP scaler, epoch/global step, and Python/NumPy/PyTorch
random states. The associated serialization and restoration tests pass.

## Readiness

The data protocol, split, configuration, short speed test, and checkpoint
mechanism are ready. Start the formal UIIS training only during a stable
four-hour-or-longer power window.
