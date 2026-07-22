# SegFormer-B0 SUIM v2 Baseline

Date: 2026-07-22  
Experiment: `segformer_b0_suim_v2_scene`  
Status: accepted clean baseline for the degradation pilot

## Provenance

- Git commit: `7144d0034fbace2c72632bf63d8e661bd62601c8`
- Branch: `experiment/baseline-v2`
- Split: `v2_scene_grouped_deduplicated` (train 1,167 / validation 146 / calibration 146 / official TEST 110)
- Data protocol: reviewed scene groups, exact duplicates, confirmed near-duplicate scenes, conflicting-label duplicates, and all 37 size-anomalous masks excluded from the development protocol.
- Configuration: `configs/baseline/segformer_b0_suim_v2_scene.yaml`
- Seed: `20260721`
- Model: ImageNet-pretrained `nvidia/mit-b0` SegFormer with an 8-class segmentation head.
- Input / training: 384 x 384, batch size 4, AdamW, learning rate 6e-5, weight decay 0.01, cross-entropy, AMP.
- Checkpoint: `outputs/segformer_b0_suim_v2_scene/checkpoints/best.pt`
- Checkpoint SHA-256: `62CF10BA021B7E24429477C9C9C4690650EB0945D39E19DA0A8DEB3BB1132A5A`
- Configuration SHA-256: `AA23D9893C22F92D663F51E3A0C1FE1DD06AF37CD97F3FC937C99B1C2138197A`
- Official TEST evaluated: **no**.

## Training and checkpoint selection

- Planned and completed epochs: 100 / 100.
- Training log span: 07:16:22 to 09:33:34 on 2026-07-22 (2 h 17 m 12 s from the first epoch record; startup excluded).
- Best validation mIoU checkpoint: epoch 80, 0.602939.
- Epoch-100 validation mIoU: 0.589127.
- Epoch-100 train loss: 0.126526.
- Minimum logged validation loss: 0.518005 at epoch 15. It is not the model-selection criterion; `best.pt` is selected solely by validation mIoU.

## Re-evaluation of frozen `best.pt`

| Split | Loss | Pixel Accuracy | Mean Accuracy | mIoU | Mean Dice | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.752677 | 0.856697 | 0.694933 | 0.602939 | 0.719708 | 0.752650 | 0.239949 | 0.089123 |
| Calibration | 0.805830 | 0.836863 | 0.746654 | 0.652563 | 0.767237 | 0.805796 | 0.270882 | 0.100071 |

The calibration split is not used to choose the checkpoint. It is reserved for fitting calibration-only parameters in the next phase.

## Per-class results

| Class | Validation IoU | Validation Dice | Validation classwise ECE | Calibration IoU | Calibration Dice | Calibration classwise ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| background_waterbody | 0.850889 | 0.919438 | 0.026123 | 0.881749 | 0.937159 | 0.017862 |
| human_divers | 0.770223 | 0.870199 | 0.001260 | 0.761224 | 0.864426 | 0.001983 |
| aquatic_plants | 0.136069 | 0.239543 | 0.021492 | 0.195510 | 0.327074 | 0.015099 |
| wrecks_ruins | 0.672778 | 0.804384 | 0.015625 | 0.725564 | 0.840958 | 0.016304 |
| robots_instruments | 0.305870 | 0.468454 | 0.001219 | 0.750296 | 0.857336 | 0.000266 |
| reefs_invertebrates | 0.797895 | 0.887588 | 0.057779 | 0.727180 | 0.842043 | 0.078498 |
| fish_vertebrates | 0.658678 | 0.794220 | 0.013565 | 0.710082 | 0.830466 | 0.009500 |
| seafloor_rocks | 0.631109 | 0.773841 | 0.045742 | 0.468897 | 0.638434 | 0.065951 |

## Failure cases and interpretation

- `aquatic_plants` is the weakest validation class (IoU 0.136069), so vegetation-like regions should be a primary target when reviewing degraded predictions.
- `robots_instruments` is also weak on validation (IoU 0.305870) but substantially higher on calibration (0.750296). This indicates limited-support or composition sensitivity; report both splits rather than treating the calibration score as a generalization claim.
- The largest classwise calibration errors are associated with `reefs_invertebrates` and `seafloor_rocks`, while the global ECE is 0.089123 on validation and 0.100071 on calibration.
- The lower v2 validation mIoU than the invalid v1 engineering result (0.651173) is expected after removing leakage and uncertain labels. V1 remains an internal, invalid-for-paper comparison only.

## Acceptance boundary

The model, split, configuration, and `best.pt` are frozen for the controlled-degradation pilot. No official TEST prediction, hyperparameter selection, or calibration fitting was performed in this acceptance step. The next phase must fit any temperature or calibration parameter on `calibration` only and compare clean/degraded conditions on `validation`.

Local generated evaluation artifacts are retained under `outputs/segformer_b0_suim_v2_scene/evaluation/`, including metric JSON files, per-class CSVs, confusion matrices, metadata, and validation examples. Checkpoints and large generated images are intentionally not committed.
