# SegFormer-B0 / SUIM clean-baseline freeze

Date: 2026-07-21  
Experiment: `segformer_b0_suim_ce`  
Configuration: `configs/segformer_b0_suim_baseline.yaml`  
Fixed split: `v1_seed_20260721` (train 1,220 / val 152 / calibration 153 / official test 110)

## Scope and reproducibility

- Model: ImageNet-pretrained `nvidia/mit-b0` SegFormer encoder with an 8-class randomly initialized segmentation decode head.
- Input / optimizer: 384 x 384, batch size 4, AdamW, learning rate 6e-5, weight decay 0.01, cross-entropy loss, AMP.
- The run reached all 100 planned epochs. The training log spans 20:03:25 to 22:26:20 (2 h 22 m 56 s from the first epoch log; approximately 85.8 s per epoch). Startup before the first epoch is excluded.
- GPU peak memory was not instrumented during this completed run, so it is deliberately recorded as unavailable rather than reconstructed. The separate two-image smoke test used about 0.34 GB.
- Official TEST was not evaluated by this acceptance workflow.

## Frozen checkpoint

- Selection: `best.pt`, epoch 100, because this is the highest validation mIoU checkpoint.
- Best validation mIoU: 0.651173.
- Epoch-100 validation mIoU: 0.651173.
- Minimum logged validation loss: 0.568116 at epoch 5. This is not the selection criterion; checkpoint selection uses validation mIoU.
- `best.pt` SHA-256: `863A2B073AC4055BAAAF64F1E48C91FA737C69E7DD301E69AE746BCEE02D3512`.
- Configuration SHA-256: `E8E69A0E72BFA2D3BCFD6929FBCB7C68568071E44FA9AA69CE773BADCD8A1BBD`.

## Re-evaluation with `best.pt`

| Split | Loss | Pixel Accuracy | mIoU | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.830859 | 0.843144 | 0.651173 | 0.830815 | 0.264408 | 0.103496 |
| Calibration | 0.857235 | 0.839791 | 0.684416 | 0.872916 | 0.273598 | 0.111259 |

Artifacts are kept locally under `outputs/segformer_b0_suim_ce/evaluation/`: metric JSON files, per-class CSVs, confusion matrices, training curves, and 30 validation visualizations (10 best / 10 middle / 10 worst per-image mIoU). These are generated outputs and are intentionally not versioned with checkpoints.

## Data-quality audit

- All 1,635 processed masks contain only class ids 0--7.
- RGB transition-color quantization affected 10,095,731 / 597,454,900 original mask pixels (1.6898% overall). Median per-image ratio is 0, p95 is 3.7811%, and maximum is 84.9795%; the top 20 are recorded for manual review.
- 37 masks were converted by explicit nearest-neighbor resizing of the class-index mask because the raw mask height was 55 pixels larger. They occur in train 28, val 3, calibration 6, test 0. The raw files remain unchanged. This repair remains an open sensitivity-analysis item (top crop vs bottom crop vs resize / exclusion).
- Cross-split scan found 9 byte-identical RGB images: 1 train--val, 4 train--calibration, and 4 train--official-test. The pHash montage visually confirms identical or near-identical content for the closest candidates. Therefore the present random split is valid only as an engineering baseline, not as a leakage-safe paper split. A grouped re-split is required before official paper results and all formal comparisons.

## Interpretation boundary

This report freezes the clean-model engineering baseline only. It does not claim a clean, paper-ready generalization score because the cross-split duplicate leakage has now been demonstrated. Do not use the official TEST score for model choice or calibration design.
