# UIIS SegFormer-B0 Reliability Benchmark Extension

## Status and provenance

- Dataset protocol: `UIIS_alpha010_confirmation_v1`.
- Model: frozen SegFormer-B0 checkpoint, SHA-256 `C2EAD86D324A02AD874C5DBDF88CC5238F9ABC0C9E673A1EE08D5CFE99DB1B52`.
- Evaluation split: 511 UIIS confirmation image clusters under 13 fixed conditions.
- Calibration split: 508 images; used only to fit temperature and CRC coverage parameters.
- SUIM official TEST evaluated: **false**.
- Model retrained: **false**.
- Evidence level: external fixed-protocol benchmark extension. UIIS confirmation had previously been opened for the DARC negative control; this is not a second blind confirmation.

## 1. Calibration

On clean UIIS confirmation, raw ECE was `0.120271`. Clean-global temperature scaling reduced it to `0.025261`; clean mIoU remained `0.518948` before and after scaling, as required by the argmax invariant.

## 2. Calibration and error ranking are separable

The primary ranking comparison was raw MSP versus clean-global calibrated MSP over all 13 conditions, using 1,000 image-cluster bootstrap draws. Raw MSP minus calibrated MSP eAURC improvement was `+0.000375` with 95% CI `[-0.000126, +0.000901]`. The interval crosses zero.

Thus, the substantial probability-calibration improvement does not establish a stable improvement in error ranking. This is consistent with treating calibration and ranking as separate reliability axes.

## 3. Boundary-dominated residual risk

The radius-3 boundary analysis produced 52 aggregate rows, 26,572 image-level rows, and 26 image-cluster bootstrap rows. Every condition and both raw and clean-global probability variants had a positive 95% confidence interval for boundary minus interior error rate.

- Mean boundary-minus-interior error-rate gap across the 13 conditions: `22.76pp`.
- Clean/raw gap: `22.68pp`, 95% CI `[21.29pp, 24.04pp]`.
- Temperature scaling leaves the prediction map unchanged, so this segmentation error gap is identical for raw and clean-global outputs.

Ground-truth boundaries were used only to stratify evaluation, never as model input.

## 4. Fixed quality-group CRC sensitivity

The analysis reused the frozen train-derived descriptors, three KMeans seeds, raw-MSP selector, global fallback, and 1,000 original-image-cluster bootstrap draws. It compared Global CRC, Oracle condition CRC (diagnostic only), and quality-group CRC with global fallback at the three fixed risk targets.

| Target alpha | Quality-group coverage change vs Global CRC | Interpretation |
| ---: | ---: | --- |
| 0.05 | `+0.584` to `+0.587pp` across seeds | Small gain; does not establish a general method benefit. |
| 0.10 | `-1.627` to `-2.142pp` across seeds; all 95% CIs negative | Stable degradation. |
| 0.15 | `-0.214` to `-0.222pp` across seeds; all 95% CIs negative | Stable degradation. |

At alpha `0.10`, the previously interesting operating point, quality grouping reduces rather than improves coverage. The fixed UIIS extension therefore supports the existing negative conclusion: visual quality grouping is not a stable proxy for segmentation difficulty under CRC.

## Evidence decision

| Question | Decision |
| --- | --- |
| Calibration improvement | PASS |
| Calibration-ranking separation | PASS |
| Boundary-dominated residual risk | PASS |
| Quality-group CRC instability | PASS |
| Official SUIM TEST accessed | FALSE |
| Model retrained | FALSE |
| New blind confirmation | FALSE |

## Files

- Temperature metrics: `experiments/uiis_segformer_temperature_scaling_metrics.csv`.
- Ranking outputs: `outputs/uiis_uncertainty_ranking/`.
- Boundary outputs: `outputs/uiis_boundary_analysis/`.
- CRC sensitivity outputs: `outputs/uiis_crc_sensitivity/`.
