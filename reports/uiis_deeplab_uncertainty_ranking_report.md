# UIIS DeepLabV3-MobileNetV3 Error-Ranking Benchmark

## Protocol

- Evaluation data: UIIS confirmation, 511 original-image clusters under all 13
  frozen image conditions.
- Scores: raw MSP, clean-global calibrated MSP, entropy, probability margin,
  logit margin, energy, and local disagreement.
- Regions: full image, GT boundary (radius 3), and interior. Boundary masks are
  evaluation strata only and are not model inputs.
- Statistics: 1,000 paired cluster-bootstrap replicates, with each `sample_id`
  retaining all 13 conditions.
- Official SUIM TEST evaluated: **false**. Model retrained: **false**.

## Acceptance

- Aggregate rows: `273 = 13 conditions × 7 scores × 3 regions`.
- Per-image rows: `139,503 = 511 images × 13 conditions × 7 scores × 3 regions`.
- Paired bootstrap rows: `63 = 3 regions × 7 scores × 3 metrics`.
- All aggregate and bootstrap values are finite; all expected keys are unique.
- The fixed ranking test suite passed (`12 passed`).

Twenty-eight per-image region-score entries have no error pixels, so error AUPRC
and top-fraction error recall are mathematically undefined; seven additional
entries have a single-class error label and therefore undefined error AUROC.
These are preserved as missing only in the per-image diagnostic table and are
excluded from the corresponding cluster-level mean, rather than being assigned an
arbitrary score. They do not occur in aggregate or bootstrap outputs.

## Confirmation mean over 13 conditions

| Score | eAURC ↓ | Error AUPRC ↑ | Top-10% error recall ↑ |
| --- | ---: | ---: | ---: |
| Raw MSP | 0.153199 | 0.501528 | 0.194403 |
| Calibrated MSP | 0.153024 | 0.502594 | 0.195486 |
| Entropy | 0.159903 | 0.469418 | 0.175967 |
| Probability margin | 0.153821 | 0.491022 | 0.190485 |
| Logit margin | 0.155222 | 0.485755 | 0.187650 |
| Energy | 0.174502 | 0.431413 | 0.162034 |
| Local disagreement | 0.248097 | 0.337863 | 0.122044 |

## Primary comparison: raw MSP versus calibrated MSP

Positive improvement denotes that raw MSP is better. On the full image, raw MSP
has higher error AUPRC (`+0.004846`, 95% CI `[+0.003494, +0.006312]`) and
top-10% error recall (`+0.004136`, 95% CI `[+0.002955, +0.005435]`), while its
eAURC difference is not stably positive (`+0.000362`, 95% CI
`[-0.000142, +0.000812]`). Interior results show the same direction for error
AUPRC (`+0.008750`, CI `[+0.006776, +0.010935]`) and top-10% recall
(`+0.005935`, CI `[+0.004471, +0.007547]`).

Thus, although temperature scaling improved probability calibration in the
preceding benchmark, it did not provide a stable error-ranking gain for this
UIIS DeepLab model. This is a fixed external benchmark extension, not a new
blind confirmation.
