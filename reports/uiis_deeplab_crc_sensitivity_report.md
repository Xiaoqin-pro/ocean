# UIIS DeepLabV3-MobileNetV3 CRC Sensitivity Analysis

## Fixed protocol

- Calibration: 508 UIIS images under all 13 frozen conditions.
- Evaluation: 511 UIIS confirmation images under the same 13 conditions.
- Selector: raw MSP; coverage grid: 0.01 to 1.00 in 0.01 increments.
- Methods: Global CRC, Oracle condition CRC (diagnostic only), and 3-group
  frozen quality CRC with Global fallback.
- Quality seeds: `20260722`, `20260723`, `20260724`; no descriptors, KMeans
  settings, selector, fallback, or seed was altered.
- Targets: `alpha = 0.05, 0.10, 0.15`; 1,000 paired original-image cluster
  bootstrap replicates.
- Official SUIM TEST evaluated: **false**. Model retrained: **false**.

## Acceptance

- Condition rows: `195`.
- Bootstrap rows: `60`.
- Every actual metric and bootstrap interval is finite. The only null field is
  `seed` for Global/Oracle rows, where no quality clustering is used.
- Existing UIIS CRC and conformal-risk tests passed (`10 passed`).

## Result: quality-conditioned CRC remains unstable

Mean quality-group coverage changes versus Global CRC (averaged across the three
frozen seeds) are:

| Target α | Δ coverage | 95% CI | Interpretation |
| --- | ---: | --- | --- |
| 0.05 | +0.586pp | [+0.532pp, +0.641pp] | Small gain, but mean risk-excess reduction is negative |
| 0.10 | +0.766pp | [+0.522pp, +1.008pp] | Gain accompanies worse worst-condition risk excess |
| 0.15 | −0.472pp | [−0.836pp, −0.137pp] | Coverage is significantly worse |

At α=0.10, low-light-s3 risk excess improves, but the worst-condition risk
excess deteriorates by about 0.90pp (CI entirely below zero under the
``reduction`` convention). At α=0.15, blur-s3 risk excess also deteriorates.
Thus any gain is target- and condition-dependent rather than a reliable property
of image-quality grouping.

Oracle condition CRC has substantially more headroom (+2.00pp, +3.77pp, and
+1.38pp at α=0.05, 0.10, and 0.15), showing that the lack of stable quality
CRC gain is not due to a universally saturated coverage problem. It is evidence
that the frozen visual-quality proxy does not stably capture model difficulty.

This is a fixed external benchmark extension, not a renewed DARC method claim.
