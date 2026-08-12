# Parameter-efficiency baseline report

## Protocol

This comparison uses the same UIIS train-only protocol for all trainable
baselines: 2,371 training images, the four views (clean, s1, s2, s3), 384
pixel crops, batch size 4, AdamW, and eight epochs. The source UIIS-F4 expert
has 3,716,200 parameters. The full-F4 checkpoint was already available from
the matching protocol; head-only and last-block-only were trained from the
same frozen source checkpoint. No confirmation or SUIM labels were used for
optimization.

The evaluation uses all 13 locked conditions on the 511-image UIIS
confirmation role and, separately, the 110-image official SUIM test role.

## Results

| model | trainable parameters | expert fraction | UIIS confirmation mIoU | gain vs frozen | SUIM mIoU | change vs frozen |
|---|---:|---:|---:|---:|---:|---:|
| Frozen F4 | 0 | 0.000% | 0.479872 | 0.000 pp | 0.413955 | 0.000 pp |
| SADR | 11,012 | 0.296% | 0.485939 | +0.607 pp | 0.412657 | -0.130 pp |
| Head-only FT | 396,808 | 10.676% | 0.481789 | +0.192 pp | 0.417765 | +0.381 pp |
| Last-block-only FT | 800,000 | 21.527% | 0.482990 | +0.312 pp | 0.419533 | +0.558 pp |
| Full FT | 3,716,200 | 100.000% | 0.502678 | +2.281 pp | 0.445184 | +3.123 pp |

The result does not support a claim that SADR reaches full fine-tuning
performance. Full fine-tuning is clearly stronger on both roles, at the cost
of updating the entire expert. The useful finding is narrower and safer:
under the locked UIIS protocol, a 0.296% residual adapter produces a larger
confirmation gain than either a 10.7% decoder-head update or a 21.5% final
block update. Its source-domain cost is small, but unlike the partial and full
fine-tuning baselines it does not improve the untouched SUIM test.

## Interpretation for the paper

Parameter efficiency should appear as a supporting result, not as the sole
innovation. It establishes the correct reference frame for the 0.7 pp gain
and prevents an unfair comparison against only a frozen model. The central
claim remains the task-supervised semantic correction and its
severity-dependent, family-selective transfer to unseen synthetic
compositions. The Pareto result should be reported together with the
negative SUIM transfer, not used to imply universal superiority.
