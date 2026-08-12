# Parameter-efficiency baseline report

## Protocol

All trainable baselines use the same UIIS train-only split (2,371 images),
the same degradation registry and four views (clean, s1, s2, s3), the same
eight-epoch adaptation budget, source UIIS-F4 checkpoint, and locked
13-condition evaluation protocol. Their method-specific optimization
settings are retained: SADR uses its fixed 256-pixel/batch-8 residual-
adaptation configuration, while head-only, last-block-only, and full FT share
the fixed 384-pixel/batch-4 fine-tuning configuration. Thus this is a matched
data/exposure/budget comparison, not a claim that every optimizer,
resolution, or batch setting is identical. The source UIIS-F4 expert has
3,716,200 parameters. No confirmation or SUIM labels were used for
optimization.

The evaluation uses all 13 locked conditions on the 511-image UIIS
confirmation role and, separately, the 110-image official SUIM test role.

## Results

| model | trainable parameters | expert fraction | UIIS confirmation mIoU | gain vs frozen | SUIM mIoU | change vs frozen |
|---|---:|---:|---:|---:|---:|---:|
| Frozen F4 | 0 | 0.000% | 0.479872 | 0.000 pp | 0.413955 | 0.000 pp |
| SADR | 11,012 | 0.296% | 0.485939 | +0.607 pp | 0.412657 | -0.130 pp |
| Rank-2 LoRA (q/v) | 8,192 | 0.220% | 0.482136 | +0.226 pp | 0.418042 | +0.409 pp |
| Head-only FT | 396,808 | 10.676% | 0.481789 | +0.192 pp | 0.417765 | +0.381 pp |
| Last-block-only FT | 800,000 | 21.527% | 0.482990 | +0.312 pp | 0.419533 | +0.558 pp |
| Full FT | 3,716,200 | 100.000% | 0.502678 | +2.281 pp | 0.445184 | +3.123 pp |

The result does not support a claim that SADR reaches full fine-tuning
performance. Full fine-tuning is clearly stronger on both roles, at the cost
of updating the entire expert. The useful finding is narrower and safer:
under the locked UIIS protocol, a 0.296% residual adapter produces a larger
confirmation gain than either a 10.7% decoder-head update, a 21.5% final
block update, or a parameter-matched rank-2 q/v LoRA adapter (0.220%,
+0.226 pp). Its source-domain cost is small, but unlike the partial and full
fine-tuning baselines or LoRA it does not improve the untouched SUIM test.

## Ordered-composition diagnostic

The same five frozen checkpoints were evaluated after freezing on four
unseen severity-2 family pairs and both application orders. This is a
post-freeze diagnostic: the confirmation result reuses 511 scenes and the
calibration result is an audit rather than a pristine final test.

| model | confirmation composition gain | positive / 8 cases | calibration audit gain | positive / 8 cases |
|---|---:|---:|---:|---:|
| SADR | +1.141 pp | 8/8 | -0.270 pp | 4/8 |
| Rank-2 LoRA (q/v) | +0.667 pp | 8/8 | not run | -- |
| Head-only FT | +0.881 pp | 6/8 | +1.021 pp | 8/8 |
| Last-block-only FT | +1.693 pp | 6/8 | +1.557 pp | 8/8 |
| Full FT | +5.807 pp | 8/8 | +3.690 pp | 8/8 |

Positive composition transfer is not unique to SADR; rank-2 LoRA, full and
partial fine-tuning also transfer, and full FT is strongest. The defensible
statement is that SADR retains measurable, family-selective transfer while
leaving the expert untouched and updating only 0.296% of its parameters. On
the held-out audit, SADR is not universal: lowlight+blur and turbidity+blur
are negative.

## Interpretation for the paper

Parameter efficiency should appear as a supporting result, not as the sole
innovation. It establishes the correct reference frame for the 0.7 pp gain
and prevents an unfair comparison against only a frozen model. The central
claim should be framed as constrained adaptation with an empirically
structured, but non-exclusive, composition-transfer profile. The matched
LoRA result further shows that the advantage is not explained by parameter
count alone. The Pareto result should be reported together with the negative SUIM transfer and the
full/partial-FT composition controls, not used to imply universal superiority
or SADR-specific compositionality.
