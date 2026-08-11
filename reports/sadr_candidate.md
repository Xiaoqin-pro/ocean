# SADR candidate result

## Method

Semantic-Aware Degradation Restoration (SADR) keeps the trained UIIS-F4
segmentation expert frozen and inserts a zero-initialized residual image
front-end before it.  The front-end is trained only on the public UIIS train
split.  Each training sample supplies a clean image and three fixed synthetic
degradations from the preregistered 13-condition registry.  The objective is

\[
 L = L_{seg}(f(R(x_d)), y) + 0.10\|R(x_d)-x_c\|_1
     + 0.10\|R(x_c)-x_c\|_1 + 0.05L_{KD} + 0.01L_{res}.
\]

The inference path is a single residual rectification followed by the frozen
expert; no confirmation labels or SUIM test data are used during training.

## Frozen protocol

| role | split | images | use |
|---|---:|---:|---|
| train | UIIS train | 2,371 | optimization only |
| calibration | UIIS calibration | 508 | not used by main SADR; used only for the failed quality-gate diagnostic |
| confirmation | UIIS confirmation | 511 | one-time final evaluation |
| external | SUIM official test | 110 | one-time external check |

All reported means average the same 13 fixed conditions: clean, three color,
three turbidity, three low-light, and three blur severities.

## Main result

| model | UIIS confirmation mean mIoU | change |
|---|---:|---:|
| UIIS-F4 | 0.479872 | — |
| SADR-4 | 0.483846 | +0.397 pp |
| **SADR-8** | **0.485939** | **+0.607 pp** |

SADR-8 improves clean, color, turbidity, and low-light conditions.  Blur-s3
remains a weakness (0.3948 versus 0.4001 for UIIS-F4).

On the untouched SUIM official test, UIIS-F4 is 0.413955 and SADR-8 is
0.412657 (−0.130 pp).  This is a small source-domain cost, not the catastrophic
forgetting seen in the joint-domain and routing routes.

## Ablations on the same confirmation split

| variant | change from SADR-8 | mean mIoU | gain |
|---|---|---:|---:|
| pixel-only | segmentation weight 0, reconstruction only | 0.482171 | +0.230 pp |
| no-reconstruction | reconstruction weight 0 | 0.485510 | +0.564 pp |
| no-distillation | distillation weight 0 | 0.485869 | +0.600 pp |
| SADR-8 | full objective | 0.485939 | +0.607 pp |

The decisive comparison is pixel-only versus task-supervised SADR: image
reconstruction alone gives only a small gain, while optimizing the frozen
segmenter’s semantic loss produces the robust improvement.  The no-distillation
result shows that the semantic task loss is the essential component; the
distillation term is optional stabilization rather than the claimed novelty.

## Current scientific status

SADR-8 is the first route in this repository that passes the preregistered
\(+0.50\) pp confirmation gate and retains the external SUIM score within
0.13 pp.  It is therefore a defensible paper candidate, but not yet a complete
submission: the contribution should be framed as a task-aware residual
restoration front-end, and the paper still needs qualitative figures,
parameter/FLOP accounting, a second random seed, and a comparison with at
least one conventional enhancement baseline.

The failed routes remain recorded separately and should be used as negative
evidence rather than omitted.
