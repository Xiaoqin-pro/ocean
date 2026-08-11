# SADR semantic compositionality study

## Research question

The main SADR result is not only a mean-mIoU change.  It raises a more
specific question: does a correction learned from one degradation family have
structure that can transfer to an unseen composition of families?

We define the frozen-expert semantic correction for an image (x) as

\[
\Delta(x)=f(R_\theta(x))-f(x),
\]

where (f) is the frozen UIIS-F4 expert and (R_\theta) is SADR.  On the
confirmation images (post-freeze diagnostic only), we compare the correction
for two ordered operators with the corresponding atomic corrections.  No
checkpoint, threshold, or parameter is selected using these images.

## Three-seed screen

The table reports mean cosine similarity over all 511 confirmation images;
the parenthesized value is the sample standard deviation over the three SADR
seeds (20260811, 20260812, 20260813).

| relation in frozen-expert logit space | cosine | relative error |
|---|---:|---:|
| binary additivity, \(\Delta_{AB}\) vs \(\Delta_A+\Delta_B\) | 0.785 (0.011) | 0.389 |
| order invariance, \(\Delta_{AB}\) vs \(\Delta_{BA}\) | **0.975 (0.001)** | 0.114 |
| self-composition, \(\Delta_{AA}\) vs \(2\Delta_A\) | 0.765 (0.014) | 0.424 |
| triple closure, \(\Delta_{ABC}\) vs \(\Delta_A+\Delta_B+\Delta_C\) | 0.535 (0.026) | 0.560 |

The stable hierarchy is the result: semantic corrections are strongly
order-consistent, moderately additive for a binary composition, and only
partially closed under a three-way sum.  This argues against claiming a
globally linear correction law.  It supports the narrower scientific claim
that a frozen task expert induces an order-stable semantic correction field
that can extrapolate beyond the single-degradation training views.

For comparison, the image residual itself is more pixel-additive (mean
cosines 0.937 for binary addition and 0.697 for the triple relation).  The
semantic space therefore exposes a different structure from raw pixel
reconstruction.  On the seed-20260811 matched control, SADR's semantic
binary/triple cosines were 0.773/0.507, versus 0.722/0.422 for the
pixel-only front-end.  This is a mechanism control, not an independent test
set.

## Unseen-composition transfer

The frozen checkpoints were evaluated on eight ordered compositions of the
registered operators: color+turbidity, lowlight+blur, color+lowlight, and
turbidity+blur, each in both orders.  These images reuse the 511 confirmation
scenes, so this is a stress test rather than an independent generalization
split.

| seed | mean gain over eight compositions (pp) |
|---:|---:|
| 20260811 | +1.241 |
| 20260812 | +1.264 |
| 20260813 | +1.401 |
| **mean** | **+1.302** |

All 24 seed-by-composition values are positive.  The corresponding main
13-condition confirmation gain is +0.746 pp.  The larger stress-test gain is
consistent with the lower frozen-expert baseline on compounded degradations;
it is not evidence that the confirmation split is independent.

The pixel-only seed-20260811 control reaches only +0.486 pp on the same eight
compositions, leaving a +0.755 pp gap to full SADR.  The gap is largest on the
lowlight/blur pair, where pixel reconstruction alone is nearly ineffective.

## Direct regularizer controls (negative evidence)

We also tested whether the observed structure could simply be imposed as a
training loss.  Each variant used the UIIS train split only and was evaluated
with the locked confirmation protocol.

| candidate | confirmation gain (pp) | decision |
|---|---:|---|
| additive semantic-logit constraint, sparse trigger | +0.363 | reject; harms primary task |
| order-consistent semantic-logit constraint | +0.359 | reject; no recovery of SADR gain |
| order-consistent semantic-probability KL (correctly normalized) | +0.356 | reject; no recovery of SADR gain |
| unlabeled composite semantic distillation | +0.356 | reject; no recovery of SADR gain |

The additive constraint also produced +0.763 pp on the eight compositions,
but this is below the ordinary SADR stress-test gain (+1.241 pp) and came with
the primary-task drop.  The result is useful scientifically: the observed
semantic structure is an emergent property of task-supervised correction, not
a relation that can be safely enforced with a generic penalty.  We therefore
do **not** include any of these regularizers in the claimed method.

## Safe paper contribution

The defensible contribution is a mechanism-level finding rather than a claim
of exact linear composition:

> A zero-initialized, task-supervised residual front-end before a frozen
> underwater segmentation expert learns semantic corrections that remain
> strongly invariant to degradation order and transfer positively to unseen
> ordered compositions, whereas pixel-only correction and direct
> compositionality penalties do not reproduce the same behavior.

The paper must call this a post-freeze compositional stress test, report the
same-scene limitation, retain the negative regularizer controls, and avoid
claiming real-world UVMulti or universal degradation robustness.

