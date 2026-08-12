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

## Three-seed fixed-pair screen

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
partially closed under a three-way sum. This argues against claiming a
globally linear correction law. It supports only the narrower scientific
claim that a frozen task expert can induce a partially compositional semantic
correction field that extrapolates beyond the single-degradation training
views.

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

## All-family/severity screening and null baseline

The fixed-pair screen above used one color/turbidity/low-light triplet. To test
whether it was an artefact of that choice, we evaluated all six unordered
family pairs at matched severities 1, 2, and 3 (both application orders), on a
128-image confirmation subset for each of the three frozen seeds. This is a
screening subset, not a replacement for the full 511-image primary result.

| matched severity | semantic order cosine | semantic add cosine | pixel AB/BA cosine | correct-add minus shuffled-add |
|---:|---:|---:|---:|---:|
| 1 | 0.995 | 0.877 | 0.999 | +0.315 |
| 2 | 0.952 | 0.583 | 0.991 | +0.243 |
| 3 | 0.859 | 0.367 | 0.953 | +0.143 |
| **all 54 relations** | **0.935** | **0.609** | **0.981** | **+0.234** |

The correct same-image additive relation exceeded a shuffled-image null in
all 54 seed-by-relation cases. At the same time, both order consistency and
additivity degrade as severity increases. The pixel operators are often close
to commutative, so order similarity alone is not sufficient evidence; the
turbidity-s3/low-light-s3 pair is a useful counterexample (pixel AB/BA cosine
about 0.75 while semantic order similarity is also substantially lower than
the mild-severity pairs). The supported statement is therefore
severity-dependent, family-selective partial compositionality, not universal
order invariance or a linear correction law.

We also aligned the semantic controls with the actual composite mIoU gains on
the same 128-image screen. Across the three seeds, additivity cosine correlated
only weakly with mean ordered-composite gain (Pearson r = 0.327, 0.193, and
0.320); order cosine was essentially uncorrelated (r = 0.039, 0.038, and
-0.045). The shuffled-add null was not a performance predictor either (r =
0.446, 0.338, and 0.439). Thus these geometries are evidence of structured
correction, but not a sufficient rule for deciding which family pair will
benefit. This negative alignment result rules out the overly simple story that
"more compositional" automatically means "larger mIoU gain."

## Held-out scene audit

We additionally ran the eight ordered stress-test compositions on all 508 UIIS
calibration images. These images were not used by the SADR optimizer or the
primary confirmation threshold, but an earlier rejected quality-gate
diagnostic had inspected this split; we therefore call it a held-out SADR
audit, not a pristine final generalization test.

The three seed gains were +0.189, +0.212, and +0.433 pp (mean +0.278 pp), with
15 of 24 seed-by-composition values positive. Transfer is family-selective:
color/turbidity averages +0.772 pp across orders, color/low-light +0.935 pp,
whereas low-light/blur averages -0.590 pp and turbidity/blur is approximately
zero. This audit prevents a universal unseen-composition claim and makes the
negative low-light/blur result an explicit limitation.

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

## Unfitted self-gating control (negative evidence)

To test whether the residual could be selected safely without labels, we also
implemented four prediction-only per-image selectors: lower output entropy,
higher top-1 margin, higher mean confidence, and their entropy/margin
conjunction. No gate parameters were fitted. On the 508-image calibration
audit, the best entropy selector reached 0.428872 mean mIoU versus 0.426343
for always-on SADR (+0.254 pp over SADR). On the locked 511-image confirmation
diagnostic, the same selector reached 0.484328 versus 0.485939 for SADR
(-0.161 pp). Margin and confidence behaved similarly. This split reversal
rejects prediction confidence as a deployment gate and is retained as a
negative control; the claimed method remains the fixed always-on adapter.

For completeness, a label-fitted quality gate trained only on the 2,371-image
UIIS train split was also audited. It reduced confirmation mean mIoU from
0.485939 for always-on SADR to 0.484370 (-0.157 pp). Its SUIM mean was 0.413257,
which is a smaller source-domain cost than SADR but does not compensate for
the primary robustness loss. This train-only selector is therefore rejected
as well; no learned or confidence-based gate is part of the method.

## Safe paper contribution

The matched parameter-efficiency control changes the scope of the mechanism
claim. Full and partial fine-tuning also transfer to ordered compositions,
and full FT is stronger. The defensible contribution is therefore a
constrained-adaptation finding rather than a SADR-exclusive mechanism or a
claim of exact linear composition:

> A zero-initialized, task-supervised residual front-end before a frozen
> underwater segmentation expert can retain measurable, family-selective
> transfer to selected unseen ordered compositions while updating 0.296% of
> the expert; the transfer is weaker or absent on some held-out families and
> is not exclusive to this front-end, since partial and full fine-tuning also
> transfer.

The paper must call this a post-freeze compositional stress test, report the
same-scene and held-out-audit limitations, retain the negative regularizer
controls, and avoid claiming real-world UVMulti or universal degradation
robustness.
