# SADR candidate result

## Method

Semantic-Aware Degradation Restoration (SADR) keeps the trained UIIS-F4
segmentation expert frozen and inserts a zero-initialized residual image
front-end before it. The front-end is trained only on the public UIIS train
split. Each training sample supplies a clean image and three fixed synthetic
degradations from the preregistered 13-condition registry. The objective is

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
| UIIS-F4 | 0.479872 | baseline |
| SADR-4 | 0.483846 | +0.397 pp |
| **SADR-8, seed 20260811** | **0.485939** | **+0.607 pp** |
| **SADR-8, seed 20260812** | **0.487471** | **+0.760 pp** |
| **SADR-8, seed 20260813** | **0.488595** | **+0.872 pp** |

Across three seeds, the confirmation gain is +0.746 pp on average (sample
standard deviation 0.133 pp; 95% t interval [0.415, 1.078] pp; one-sample
t-test against zero, p=0.0105). This is a small-n stability indication, not a
large-sample significance claim.

The gain is not uniform across degradation families. Averaged over seeds and
the three severities, color attenuation improves by +1.56 pp, low-light by
+0.89 pp, turbidity by +0.62 pp, while blur is essentially unchanged (+0.03
pp). Blur-s3 remains the only consistently negative condition (about -0.43
pp).

On the untouched SUIM official test, UIIS-F4 is 0.413955. SADR-8 changes are
-0.130, -0.197, and -0.207 pp for the three seeds (mean -0.178 pp, sample
standard deviation 0.034 pp). This is a small source-domain cost, not the
catastrophic forgetting seen in the joint-domain and routing routes, but it
must be reported rather than hidden.

## Unseen-composition stress test

After the main checkpoint was frozen, we ran a diagnostic stress test on the
same 511 confirmation images but with eight novel compositions that were not
present in the training registry: color/turbidity, lowlight/blur,
color/lowlight, and turbidity/blur, each in both application orders. No
parameters or thresholds were changed for this test. The three SADR seeds
improved by +1.241, +1.264, and +1.401 pp respectively (mean +1.302 pp); all
24 seed-by-composition results were positive. This is evidence for
compositional extrapolation of the residual adapter, not a replacement for
the preregistered single-degradation confirmation result because the image
split is reused diagnostically.

| unseen composition family | mean gain over three seeds (pp) |
|---|---:|
| color + turbidity (both orders) | +1.830 |
| lowlight + blur (both orders) | +0.825 |
| color + lowlight (both orders) | +1.732 |
| turbidity + blur (both orders) | +0.821 |

As a mechanism control, the seed-20260811 pixel-only checkpoint reaches only
+0.486 pp on the same eight compositions, versus +1.241 pp for full SADR. The
gap is especially clear for lowlight/blur, where pixel-only gains are near
zero. This supports task supervision as the source of compositional transfer.

The broader all-family/severity screen evaluates all six family pairs, both
orders, and matched severities 1--3 on a 128-image confirmation subset per
seed. Across 54 seed-by-relation cases, semantic order cosine is 0.995, 0.952,
and 0.859 at severities 1, 2, and 3; semantic additivity is 0.877, 0.583,
and 0.367. The same-image additive relation beats a shuffled-image null in all
54 cases (mean cosine gap +0.234). Pixel AB/BA similarity is often high, so
the operator-commutativity control is reported alongside the semantic result.
These measurements support a severity-dependent, family-selective partial
compositionality claim rather than universal order stability.

An alignment audit paired each relation's semantic geometry with its actual
ordered-composite mIoU gain on the same 128-image screen. Additivity cosine had
only weak cross-relation correlations with gain (r = 0.327, 0.193, 0.320 for
the three seeds), while order cosine was essentially uncorrelated (r = 0.039,
0.038, -0.045). The shuffled-add null was not a reliable gain predictor
either. The geometry is therefore mechanistic evidence, not a sufficient
selection rule for predicting which family pair benefits.

An unfitted self-gating control also failed to provide a deployable rescue:
the best entropy-based selector improved the calibration audit by +0.254 pp
over always-on SADR, but fell -0.161 pp below SADR on confirmation. Margin and
confidence selectors showed the same split reversal. The final method remains
always-on SADR; the self-gate is recorded as negative evidence.

As a held-out scene audit, the eight ordered compositions were also evaluated
on all 508 calibration images. This split was not used by SADR optimization or
the primary confirmation threshold, but a previously rejected quality-gate
diagnostic had inspected it, so it is not presented as a pristine final test.
Seed gains were +0.189, +0.212, and +0.433 pp (mean +0.278 pp), with 15/24
positive seed-by-composition values. Color/turbidity and color/low-light
transfer, while low-light/blur averaged -0.590 pp and turbidity/blur was near
zero. The negative family result is retained as a limitation.

## Ablations on the same confirmation split

| variant | change from SADR-8 | mean mIoU | gain |
|---|---|---:|---:|
| pixel-only | segmentation weight 0, reconstruction only | 0.482171 | +0.230 pp |
| no-reconstruction | reconstruction weight 0 | 0.485510 | +0.564 pp |
| no-distillation | distillation weight 0 | 0.485869 | +0.600 pp |
| SADR-8 | full objective | 0.485939 | +0.607 pp |

The decisive comparison is pixel-only versus task-supervised SADR: image
reconstruction alone gives only a small gain, while optimizing the frozen
segmenter's semantic loss produces the robust improvement. The no-distillation
result shows that the semantic task loss is the essential component; the
distillation term is optional stabilization rather than the claimed novelty.

## Complexity and negative controls

The SADR front-end has 11,012 trainable parameters. The frozen UIIS-F4 expert
has 3,716,200 parameters, so the trainable overhead is 0.296% of the expert.
At 384x384 on the experiment laptop GPU, a 30-iteration CUDA-event benchmark
measured 11.14 ms for UIIS-F4 and 13.53 ms for UIIS-F4+SADR (+21.4% latency).
It adds one small convolutional image pass before the unchanged segmenter.

Conventional controls on the same confirmation protocol are weaker: GrayWorld
gives +0.221 pp on UIIS confirmation and -1.931 pp on SUIM official, while
LAB-CLAHE gives -1.596 pp on confirmation and +0.500 pp on SUIM official.
These controls are useful precisely because their direction changes by domain;
they do not provide the consistent multi-condition gain of SADR.

Two more expressive variants were tested without changing the frozen protocol.
A low/high-frequency split front-end reached only +0.269 pp on confirmation
and +0.064 pp on SUIM official. A four-expert degradation-family router
reached +0.345 pp after correcting the route-label sampler (the uncorrected
random-label version was +0.369 pp), with SUIM changes of -0.133 pp and
-0.157 pp respectively. The corrected router classified the synthetic family
only 31.4% accurately (25% chance), so it is retained as a negative control,
not folded into SADR.

A semantic-feature consistency term (cosine matching of the frozen expert's
last-stage global feature for clean and restored views) reached +0.399 pp in a
four-epoch screening run and +0.621 pp in a matched eight-epoch run, versus
+0.607 pp for the ordinary SADR run with the same seed. The +0.014 pp
difference is not a meaningful new contribution, so the feature term is kept
as a negative control rather than added to the main method.

A stricter semantic-view consistency term was also screened. It matches the
frozen expert's probability maps for the three severities of the same
synthetic degradation family. With the same seed and eight epochs it reached
+0.607 pp on confirmation, versus +0.607 pp for ordinary SADR (difference
0.0003 pp), and -0.138 pp on SUIM official. It is therefore a matched
negative control: the gain comes from task-supervised residual adaptation,
not from adding a generic consistency penalty.

An edge-preserving reconstruction term was screened specifically for the
blur-s3 weakness. It adds first-order gradient matching to the pixel
reconstruction objective. The same-seed confirmation gain was +0.612 pp
(ordinary SADR: +0.607 pp), while blur-s3 remained negative (-0.47 pp versus
-0.54 pp for ordinary SADR); the SUIM change was -0.117 pp. The 0.006 pp mean
difference is too small to justify a second claimed mechanism, so this variant
is also retained as a negative control.

The small UVMulti held-out sanity subset also did not improve: common-class
mIoU stayed around 0.322 on raw frames and 0.319 on enhanced frames for both
SADR seeds. This prevents claiming cross-dataset generalization from the
available partial UVMulti download.

The calibration-fitted quality gate was deliberately rejected: it reduced the
confirmation gain to +0.243 pp. The final candidate therefore uses no
confirmation-derived gate.

## Current scientific status

SADR-8 is the first route in this repository that passes the preregistered
+0.50 pp confirmation gate across three seeds. The stronger mechanism result
is the combined fixed-pair and all-family analysis: the corrections are highly
order-consistent for mild matched severities but degrade with severity, and
the correct same-image additive relation beats a shuffled null in all 54
screening cases. The confirmation stress test improves eight reused-scene
compositions by +1.302 pp, while the calibration audit is only +0.278 pp on
average. This gives the
paper a specific, falsifiable question about when task-supervised correction
fields transfer, rather than a universal compositionality claim or only a
parameter-count claim.

Direct additive, order-consistency, probability-KL, and unlabeled composite
distillation penalties all reduced the primary confirmation gain to about
+0.36 pp in matched seed-1 screens. They are retained as negative evidence,
not folded into SADR. The safe method claim is therefore the emergent,
severity-dependent and family-selective semantic correction of the frozen-
expert adapter, accompanied by the compositional stress test and its
operator/null controls; it is not an exact linear-composition law.

The contribution should still not be framed as the first task-driven
underwater enhancement method. Prior work already uses semantic or
downstream-task guidance. The novelty claim should be the frozen-expert,
zero-initialized adapter, the observed severity-dependent semantic
compositional-transfer mechanism with operator/null controls, and the fixed
multi-condition protocol. The paper still needs
qualitative figures, exact FLOP/latency accounting, and a comparison with at
least one conventional enhancement baseline beyond GrayWorld.

The failed routes remain recorded separately and should be used as negative
evidence rather than omitted.
