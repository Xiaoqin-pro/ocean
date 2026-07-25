# AquaRiskMap pilot preregistration

**Status:** preregistered pilot protocol; no AquaRiskMap result exists yet.  
**Branch:** `experiment/aquariskmap-pilot`  
**Parent benchmark freeze:** `8d24fdf` (`experiment/uwr-benchmark-cnn-replication`)  
**Date:** 2026-07-25  

## Revision record

* **v1 / `0dd07c1`:** initial preregistration, committed before implementation.
* **v1.1:** resolves five protocol ambiguities
  identified in an independent review before the first AquaRiskMap code: Stage-B
  retraining versus zero-shot transfer, class-space-dependent logits, paired
  trajectory batches, supervision resolution, and the CRC monotone envelope.
  No pilot result was inspected to make these changes.

## 1. Purpose and falsifiable hypothesis

UWR-Bench is frozen before this protocol.  Its four completed dataset--model
units establish that probability calibration, error ranking, boundary risk, and
quality-proxy conditional selection are distinct questions.  In particular,
the DARC-Seg visual-quality-group CRC hypothesis is a negative control and
will not be revisited in this branch.

This pilot asks one narrower question:

> Can a lightweight, frozen-model error ranker trained with paired underwater
> degradation responses rank pixel errors materially better than raw maximum
> softmax probability (MSP), while a standard *global* CRC rule maintains the
> same target risk?

The proposed ranker is provisionally named **AquaRiskMap**.  It does not claim
to be the first semantic-segmentation error localizer.  Error Localization
Networks already learn error maps from an image and a segmentation prediction
([Kwon and Kwak, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Kwon_Semi-Supervised_Semantic_Segmentation_With_Error_Localization_Network_CVPR_2022_paper.pdf)).
The testable distinction here is fixed **same-scene, multi-degradation response
supervision** for a frozen underwater segmenter, evaluated separately on full,
boundary, and interior pixels under a leakage-audited protocol.

This work also does **not** claim per-image adaptive risk control.  The final
selection layer is a standard global CRC fit only on calibration data; adaptive
conditional CRC is already an established line of work
([Blot et al., 2025](https://proceedings.mlr.press/v258/blot25a.html)).

## 2. Frozen assets and data roles

### 2.1 Frozen SUIM protocol

All pilot data use `v2_scene_grouped_deduplicated` only:

| role | images | allowed use |
| --- | ---: | --- |
| formal train | 1,167 | create the risk-head train/development split only |
| validation | 146 | fixed pilot evaluation only |
| calibration | 146 | fit temperature/Global CRC only after the method is frozen |
| official TEST | 110 | **locked; no read, cache, prediction, or metric** |

The scene-grouped split, exclusions, and scene-leakage guarantees are those
recorded in `reports/suim_v2_scene_grouped_report.md`.  No sample is moved
between these roles in this branch.

### 2.2 Frozen base segmenters

The two base predictors remain fixed and are never fine-tuned by this pilot:

| base model | frozen checkpoint | source configuration |
| --- | --- | --- |
| SegFormer-B0 | `outputs/segformer_b0_suim_v2_scene/checkpoints/best.pt` | `configs/baseline/segformer_b0_suim_v2_scene.yaml` |
| DeepLabV3-MobileNetV3-Large | final frozen SUIM v2 checkpoint recorded in `reports/deeplabv3_suim_cnn_replication_report.md` | `configs/baseline/deeplabv3_mobilenetv3_suim_v2_scene.yaml` |

No official SUIM TEST output is permitted in a cache, a loss, a threshold fit,
or any diagnostic.  Existing UWR-Bench outputs remain immutable baselines.

### 2.3 Risk-head split and its evidence level

Before any risk-head cache is generated, formal-train **scene groups** will be
assigned with `GroupShuffleSplit(random_state=20260725)` to:

* `risk_head_train`: 80% of formal-train scene groups;
* `risk_head_development`: 20% of formal-train scene groups.

The split script will emit the exact CSV files, SHA-256 values, class-pixel
counts, scene-group overlap check, and deterministic test.  Class labels,
degradation labels, validation outputs, calibration outputs, and TEST outputs
may not influence this assignment.

Because each frozen base segmenter was originally trained on the full formal
train partition, this inner split cannot make base-predictor logits genuinely
out-of-sample.  Therefore it is used only to detect risk-head optimization
failure and to prevent direct sample reuse while training the head.  It is
**not** presented as independent generalization evidence.  The fixed SUIM
validation split is the pilot evaluation set; a fresh external confirmation is
required before any method claim is made.

## 3. Inputs, output, and implementation budget

For one image, AquaRiskMap receives features derived only from the image and
the frozen base segmenter prediction.  All inputs are resized to a fixed
`96 x 96` grid; the final one-channel score is bilinearly upsampled to the
label resolution for evaluation.

| feature block | channels | definition |
| --- | ---: | --- |
| normalized RGB | 3 | the model input image after the existing deterministic normalization |
| ranked probability profile | 4 | class-agnostic top-1 through top-4 sorted softmax probabilities |
| ranked probability gaps | 3 | top-1/top-2, top-2/top-3, and top-3/top-4 gaps |
| confidence statistics | 2 | normalized entropy and class-count-normalized log-sum-exp energy |
| local prediction features | 2 | fixed 3x3 label-disagreement fraction and predicted-label boundary indicator |

The total input dimensionality is **14 channels**.  In particular, raw
class-indexed logits are deliberately excluded: the input has no dependence on
the number or semantic identity of classes.  This permits a future zero-shot
cross-class-space diagnostic without redefining the network input.  No
ground-truth boundary, true degradation label, quality-cluster identifier, or
test-time image pair is an inference input.

The architecture is fixed as follows:

1. depthwise-separable `3x3` stem, `14 -> 64`, GroupNorm(8), GELU;
2. three residual depthwise-separable `3x3` blocks at 64 channels, with
   dilations 1, 2, and 3 respectively, each followed by GroupNorm(8) and GELU;
3. `1x1` projection `64 -> 1` producing an error-risk logit.

The implementation must assert fewer than **0.20M trainable parameters**.  It
may not contain an attention block, a pretrained feature extractor, or an
additional segmentation decoder.

## 4. Fixed paired-degradation supervision

For every risk-head image, use the already frozen deterministic degradation
registry:

`clean`, `color_s1..s3`, `turbidity_s1..s3`, `lowlight_s1..s3`, and
`blur_s1..s3`.

For each condition, derive the binary pixel target
`e = 1[predicted_class != ground_truth_class]`, excluding ignore pixels.  The
same-scene clean/degraded outputs are paired only during training.  At
inference, AquaRiskMap consumes one image and one frozen-model prediction.

Each training batch is exactly four base scenes, each represented by its clean
view and one degraded view (eight image-condition pairs).  The degraded view
for scene `s` in epoch `e` is selected without replacement by
`degradation_index = (stable_hash(s) + e) mod 12`, with the 12 non-clean
conditions in the fixed registry order above.  Thus every batch contains four
valid clean-to-degraded trajectories.  Batch ordering is a deterministic
shuffle with seed `20260725 + epoch`; a batch index is included in all
pair-sampling random states.

The loss is fixed before implementation:

```
L = L_error + 0.25 L_rank + 0.25 L_trajectory
```

* `L_error`: binary cross entropy on error targets.  Positive weight is the
  inverse class-frequency weight measured once on `risk_head_train`, clipped
  to `[1, 10]`; it is saved in the training metadata.
* `L_rank`: mean `softplus(r_correct - r_error)` over 2,048 deterministic
  error/correct pairs per batch.  Correct and error pixels are sampled with
  replacement only when their eligible pool is smaller than 2,048; the random
  state is `20260725 + 100003 * epoch + batch_index`.
* `L_trajectory`: for pixels correct on clean but wrong under a paired
  degradation, mean `relu(0.10 - (sigmoid(r_degraded) - sigmoid(r_clean)))`.
  It is zero for a batch without newly erroneous pixels.

For `L_error` only, pixels inside a ground-truth radius-3 boundary band receive
weight 2 and other valid pixels weight 1.  Ground-truth boundaries are never
an inference feature, a selector input, or a calibration input.

The head always produces a `96 x 96` risk-logit map, which is bilinearly
upsampled to the original `384 x 384` label grid *before* `L_error`, `L_rank`,
and `L_trajectory` are calculated.  Error labels and boundary labels are never
downsampled for supervision.  If this causes an OOM, only the number of base
scenes per batch may be reduced; the training-loss resolution remains fixed and
the reduction is recorded.

## 5. Fixed optimization

Each base model receives one independently initialized risk head.

| setting | fixed value |
| --- | --- |
| seed | 20260725 |
| optimizer | AdamW |
| learning rate | 1e-3 |
| weight decay | 1e-4 |
| epochs | 20 |
| batch size | 4 base scenes × (clean + one paired degradation) = 8 image-condition pairs; base-scene count may be reduced only for OOM and recorded |
| AMP | enabled |
| checkpoint selection | final epoch only; no early stopping |
| cache precision | frozen base logits may be `float16`; all head losses use `float32` |

The architecture, loss coefficients, resolution, optimizer, and epoch count
cannot be changed after seeing SUIM validation metrics.  A technical failure
(missing cache, non-finite loss, shape mismatch, or OOM) may be repaired only
without changing any listed statistical choice, and must be documented.

## 6. Baselines and fixed evaluation

The principal comparator is raw MSP.  The already frozen secondary scores are
also reported without selecting a new best baseline: calibrated MSP, entropy,
probability margin, logit margin, energy, and local disagreement.

Pilot evaluation uses all 13 conditions of the frozen **SUIM validation**
partition.  Metrics are evaluated for full, GT-boundary radius-3, and interior
pixels:

* eAURC;
* error AUPRC;
* top-10% uncertainty error recall;
* error AUROC (secondary);
* mean score on wrong and correct pixels (diagnostic).

All confidence-score comparisons use 1,000 paired scene-cluster bootstrap
replicates, with the original `sample_id` and all its 13 conditions retained as
one cluster.  Ties must be handled deterministically.  No result is selected
by condition, severity, class, or random seed.

After the architecture and score are frozen, a **Global CRC** is fit on all 13
conditions of the frozen SUIM calibration partition at `alpha = 0.10`.  It
uses the AquaRiskMap score only as a global pixel ranking score.  The same
global procedure is fit for raw MSP and all fixed secondary baselines.

The certification implementation is fixed to the existing UWR-Bench rule.  For
each original image and condition, it evaluates tie-aware selective risk on the
coverage grid `c ∈ {0.01, 0.02, ..., 1.00}` and replaces the raw curve `R(c)` by
the conservative monotone envelope

```
R_tilde(c) = max_{c' <= c} R(c').
```

Calibration chooses coverage only from this envelope; it never exploits a
non-monotone dip in raw empirical risk.  The validation set is then evaluated
once for coverage, selective risk, risk excess, foreground/background coverage,
boundary coverage, and macro-class coverage.  No per-image threshold, quality
group, KMeans model, condition identifier, or adaptive alpha is permitted.

## 7. Predefined progression gates

The following are decision gates, not objectives to tune against.

### Stage A -- SUIM low-cost pilot

Relative to raw MSP, a backbone has a **complete pass** only if the 13-condition
validation aggregate satisfies all of the following:

1. full-pixel eAURC decreases by at least 10%;
2. boundary-pixel eAURC decreases by at least 10%;
3. error AUPRC increases by at least 3 percentage points;
4. top-10% error recall increases by at least 5 percentage points;
5. at `alpha=0.10`, Global CRC coverage increases by at least 3 percentage
   points while selective risk is no greater than raw-MSP Global CRC;
6. the head has fewer than 0.20M parameters and median added inference time
   after 50 warm-up images is below 10% of frozen segmentation inference time.

Both SUIM backbones must have a positive direction for the two full/boundary
eAURC comparisons; at least one must satisfy the complete-pass gate.  If this
does not occur, AquaRiskMap stops permanently after Stage A.  UIIS and DUT-USEG
may not be used to modify, rescue, or retune it.

A Stage-A pass authorizes only the fixed follow-on evaluations.  It is not
final method evidence because the base segmenters saw the original formal-train
images used to fit the risk head; final method claims require the pre-registered
external confirmation in Stage C.

### Stage B -- fixed 2x2 extension

Only after a Stage-A pass, run the **same architecture, class-agnostic inputs,
losses, schedule, score definition, and evaluation protocol**, but train a
separate risk head on each dataset's own formal train partition.  Thus the
primary Stage-B experiment tests method reproducibility, not zero-shot weight
transfer.  On UIIS, use only UIIS train to fit the head, UIIS calibration for
Global CRC, and UIIS confirmation for evaluation.

The SUIM-trained head applied to UIIS without retraining is a frozen secondary
zero-shot diagnostic.  It is reported regardless of direction but is neither a
success gate nor a hyperparameter-selection source.  The four-unit primary
extension must show positive error-AUPRC direction in all four, non-worse
boundary ranking in all four, and a statistically supported eAURC gain in at
least three of four.  UIIS remains a fixed external benchmark extension, not a
fresh blind confirmation, because its confirmation partition was previously
used by UWR-Bench.

### Stage C -- fresh external confirmation

Only after Stage B is frozen, acquire and pre-register a new external dataset
before evaluating it.  DUT-USEG is a candidate because it offers real underwater
semantic/instance annotations, but its class space differs from SUIM/UIIS
([Li et al., 2021](https://arxiv.org/abs/2108.11727)).  It can therefore confirm
error-localization and ranking transfer, including the fixed zero-shot
class-agnostic risk-head diagnostic, not direct cross-dataset mIoU equivalence.
No DUT-USEG result may alter this method.

## 8. Non-negotiable prohibitions

This branch may not:

* access, cache, inspect, or evaluate SUIM official TEST;
* alter any UWR-Bench frozen result or DARC-Seg setting;
* train a third segmentation backbone or fine-tune either frozen base model;
* choose features, losses, conditions, thresholds, or seeds from validation
  results;
* use a GT boundary at inference;
* use a degradation label, scene ID, quality group, or image pair at inference;
* claim adaptive/conditional CRC or a new general error-localization task.

## 9. Required artifacts before any pilot result is trusted

1. group-aware risk-head split CSVs and integrity test;
2. configuration file encoding every fixed choice above;
3. head parameter-count test and forbidden-TEST-path test;
4. deterministic cache test covering all 13 conditions;
5. loss/trajectory unit tests, including no-new-error batches;
6. ranking/CRC invariant tests and atomic result-write test;
7. one immutable pilot report documenting all gates, including failure.

## 10. Related-work boundary to retain in the final manuscript

| work | overlap | fixed AquaRiskMap distinction |
| --- | --- | --- |
| Error Localization Network (CVPR 2022) | pixel error localization from image and segmentation prediction | same-scene underwater degradation-response supervision, leakage-audited reliability evaluation, and no claim of first error localization |
| Automatically Adaptive CRC (2025) | conditional/adaptive risk control | only one global CRC threshold; no input-adaptive risk budget or condition-specific threshold |
| Soft Dice Confidence (2026) | selective semantic-segmentation confidence | multi-class pixel risk ranking and local accepted regions, not binary image-level Dice confidence ([link](https://link.springer.com/article/10.1007/s10994-026-07096-w)) |
| model-agnostic conformal semantic-segmentation work (2026) | uncertainty and conformal guarantees for segmentation | Global CRC is an evaluation/certification layer, not claimed as the new method ([link](https://drops.dagstuhl.de/entities/document/10.4230/OASIcs.AEiC.2026.1)) |

Until these artifacts are committed, this document authorizes **protocol
implementation only**, not a pilot training run.
