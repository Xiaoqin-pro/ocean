# SADR: Parameter-Efficient Task-Supervised Residual Adaptation for Robust Underwater Segmentation with a Frozen Expert

## Abstract

Underwater image degradations can reduce the reliability of a deployed
segmentation model, while updating the complete model may be undesirable when
the expert is already validated, expensive to retrain, or shared across
platforms. We study whether degradation robustness can instead be acquired by
learning a small input-side correction while leaving the segmentation expert
frozen. We propose SADR, a zero-initialized task-supervised residual adapter
trained from clean/degraded public UIIS pairs. On a locked 13-condition UIIS
protocol, SADR improves mean mIoU over the frozen UIIS-F4 expert by 0.607,
0.760, and 0.872 percentage points across three seeds (0.746 pp mean), while
updating only 11,012 parameters, or 0.296% of the expert. A parameter-matched
rank-2 q/v LoRA control reaches +0.226 pp, and head-only and last-block-only
fine-tuning reach +0.192 and +0.312 pp, respectively; unconstrained full
fine-tuning is stronger at +2.281 pp. Thus SADR is a constrained-adaptation
method rather than an accuracy substitute for full fine-tuning. Post-freeze
composition diagnostics show measurable but non-exclusive and
family-selective transfer to unseen degradation pairs. The untouched SUIM
check changes by -0.178 pp on average, and blur and held-out composition
failures remain explicit limitations. The results position task-supervised
input residual placement as a practical low-update robustness option, with
composition analysis used as a mechanism diagnostic rather than an exclusive
capability claim.

## 1. Introduction

Underwater perception systems are exposed to color attenuation, turbidity,
low illumination, and blur. A segmentation expert trained on one distribution
can therefore fail when the visual conditions change, even when its semantic
capacity is adequate. A natural response is to retrain or fine-tune the whole
network. That response is not always operationally acceptable: the expert may
already be validated, its weights may be shared by multiple deployments, or
the available adaptation data may describe only a narrow degradation family.

This paper asks a constrained question: can a pretrained underwater
segmentation expert acquire useful degradation robustness without updating
the expert itself? The question is deliberately different from maximizing
in-distribution mIoU. It concerns the operating point obtained when the
update budget is extremely small and the deployed expert must remain intact.

We introduce Semantic-Aware Degradation Restoration (SADR), a residual image
front-end placed before a frozen UIIS-F4 expert. The front-end is initialized
to the identity through a zero-initialized final projection and is optimized
with segmentation supervision from paired clean and synthetic degraded views,
plus restrained reconstruction, identity, distillation, and residual terms.
The semantic loss is applied after the frozen expert, so the correction is
task-aligned rather than defined by visual fidelity alone.

The central result is not that SADR is the highest-accuracy adaptation. Full
fine-tuning is clearly stronger in our controlled comparison. Instead, SADR
occupies a distinct low-update operating point: it improves the target
robustness protocol more than the tested decoder-head, last-block, and
parameter-matched LoRA controls while changing only 0.296% of the expert.
We further analyze whether corrections learned from single degradations have
structure under unseen compositions. The transfer is measurable, but it also
appears for partial and full fine-tuning and fails on some held-out families.
We therefore treat composition as a mechanism and boundary analysis, not as a
SADR-exclusive ability.

Our contributions are threefold:

1. We formulate frozen-expert, task-supervised residual adaptation for
   underwater segmentation with a zero-initialized input correction and a
   small trainable footprint.
2. We establish a locked three-seed robustness result and a controlled
   parameter-efficiency comparison spanning rank-2 LoRA, partial fine-tuning,
   and full fine-tuning.
3. We provide a transparent composition-transfer and boundary audit,
   including severity/family structure, shuffled nulls, SUIM source-domain
   cost, blur failures, and the limits of synthetic post-freeze diagnostics.

## 2. Related Work and Positioning

Task-driven underwater enhancement has previously connected enhancement with
downstream perception. Our distinction is operational: SADR keeps a validated
segmentation expert frozen and learns a small input-side residual under a
locked robustness protocol. Parameter-efficient adaptation and low-rank
adapters provide a relevant comparison, but the present study does not claim
that SADR is a general replacement for PEFT methods. Generic visual
enhancement is also not assumed to improve segmentation; the relevant metric
is the frozen expert's task output.

## 3. Method

Let (f) denote the frozen UIIS-F4 segmentation expert and (R_	heta) the
SADR front-end. For a degraded image (x_d), the deployed prediction is

\[
  \hat y = f(R_\theta(x_d)), \qquad
  R_\theta(x) = x + r_\theta(x).
\]

The last projection of (r_	heta) is zero initialized, so the initial
front-end is an identity mapping. Training uses a clean view (x_c), three
registered degraded views (x_{s1},x_{s2},x_{s3}), and the shared mask (y).
The main objective is

\[
\mathcal L = \mathcal L_{seg}
 + 0.10\mathcal L_{rec}
 + 0.10\mathcal L_{id}
 + 0.05\mathcal L_{KD}
 + 0.01\mathcal L_{res}.
\]

Here (mathcal L_{seg}) is the mean segmentation loss after (f) over the
three degraded views, (mathcal L_{rec}) encourages restored degraded views
to remain close to the clean view, (mathcal L_{id}) preserves clean inputs,
(mathcal L_{KD}) retains the frozen expert's clean predictions, and
(mathcal L_{res}) controls residual magnitude. The expert parameters are
never updated.

## 4. Experimental Protocol

Training uses only 2,371 UIIS images. Confirmation contains 511 images and is
evaluated once under clean plus twelve deterministic degradation conditions.
The official SUIM test contains 110 images and is used only as an external
source-domain check. The calibration split contains 508 images and is used
only for post-freeze audits; an earlier rejected selector had inspected this
role, so it is not treated as a pristine final test.

SADR uses its fixed 256-pixel/batch-8 configuration. The partial/full FT and
rank-2 q/v LoRA controls use a fixed 384-pixel/batch-4 configuration. They
share the training split, degradation exposure, source checkpoint,
eight-epoch budget, and locked evaluation roles, but not every optimization
hyperparameter. All comparisons were frozen before confirmation/test access.

## 5. Main Robustness Results

The frozen expert reaches 0.479872 mean mIoU on the 13-condition UIIS
confirmation role. SADR reaches 0.485939, 0.487471, and 0.488595 for three
seeds, corresponding to +0.607, +0.760, and +0.872 pp and a +0.746 pp mean.
The gain is family-selective: color and low-light conditions benefit most,
while blur-s3 remains a limitation. On the untouched SUIM role, the three
SADR seeds change by -0.130, -0.197, and -0.207 pp, so the method is not
presented as universal source-domain improvement.

## 6. Parameter-Efficient Adaptation

The matched comparison provides the correct context for the primary gain.
Rank-2 q/v LoRA updates 8,192 parameters and reaches +0.226 pp on
confirmation. Decoder-head-only and last-block-only fine-tuning update
396,808 and 800,000 parameters and reach +0.192 and +0.312 pp. Full
fine-tuning updates all 3,716,200 parameters and reaches +2.281 pp. SADR is
therefore not an accuracy winner; it is a low-update operating point that
outperforms the tested partial and matched low-rank controls on the target
robustness protocol. The accompanying Pareto figure should be labeled
“Target robustness–parameter trade-off on UIIS,” not general computational
efficiency.

## 7. Composition-Transfer Analysis

After freezing, we evaluate four severity-2 family pairs in both orders. On
the reused confirmation scenes, gains are +1.141 pp for SADR, +0.667 pp for
rank-2 LoRA, +0.881 pp for head-only, +1.693 pp for last-block-only, and
+5.807 pp for full FT. The 508-image calibration audit gives -0.270 pp for
SADR, +1.021 pp for head-only, +1.557 pp for last-block-only, and +3.690 pp
for full FT. These results show that composition transfer is measurable but
not exclusive to SADR and not universal.

In a broader SADR-only geometry screen, same-image additive correction exceeds
a shuffled-image null in all 54 seed-by-relation cases, while order and
additivity decline with severity. This supports structured correction
geometry, but cosine structure is not a sufficient predictor of mIoU gain.

## 8. Ablations and Negative Controls

The main ablation table should retain pixel-only, no-reconstruction,
no-distillation, GrayWorld, LAB-CLAHE, rank-2 LoRA, and the partial/full FT
controls. Frequency, routing, basis, selector, and explicit composition-loss
variants belong in the appendix as negative controls. The key distinction is
between task supervision and visual reconstruction: the pixel-only control
does not reproduce the full SADR composition result, while directly imposing
composition relations also harms the primary confirmation score.

## 9. Limitations

SADR does not match full fine-tuning and incurs a small SUIM source-domain
cost. Blur-s3 remains weak. Composition diagnostics reuse confirmation scenes
or rely on a calibration audit that is not pristine, and held-out composition
transfer is family-selective. The adapter adds 21.4% measured single-model
inference latency despite its small parameter count. UVMulti provides only a
negative/heterogeneous sanity audit, not real-world validation. Finally, the
study uses public synthetic degradation operators and should not be read as a
universal underwater robustness claim.

## 10. Conclusion

SADR shows that a frozen underwater segmentation expert can be made more
robust on a locked degradation protocol by learning a task-supervised,
zero-initialized input residual with only 0.296% trainable parameters. Full
fine-tuning remains the accuracy upper reference, while matched LoRA and
partial fine-tuning controls clarify that the result is not explained by
parameter count alone. Composition transfer and semantic correction geometry
are useful diagnostics of the constrained adaptation regime, with explicit
family and domain boundaries. The resulting contribution is a practical and
auditable low-update adaptation point, not a claim of universal or exclusive
compositional robustness.
