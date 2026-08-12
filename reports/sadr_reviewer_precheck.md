# Pre-submission reviewer precheck

## Highest-risk objections

1. **Novelty overlap.** Task-driven underwater enhancement already exists.
   Response: narrow the claim to a frozen-expert, zero-initialized,
   parameter-efficient robustness adapter and cite TFUIE/STSC/HSRUIE.
2. **Synthetic-to-real gap.** The locked 13-condition test is generated from
   deterministic transformations. Response: report the negative UVMulti sanity
   result and the SUIM source-domain cost; do not call it real-world validation.
3. **Magnitude.** +0.746 pp is modest. Response: show three seeds, family and
   severity error bars, +0.296% parameter overhead, and the pixel-only,
   frequency, routing, feature, GrayWorld, and CLAHE controls.
4. **Source-domain regression.** SUIM drops by -0.178 pp on average. Response:
   state it in the abstract/results and frame SADR as a robustness adapter, not
   a universally better segmenter.
5. **Latency.** Tiny parameter count still gives +21.4% measured latency at
   384x384. Response: report both parameter and latency costs and avoid
   “free”/“negligible runtime” language.
6. **Small-seed statistics.** n=3 is not a population estimate. Response: use
   the t-test only as supporting evidence, report raw seed values, and avoid
   claiming statistical significance across all underwater scenes.
7. **Baseline breadth.** GrayWorld and CLAHE are conventional controls, not
   state-of-the-art UIE methods. Response: add a reproducible public learned
   UIE baseline only if it can be run without changing the frozen protocol;
   otherwise state this limitation explicitly.
8. **Composition diagnostic reuse.** The unseen-composition stress test reuses
   the confirmation images. Response: label it post-freeze diagnostic evidence,
   report both operator orders, and never present it as an independent split;
   the primary claim remains the locked single-degradation result.
9. **Held-out composition weakness.** The calibration audit averages only
   +0.278 pp and lowlight/blur is negative across orders. Response: report the
   family-selective result and do not claim universal composition
   generalization.
10. **Operator commutativity.** A reviewer may argue that AB and BA images are
   nearly identical. Response: report pixel AB/BA cosines beside semantic
   order cosines; include the turbidity-s3/lowlight-s3 counterexample and the
   shuffled-image null.
11. **Mechanism attribution.** A reviewer may attribute the composition gain to
   generic image reconstruction. Response: show the pixel-only composition
   control (+0.486 pp versus full SADR +1.241 pp for the same seed).
12. **Overclaiming compositionality.** A reviewer may ask whether the method
    assumes a linear correction law. Response: report the measured hierarchy
    that matched-severity all-family order/additivity degrade from
    0.995/0.877 to 0.859/0.367, direct additive/order/KL/distillation penalties
    reduce the primary gain to about +0.36 pp, and describe transfer as an
    emergent, partial semantic property rather than an imposed equation.
13. **Mechanism sufficiency.** A reviewer may ask whether cosine similarity
    predicts which composition improves. Response: report the alignment audit;
    additivity/gain correlations are only 0.193--0.327 and order/gain is near
    zero, so geometry is presented as evidence of structure, not a selector.
14. **Unfitted gate instability.** A reviewer may ask why not select SADR only
    when its output is more confident. Response: the entropy selector is
    positive on calibration (+0.254 pp over SADR) but negative on confirmation
    (-0.161 pp); retain it as a transparent negative control.

## Data-access audit

- Train optimization uses only the fixed 2,371-image UIIS train split.
- Confirmation (511) and SUIM official test (110) are read only by evaluation
  scripts and are blocked by checkpoint flags during training.
- UVMulti is not used to tune the method and remains untracked local data.
