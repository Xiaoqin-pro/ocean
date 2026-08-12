# SADR paper outline for IMTS

## Working title

**SADR: Emergent Partial Semantic Composition for Robust Underwater
Segmentation with a Frozen Expert**

## One-sentence thesis

Rather than retraining an underwater segmenter or optimizing enhancement for
human visual quality, learn a zero-initialized residual image adapter using the
frozen segmenter's semantic loss and clean/degraded public training pairs; the
resulting semantic correction exhibits severity-dependent, family-selective
composition structure, with only 0.296% trainable parameter overhead.

## Contributions to claim

1. A task-aware residual adapter that is safe at initialization: the frozen
   UIIS-F4 expert is unchanged and the last adapter layer is zero initialized.
2. A paired semantic-restoration objective that combines segmentation loss,
   clean identity, reconstruction, confidence-weighted distillation, and
   residual regularization. The ablation shows task loss, not pixel recovery,
   is the decisive term.
3. A frozen, preregistered 13-condition robustness protocol with three seeds,
   family/severity analysis, external SUIM cost, conventional GrayWorld
   control, and explicit negative controls for frequency and routing variants.
   Report the small-sample 95% t interval for the seed gain
   ([+0.415, +1.078] pp) as a stability indication only.
4. A post-freeze compositional stress test: eight unseen ordered combinations
   of registered degradations are positive for all three seeds, with mean gain
   +1.302 pp. Label it diagnostic because it reuses the confirmation images;
   explain that its lower baseline (0.4209 versus 0.4799 in the primary table)
   creates more error headroom rather than implying a contradictory benchmark.
5. A mechanism analysis over all six family pairs and matched severities,
   including operator commutativity and shuffled-image null controls. Correct
   same-image additivity beats the null in 54/54 screening cases, while order
   and additivity degrade at high severity. Directly imposing the relations
   harms the primary task, so the transfer is treated as an emergent,
   family-selective property rather than a claimed linear law.
6. A parameter-efficiency reference that prevents interpreting the SADR gain
   against only a frozen baseline: with the same train-only budget, SADR's
   0.296% adapter gives +0.607 pp on confirmation, versus +0.192 pp for
   decoder-head-only and +0.312 pp for last-block-only fine-tuning. Full
   fine-tuning remains stronger (+2.281 pp confirmation, +3.123 pp SUIM), so
   this is a low-update robustness trade-off, not a claim of matching full FT.
7. A matched composition-transfer control showing that the stress-test gain
   is not exclusive to SADR: at severity 2, the four-pair confirmation audit
   gives +1.141 pp for SADR, +0.881 pp for head-only, +1.693 pp for
   last-block-only, and +5.807 pp for full FT; the calibration audit gives
   -0.270, +1.021, +1.557, and +3.690 pp. This narrows the mechanism claim to
   constrained, measurable transfer under a 0.296% frozen-expert update.
8. A parameter-matched rank-2 q/v LoRA control: 8,192 trainable parameters
   (0.220%) yield only +0.226 pp on confirmation, versus SADR's +0.607 pp.
   This supports the narrower claim that task-supervised input-side residual
   placement matters beyond parameter count alone.

## Claims not to make

- Do not call SADR the first task-driven underwater enhancement method. TFUIE,
  STSC, and HSRUIE already use semantic or downstream-task guidance.
- Do not claim real-world UVMulti generalization: official val/video108 is
  essentially unchanged, while train-video exploratory results are highly
  heterogeneous and not a held-out test.
- Do not claim universal blur recovery: blur-s3 is a known weakness.
- Do not report the +0.746 pp mean as a percentage; it is +0.00746 absolute
  mIoU, or +0.746 percentage points.
- Do not say SADR replaces full fine-tuning or improves SUIM: the matched
  full-FT and SUIM results show the opposite.

## Main table to reproduce

| model | confirmation mIoU | delta |
|---|---:|---:|
| UIIS-F4 | 0.479872 | -- |
| SADR-8, seed 20260811 | 0.485939 | +0.607 pp |
| SADR-8, seed 20260812 | 0.487471 | +0.760 pp |
| SADR-8, seed 20260813 | 0.488595 | +0.872 pp |
| **SADR-8, three-seed mean** | **0.487335** | **+0.746 pp** |

External SUIM changes for the same seeds are -0.130, -0.197, and -0.207 pp;
report the mean -0.178 pp and do not hide it.

## Suggested section logic

1. **Introduction:** underwater degradation harms machine perception; ordinary
   enhancement is not a safe fix; retraining a full segmenter is expensive and
   can forget the source domain. State the frozen-expert robustness gap.
2. **Related work:** separate human-oriented UIE, task-driven UIE (TFUIE/STSC/
   HSRUIE), and robust segmentation adapters. Position SADR as a lightweight
   frozen-expert robustness protocol, not as the first semantic UIE method.
3. **Method:** define the frozen expert `f`, residual adapter `R`, zero-init
   safety, paired views, objective, and inference cost. Include a diagram of
   clean/degraded views flowing through `R` into `f`.
4. **Protocol:** describe the 2,371-image train role, 511-image confirmation
   role, 110-image SUIM external role, and 13 deterministic conditions. State
   that confirmation/test data never influence training or thresholds.
5. **Results:** main table, three-seed error bars, per-family plot, unseen
   composition stress-test table, semantic-correction compositionality
   hierarchy with all-family/null controls, held-out calibration audit, and
   qualitative grid. Use the condition-gain plot in
   `outputs/sadr_long/condition_gain_plot.png` and qualitative grid in
   `outputs/sadr_long/qualitative_grid.png`.
   Add the parameter-efficiency table and Pareto figure from
   `reports/parameter_efficiency_result.md` and
   `outputs/parameter_efficiency/parameter_pareto.png`.
   Add the full/partial-FT composition table from the same report.
6. **Ablation:** pixel-only, no reconstruction, no distillation, frequency
   split, feature consistency, routed experts, GrayWorld, and LAB-CLAHE.
   Emphasize that task loss is necessary but extra architectural branches and
   ordinary contrast enhancement did not explain the gain. Include additive,
   logit-order, probability-KL, and composite-distillation controls as failed
   attempts to impose compositionality.
7. **Limitations:** source-domain cost, blur-s3 weakness, lowlight/blur
   failure in the held-out audit, partial UVMulti, confirmation-scene reuse in
   the stress test, the 21.4% measured latency overhead despite tiny parameter
   overhead, and the five-video UVMulti heterogeneity audit. Add the unfitted confidence self-gate as a negative
   control: it reverses sign between calibration and confirmation.

## Reviewer-risk checklist

- Compare against at least one more conventional enhancement baseline than
  GrayWorld if a reproducible public implementation can be obtained.
- Report exact latency hardware and batch size, not only parameter count.
- Include per-condition results rather than only a single mean.
- Include both GrayWorld and LAB-CLAHE as non-learning controls.
- Keep confirmation/test access hashes and split counts in the appendix.
- Explain why three seeds are a stability check, not a population-level claim.
- Make the parameter-efficiency comparison explicit: partial fine-tuning is
  a matched reference, while full fine-tuning is an upper-accuracy reference;
  neither is to be hidden behind the SADR result.

## References to verify during writing

- TFUIE, *Task-Friendly Underwater Image Enhancement for Machine Vision
  Applications*, DOI [10.1109/TGRS.2023.3340244](https://doi.org/10.1109/TGRS.2023.3340244).
- STSC, *Semantic-aware Texture-Structure Feature Collaboration for Underwater
  Image Enhancement*, [arXiv:2211.10608](https://arxiv.org/abs/2211.10608).
- HSRUIE, *Task-Driven Underwater Image Enhancement via Hierarchical Semantic
  Refinement*, DOI [10.1109/TIP.2025.3647323](https://doi.org/10.1109/TIP.2025.3647323).
- UWSegFormer, a recent underwater segmentation comparison with explicit
  ablations and complexity reporting, [arXiv:2503.23422](https://arxiv.org/abs/2503.23422).
