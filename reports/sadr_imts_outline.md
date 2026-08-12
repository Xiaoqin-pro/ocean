# SADR paper outline for IMTS

## Working title

**SADR: Parameter-Efficient Task-Supervised Residual Adaptation for Robust
Underwater Segmentation with a Frozen Expert**

## One-sentence thesis

We study whether a pretrained underwater segmentation expert can acquire
degradation robustness without updating the expert itself. SADR learns a
zero-initialized task-supervised residual input adapter, obtaining measurable
robustness improvement with 0.296% trainable parameters, while exhibiting
structured but non-exclusive transfer to unseen degradation compositions.

## Three contributions

1. A zero-initialized, task-supervised input residual adapter that improves a
   frozen UIIS-F4 expert without updating the expert itself.
2. A locked three-seed robustness result and parameter-efficiency comparison:
   SADR averages +0.746 pp with 0.296% trainable parameters; rank-2 LoRA and
   partial FT are weaker, while full FT is a clearly stronger accuracy upper
   reference.
3. A transparent mechanism and boundary audit: composition transfer is
   structured, severity-dependent, family-selective, and non-exclusive, with
   explicit SUIM, blur, calibration, latency, and UVMulti limitations.

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
   composition-transfer comparison table, semantic-correction
   hierarchy with all-family/null controls, held-out calibration audit, and
   qualitative grid. Use the condition-gain plot in
   `outputs/sadr_long/condition_gain_plot.png` and qualitative grid in
   `outputs/sadr_long/qualitative_grid.png`.
   Add the parameter-efficiency table and Pareto figure from
   `reports/parameter_efficiency_result.md` and
   `outputs/parameter_efficiency/parameter_pareto.png`.
   Add the full/partial-FT composition table from the same report.
6. **Ablation:** pixel-only, no reconstruction, no distillation, rank-2 LoRA,
   GrayWorld, LAB-CLAHE, and head/last/full FT. Put frequency, routing,
   basis, selectors, and explicit composition penalties in the appendix.
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
- Treat this outline, the freeze manifest, and manuscript v0.1 as writing
  artifacts only; do not reopen training or method search during drafting.

## References to verify during writing

- TFUIE, *Task-Friendly Underwater Image Enhancement for Machine Vision
  Applications*, DOI [10.1109/TGRS.2023.3340244](https://doi.org/10.1109/TGRS.2023.3340244).
- STSC, *Semantic-aware Texture-Structure Feature Collaboration for Underwater
  Image Enhancement*, [arXiv:2211.10608](https://arxiv.org/abs/2211.10608).
- HSRUIE, *Task-Driven Underwater Image Enhancement via Hierarchical Semantic
  Refinement*, DOI [10.1109/TIP.2025.3647323](https://doi.org/10.1109/TIP.2025.3647323).
- UWSegFormer, a recent underwater segmentation comparison with explicit
  ablations and complexity reporting, [arXiv:2503.23422](https://arxiv.org/abs/2503.23422).
