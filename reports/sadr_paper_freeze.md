# SADR paper-freeze v1

## Freeze identity

- Scientific freeze commit: `d03e19eba29a34061cb0789474f516faf92dfafb`
- Repository branch: `experiment/sgre-seg`
- PR: `Xiaoqin-pro/ocean#8`
- Freeze status: no new training, loss search, seed selection, or method branch
  is allowed after this point. Later commits may only reorganize evidence,
  render figures, correct non-numerical bugs, or write the manuscript.
- Test status at freeze: 174 tests passed.
- Local `data/uvmulti/` is intentionally untracked and excluded from the PR.

## Locked data roles

| role | split | images | permitted use |
|---|---|---:|---|
| train | `data/uiis_processed/splits/uiis_alpha010_confirmation/train.csv` | 2,371 | optimization only |
| calibration | `.../calibration.csv` | 508 | post-freeze audit; not a pristine final test because an earlier rejected gate inspected it |
| confirmation | `.../confirmation.csv` | 511 | primary 13-condition evaluation and post-freeze confirmation stress test |
| external | `data/suim_processed/splits/v2_scene_grouped_deduplicated/test.csv` | 110 | one-time official SUIM source-domain check |
| exploratory external | UVMulti local download | 5 labelled videos / 154 sampled frames | negative/heterogeneity audit only; no training or claim of real-world generalization |

The degradation registry is `configs/degradation_pilot.yaml` with clean plus
three severities each of color, turbidity, low-light, and blur. All primary
means use the same 13 conditions.

## Split and registry integrity hashes

These SHA-256 values are the source-of-truth identifiers used when drafting
the manuscript:

| file | SHA-256 |
|---|---|
| `train.csv` | `FF2CD6792F93A04DD0441DA2371CD44D675B777D9BC6480914E484B4667279AD` |
| `calibration.csv` | `DAEE4930B8BCD21B0F25BE8B38C2378B1BEDD016EC4048F268536F1DB8C68DA1` |
| `confirmation.csv` | `A6EC919CA841E01B418EA0B6AE00C9B660A3B80EAAF9EC1D460A15AB5B7AD08E` |
| `SUIM test.csv` | `7ACA7D08E57D27A26B13E375A35CA726DA43C9B92ABDCF32793B31CF825BAE38` |
| `configs/degradation_pilot.yaml` | `706A1DA10870C0659CD9CC35D8472F22FC727B2292933FEBD26961BB7666CF93` |

## Frozen checkpoints and seeds

| method | checkpoint | trainable parameters |
|---|---|---:|
| SADR seed 20260811 | `outputs/sadr_long/formal/checkpoints/final.pt` | 11,012 (0.296%) |
| SADR seed 20260812 | `outputs/sadr_seed2/formal/checkpoints/final.pt` | 11,012 (0.296%) |
| SADR seed 20260813 | `outputs/sadr_seed3/formal/checkpoints/final.pt` | 11,012 (0.296%) |
| frozen F4 source | `outputs/segformer_b0_uiis_alpha010_confirmation/checkpoints/last.pt` | 0 |
| full FT | `outputs/uiis_scdi_replication/formal/F4/checkpoints/final.pt` | 3,716,200 (100%) |
| head-only FT | `outputs/parameter_efficiency/formal/head/checkpoints/final.pt` | 396,808 (10.676%) |
| last-block-only FT | `outputs/parameter_efficiency/formal/last_block/checkpoints/final.pt` | 800,000 (21.527%) |
| rank-2 q/v LoRA | `outputs/parameter_efficiency/formal/lora/checkpoints/final.pt` | 8,192 (0.220%) |

SADR uses its fixed 256-pixel/batch-8 residual-adaptation configuration.
Head-only, last-block-only, full FT, and rank-2 LoRA share the fixed
384-pixel/batch-4 fine-tuning configuration. They share train split,
degradation exposure, source checkpoint, eight-epoch budget, and evaluation
roles, but not every optimizer or resolution setting.

## Primary evidence

| method/seed | UIIS confirmation mIoU | gain vs frozen |
|---|---:|---:|
| frozen F4 | 0.479872 | 0.000 pp |
| SADR / 20260811 | 0.485939 | +0.607 pp |
| SADR / 20260812 | 0.487471 | +0.760 pp |
| SADR / 20260813 | 0.488595 | +0.872 pp |
| SADR / three-seed mean | 0.487335 | **+0.746 pp** |

The corresponding official SUIM changes are -0.130, -0.197, and -0.207 pp
(mean **-0.178 pp**). The primary evidence is the locked 13-condition,
three-seed UIIS result; the SUIM result is a boundary check, not a positive
generalization claim.

Primary summaries:

- `outputs/sadr_long/confirmation_evaluation/summary.json`
- `outputs/sadr_seed2/confirmation_evaluation/summary.json`
- `outputs/sadr_seed3/confirmation_evaluation/summary.json`
- `outputs/sadr_long/suim_official_evaluation/summary.json`
- `outputs/sadr_seed2/suim_official_evaluation/summary.json`
- `outputs/sadr_seed3/suim_official_evaluation/summary.json`

## Comparative evidence

Seed-20260811 parameter-efficiency comparison on the same 13-condition
confirmation protocol:

| method | trainable parameters | confirmation gain | SUIM change |
|---|---:|---:|---:|
| SADR | 11,012 (0.296%) | **+0.607 pp** | -0.130 pp |
| rank-2 q/v LoRA | 8,192 (0.220%) | +0.226 pp | +0.409 pp |
| head-only FT | 396,808 (10.676%) | +0.192 pp | +0.381 pp |
| last-block-only FT | 800,000 (21.527%) | +0.312 pp | +0.558 pp |
| full FT | 3,716,200 (100%) | **+2.281 pp** | **+3.123 pp** |

These establish constrained adaptation, not an accuracy win over full FT.

## Mechanism diagnostic

The post-freeze severity-2 ordered-composition diagnostic averages over four
family pairs and both application orders:

| method | confirmation composition gain | calibration audit gain |
|---|---:|---:|
| SADR | +1.141 pp | -0.270 pp |
| rank-2 q/v LoRA | +0.667 pp | not run |
| head-only FT | +0.881 pp | +1.021 pp |
| last-block-only FT | +1.693 pp | +1.557 pp |
| full FT | +5.807 pp | +3.690 pp |

Composition transfer is measurable but not SADR-exclusive. The broader SADR
screen found severity-dependent order/additivity structure and a correct
same-image additive relation above a shuffled null in 54/54 cases. These are
mechanism diagnostics, not independent generalization proofs.

## Engineering and boundary evidence

- Frozen F4 latency: 11.14 ms; F4+SADR: 13.53 ms; measured at 384x384 with
  CUDA events on the experiment laptop RTX 4060-class GPU, 30 iterations.
  This is +21.4% inference latency, so parameter efficiency is not compute
  efficiency.
- Blur-s3 remains a weakness in the primary family analysis.
- Calibration eight-composition SADR audit: +0.278 pp for the original
  three-seed eight-composition diagnostic; the matched severity-2 audit in
  this freeze is -0.270 pp. These are different protocols and must not be
  mixed.
- UVMulti official `val/video108` was essentially unchanged across seeds;
  train-video results were heterogeneous. No real-world generalization claim.
- Pixel-only, explicit composition regularizers, frequency, routing, basis,
  and gate variants are negative controls or appendix material.

## Allowed claims

1. SADR is a zero-initialized, task-supervised input residual adapter before a
   frozen underwater segmentation expert.
2. It improves the locked UIIS 13-condition robustness score by +0.746 pp on
   average across three seeds while updating 0.296% of expert parameters.
3. It outperforms the tested head-only, last-block-only, and rank-2 LoRA
   controls on the target confirmation protocol, while full FT is stronger.
4. Its correction/transfer behavior shows structured, severity-dependent and
   family-selective composition diagnostics, but composition transfer is not
   unique to SADR and is not universal.

## Forbidden claims

- SADR matches or beats full fine-tuning.
- SADR is the first task-driven underwater enhancement method.
- Composition transfer is unique to SADR, universally valid, or independently
  established by the reused confirmation stress test.
- SADR improves SUIM or real-world UVMulti generalization.
- 0.296% trainable parameters implies 0.296% of training compute or latency.
- The +0.746 pp primary gain, +1.141 pp matched composition diagnostic, and
  +1.302 pp original eight-composition diagnostic are interchangeable.
