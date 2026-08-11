# Research Program Closure — 2026-07-28

## Final scope decision

The method-development phase using the existing SUIM/UIIS data, fixed
synthetic degradation registry, and SegFormer-B0 / DeepLabV3-MobileNetV3
backbones is closed.  This is a scientific stop decision, not an engineering
failure: each candidate method received its pre-registered low-cost or oracle
screen before expensive expansion.

## Evidence retained for UWR-Bench

The following fixed evidence remains valid and is the basis for a benchmark /
diagnostic paper:

1. Scene-grouped, leakage-audited SUIM v2 protocol, isolated calibration role,
   and locked official test.
2. Leakage-screened UIIS protocol and completed 2x2 matrix across dataset and
   architecture.
3. Controlled 13-condition degradation benchmark.
4. Probability calibration improves under scalar temperature scaling while
   segmentation argmax and mIoU remain invariant.
5. Probability calibration and error ranking are separate: calibration does
   not yield a stable ranking improvement.
6. Boundary pixels carry consistently higher residual error/risk than interior
   pixels.
7. Visual-quality grouped CRC is not a stable risk-coverage improvement.

These are descriptive, fixed-protocol benchmark claims.  They must not be
rewritten as evidence that any rejected method works.

## Permanently closed method routes

| Route | Frozen conclusion | Closure artifact |
| --- | --- | --- |
| DARC-Seg quality-group CRC | No stable coverage gain; negative control only. | `reports/darc_seg_method_closure.md` |
| AquaRiskMap | Stage-A failure; no repairability/ranking rescue. | `reports/aquariskmap_stage_a_decision.md` |
| FT-Reliability / TCCR | Primary full eAURC and key secondary metrics were adverse. | `reports/ft_reliability_stage_a_result.md` |
| DTH conditional horizon | 98.60%–99.43% censored labels; Gate 1 failed. | `reports/dth_horizon_gate1_result.md` |
| RCR local evidence correction | R0 oracle space exists, but R1a is scene-unstable, damaging, and uses 81.57% context area. | `reports/rcr_oracle_r1a_result.md` |

Do not change losses, severity windows, data roles, crop/context rules,
thresholds, model pairs, selectors, codecs, or bootstrap definitions to
rescue any of these routes.  In particular, RCR R1b is not authorized.

## Data-access record

The RCR development screens accessed only the already-open 231-image
development role.  TCCR and DTH did not access SUIM calibration, validation,
or official test.  The official SUIM test remains locked and may not be used
for new method development.

## Permitted next work

Only non-method-development work is permitted without a new research protocol:

- consolidate the UWR-Bench paper outline, result matrix, figures, and tables;
- add semantic-validity and engineering summaries from frozen outputs;
- prepare reproducibility documentation, environment provenance, and release
  manifests;
- freeze paper claims and the final evaluation protocol.

After those items are frozen, a user-authorized **single** official SUIM test
evaluation may be used only for final reporting.  It must not change methods,
metrics, thresholds, figures, or claims.

## What a future method paper would require

A new method direction needs new information rather than another variation of
current confidence, ranking, crop, or synthetic-degradation signals: for
example, real controlled underwater degradation sequences, new sensing
modalities, a real AUV communication task, or a collaboration with collection
and device access.  Such a direction requires an independent protocol and is
outside this closed program.
