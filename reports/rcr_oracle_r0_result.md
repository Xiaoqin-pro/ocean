# RCR Gate R0: Frozen Full-image Repairability Oracle Result

## Decision

**PASS — R1a is authorized, but has not started.**  On the already-open,
non-blind 231-scene development role, all four preregistered full-image oracle
candidates satisfy every R0 screen.  The unique candidate fixed for a future
R1a protocol is **DeepLabV3-MobileNetV3 local -> SegFormer-B0 remote** because
it has the largest macro oracle mIoU gain.

| Local -> remote candidate | Repair rate | Full oracle mIoU gain | Boundary oracle mIoU gain | 95% CI lower bound |
| --- | ---: | ---: | ---: | ---: |
| SegFormer -> DeepLab | 44.61% | +5.68pp | +12.66pp | +6.37pp |
| DeepLab -> SegFormer | 51.10% | +8.48pp | +18.56pp | +9.09pp |
| SegFormer -> probability ensemble | 29.05% | +3.57pp | +7.01pp | +4.12pp |
| DeepLab -> probability ensemble | 35.36% | +5.87pp | +10.70pp | +6.36pp |

For every candidate, foreground macro-IoU gain and boundary gain were
positive, all four degradation families had positive mean oracle mIoU gain,
and the 1,000-replicate scene-cluster bootstrap lower bound for full oracle
mIoU gain exceeded zero.  These values are an **oracle upper bound**: the
union uses ground truth to identify exactly where a remote prediction repairs a
local error.  They do not establish a deployable system or selector.

## Provenance and access boundary

- Implementation commit: `1e417687765e04e1a6523d9986946b76c2bef9af`.
- Development CSV SHA-256:
  `FF27E7E33B3BE077D44DBE1B2B7F4649ED691CBC8D8C3F0DFD06DC64D35335E6`.
- SegFormer checkpoint SHA-256:
  `62CF10BA021B7E24429477C9C9C4690650EB0945D39E19DA0A8DEB3BB1132A5A`.
- DeepLab checkpoint SHA-256:
  `041655E0793D0F5DF792571E2F1FF9322082540971AE1A12709FC960DE7AAF59`.
- Evaluation had deterministic resize-only geometry and the fixed 13-condition
  registry.  The ensemble was equal-probability averaging, never raw-logit
  averaging.
- No model was retrained.  No crop refinement or communication simulation ran.
- Calibration, validation, official SUIM test, and external data remained
  unread.

The local raw output tables remain under `outputs/rcr_oracle_r0/`, accompanied
by their SHA-256 manifest; generated results are not committed.

## Consequence

R0 establishes substantial two-model **repairability space**, not a finished
communication method.  The only permitted next experiment is a newly frozen
R1a: use the selected DeepLab -> SegFormer pair, choose GT error regions only
to propose original-RGB crops, and unconditionally replace each fixed central
core with the remote local-crop prediction.  GT must not decide whether a
remote prediction is accepted.  R1b byte-constrained encoding is blocked until
R1a passes.
