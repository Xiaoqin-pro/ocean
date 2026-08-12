# SADR unseen-composition diagnostic

## Purpose

The primary SADR protocol trains and confirms on 13 registered *single*
degradations. To check whether the adapter merely memorizes those names, we
froze each trained checkpoint and evaluated eight ordered compositions of the
same operators. The images are the 511-image confirmation split, but the
composed operators are new and no parameters or thresholds were changed.
Therefore this is a post-freeze diagnostic, not a replacement for the locked
confirmation gate.

## Results

| seed | mean gain over 8 compositions (pp) |
|---:|---:|
| 20260811 | +1.241 |
| 20260812 | +1.264 |
| 20260813 | +1.401 |
| **mean** | **+1.302** |

The larger stress-test gain is not contradictory to the smaller primary gain.
The primary table averages clean, mild, and single degradations, where the
frozen expert is already stronger (baseline mean mIoU 0.4799). The composition
stress test contains only compounded degradations, where the baseline mean is
0.4209 and the three SADR seeds average 0.4339. Thus the absolute gain rises
from +0.746 to +1.302 pp (roughly 1.55% to 3.09% relative to the corresponding
baseline) because the stress test exposes a larger error headroom. In the
paper, the 13-condition result should remain the primary claim and the
composition result should be reported as a harder, post-freeze stress test.

Every confirmation seed-by-composition result is positive. Averaged over the three seeds,
the two application orders give:

| composition family | mean gain (pp) |
|---|---:|
| color + turbidity | +1.830 |
| lowlight + blur | +0.825 |
| color + lowlight | +1.732 |
| turbidity + blur | +0.821 |

The exact per-seed JSON records are stored beside each checkpoint in
`outputs/sadr_long/unseen_compositions.json`,
`outputs/sadr_seed2/unseen_compositions.json`, and
`outputs/sadr_seed3/unseen_compositions.json`.

## Mechanism ablation

The seed-20260811 pixel-only checkpoint was evaluated with the same diagnostic
and achieved only +0.486 pp on average. Full SADR achieved +1.241 pp on the
same eight compositions, a +0.755 pp gap. In particular, pixel-only gains on
lowlight/blur were close to zero, while the task-supervised adapter remained
positive. This supports the interpretation that semantic supervision, rather
than generic image reconstruction, drives the compositional transfer.

## Interpretation boundary

This supports the narrower statement that the task-supervised residual
adapter transfers to selected unseen *compositions* of the registered image
operators on the reused confirmation scenes. A full 508-image calibration
audit averaged only +0.278 pp across the same eight ordered compositions (15/24
positive), with lowlight/blur averaging -0.590 pp. It does not establish
universal composition generalization or real-world UVMulti generalization,
because the UVMulti sanity subset did not improve, and the confirmation stress
test does not provide a new independent image split. In the paper it should be
presented as a stress test after the primary single-degradation table, with the
calibration audit and failed family reported explicitly.
