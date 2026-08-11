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

Every seed-by-composition result is positive. Averaged over the three seeds,
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
adapter transfers to unseen *compositions* of the registered image operators.
It does not establish real-world UVMulti generalization, because the UVMulti
sanity subset did not improve, and it does not provide a new independent image
split. In the paper it should be presented as a stress test after the primary
single-degradation table.
