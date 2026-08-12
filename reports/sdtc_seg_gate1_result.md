# SDTC-Seg Gate-1 result

## Frozen decision

**FAIL.** SDTC improved every evaluated condition, but the effect was far
below the preregistered publication threshold.

| Metric (SDTC - F4) | Change (mIoU points) |
|---|---:|
| 13-condition mean | +0.045 |
| four severity-3 conditions | +0.050 |
| clean | +0.058 |
| color severity 3 | +0.034 |
| turbidity severity 3 | +0.057 |
| low-light severity 3 | +0.062 |
| blur severity 3 | +0.045 |

All four severe families were positive and clean safety passed, but the
magnitude gate failed (+0.50 mean or +1.00 severe required). Formal
validation, calibration, and official SUIM test data were not evaluated.

## Interpretation

The class-tangent supervision is directionally useful, unlike DTS, but a
single shared rectifier produces almost the same small shift for every
condition. Mean class gains were largest for HD (+0.148 point), while RF and
RI decreased slightly. This supports one final structural hypothesis:
explicit semantic routing among class-specific inverse-degradation experts.
It does not support tuning the SDTC loss weights or training duration.

The comparator also has little corruption headroom in the current registry:
relative to clean, F4 loses only 0.125 point under severe turbidity, 0.122
under severe low light, 0.467 under severe color, and 1.506 under severe blur.
Future full experiments therefore need an unseen compound-degradation stress
test in addition to this mild continuity check; the stress test is not a
replacement for clean accuracy or public-dataset replication.
