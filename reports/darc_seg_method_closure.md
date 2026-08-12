# DARC-Seg method closure

## Decision

**DARC-Seg quality-group CRC is rejected as a method hypothesis.**

## Evidence

The candidate was not selected after viewing UIIS confirmation. Its broad
three-risk-target claim first failed on the SUIM pilot. The only exploratory
signal, alpha = 0.10, was then preregistered and evaluated once on the
leakage-screened UIIS confirmation set.

| UIIS seed | Coverage difference vs. Global CRC | 95% CI |
| ---: | ---: | ---: |
| 20260722 | -1.63 pp | [-1.80, -1.45] pp |
| 20260723 | -2.14 pp | [-2.32, -1.92] pp |
| 20260724 | -1.64 pp | [-1.81, -1.48] pp |

All intervals are below zero. The candidate reduced selective risk slightly
only by accepting fewer pixels. It therefore failed the preregistered primary
endpoint: higher coverage at the same target risk.

## Consequence

No additional DARC tuning is permitted: no KMeans seed, descriptor, group
count, alpha, fallback, or threshold modification. The code and results remain
as a transparent negative control.

The active research direction is now UWR-Bench: leakage-audited reliability
and selective-prediction benchmarking under underwater degradation, with
cross-architecture replication as the next gate. The SUIM official TEST
remains locked.
