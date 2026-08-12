# SADR statistical and control audit

## Main seed stability

The locked UIIS confirmation protocol gives the following 13-condition mean
mIoU gains for independent eight-epoch SADR runs:

| seed | gain (pp) |
|---:|---:|
| 20260811 | +0.6066 |
| 20260812 | +0.7599 |
| 20260813 | +0.8723 |
| **mean** | **+0.7463** |

The sample standard deviation is 0.1334 pp. A small-sample one-sample t-test
against zero gives `t(2)=9.693`, `p=0.0105`; the two-sided 95% t interval is
[+0.4150, +1.0775] pp. This is reported only as a stability indication because
the independent unit is the training seed (`n=3`), not thousands of pixels.

The untouched SUIM official changes are -0.130, -0.197, and -0.207 pp for the
same three seeds (mean -0.178 pp). The result is therefore a controlled UIIS
multi-condition robustness improvement with a small source-domain cost, not
evidence of unconditional cross-dataset improvement.

## Semantic-view consistency control

The control adds a symmetric KL penalty between the frozen expert's outputs on
the restored s1/s2/s3 views of one image. It uses the same front-end,
optimizer, seed, epoch count, train split, and evaluation protocol as SADR.

| run | confirmation gain (pp) | SUIM change (pp) |
|---|---:|---:|
| ordinary SADR, seed 20260811 | +0.6066 | not re-run in this control |
| SADR + semantic-view consistency, seed 20260811 | +0.6069 | -0.1377 |

The paired difference is +0.0003 pp, far below the seed-to-seed spread. The
consistency term is retained as a negative control and is not part of the
claimed method.

An edge-preserving reconstruction control adds first-order gradient matching to
the pixel loss. It reaches +0.6125 pp on confirmation and -0.1171 pp on SUIM
official; blur-s3 remains negative. The +0.0058 pp confirmation difference
from ordinary SADR is below the seed spread and is not claimed as a separate
mechanism.

## Safe interpretation

The evidence supports the narrower claim that a zero-initialized,
task-supervised residual adapter can improve robustness of a frozen UIIS
segmenter under the preregistered synthetic degradation family at 0.296% of
the expert's parameters. It does not support claims of state-of-the-art
performance, broad real-world UVMulti generalization, or a universally useful
consistency loss.
