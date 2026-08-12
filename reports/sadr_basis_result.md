# Compositional Basis SADR result

## Hypothesis

The basis variant replaces the single residual image head with four shared
residual primitives and an image-conditioned coefficient vector. The intended
mechanism is that a compound degradation activates several primitives without
requiring a family label or an explicit additive logit loss.

## Two implementation audits

The first implementation initialized both the primitive output and coefficient
head at zero. This created an exact symmetry: the primitive output had zero
gradient into the coefficient head, so all four coefficients remained equal.
That run is not treated as a scientific comparison.

The corrected implementation uses tiny `1e-3` primitive-output weights and a
small fixed coefficient-bias spread. This keeps the initial residual near zero
while allowing the coefficient branch to learn.

## Locked results

| variant | UIIS confirmation gain | SUIM official change | eight reused-scene compositions |
|---|---:|---:|---:|
| ordinary SADR, seed 20260811 | **+0.607 pp** | -0.130 pp | **+1.241 pp** |
| basis, symmetry-broken first audit | +0.390 pp | -0.050 pp | +0.906 pp |
| basis, learnable primitive audit | +0.375 pp | -0.064 pp | +0.853 pp |

The basis variants are below ordinary SADR on the primary protocol and on the
composition stress test. The external source-domain cost is not enough to
offset the primary robustness loss.

## Coefficient diagnosis

For the learnable audit, mean coefficients on a 64-image confirmation screen
were approximately:

| view | coefficients |
|---|---|
| clean | [0.515, 0.539, 0.565, 0.588] |
| color-s1 | [0.536, 0.559, 0.584, 0.607] |
| lowlight-s1 | [0.559, 0.581, 0.607, 0.628] |
| color-s2 + turbidity-s2 | [0.536, 0.558, 0.585, 0.607] |
| lowlight-s2 + blur-s2 | [0.602, 0.623, 0.648, 0.668] |
| color-s3 + lowlight-s2 | [0.628, 0.648, 0.672, 0.690] |

The four coefficients move together with degradation strength. They do not
show independent family-specific activation, so the proposed basis does not
provide a structural explanation for the observed family-selective transfer.

## Decision

Close this route as a negative architectural control. Do not add the basis
front-end to the claimed method. Retain the implementation and result for the
paper's negative-evidence section; it demonstrates that simply factorizing a
residual into shared primitives does not reproduce SADR's transfer.
