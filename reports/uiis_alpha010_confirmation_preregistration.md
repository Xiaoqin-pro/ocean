# UIIS α=0.10 Independent Confirmation Protocol

## Status and rationale

This confirmation is motivated by an **exploratory** SUIM pilot observation:
at the moderate target risk `alpha=0.10`, a train-only quality-group CRC showed
a positive coverage gain over global CRC. The broad three-target DARC-Seg claim
failed and is not revived by this protocol.

This document is frozen before any UIIS confirmation result is inspected.

## Dataset admission gate

UIIS may be admitted only after the following are recorded:

1. Dataset origin, version, license, and annotation conversion are audited.
2. COCO instances are deterministically rasterized to an eight-class semantic
   mask: background plus fish, reefs/invertebrates, aquatic plants, wrecks,
   divers, robots, and sea-floor.
3. Exact SHA-256 and reviewed near-duplicate comparisons against every SUIM
   train, calibration, validation, and locked TEST image show no reuse. Any
   UIIS scene linked to SUIM is excluded before split generation.
4. A fresh scene-aware UIIS `train / calibration / confirmation` split is
   created. The source's published test partition is not used for model or
   method selection.

## Frozen confirmation method

- SegFormer-B0 only; one clean UIIS training run after data admission.
- Fixed target risk: `alpha = 0.10`.
- Main ranking score: raw MSP uncertainty.
- Baseline: global image-level CRC.
- Candidate: three-group, train-only quality CRC with global fallback.
- Descriptors, 192-pixel measurement resolution, KMeans seeds, coverage grid,
  CRC correction, and cluster unit are copied from the SUIM pilot unchanged.
- Each original UIIS image is one statistical unit. Its deterministic
  degradation versions form one cluster.
- The UIIS confirmation split is evaluation only: it must not select seeds,
  descriptors, group count, model epoch, or risk target.

## Primary endpoint and decision rule

The candidate passes only if all are true on UIIS confirmation:

1. Coverage improvement over global CRC is at least 3 percentage points.
2. The 95% paired cluster-bootstrap lower bound is greater than zero.
3. Aggregate selective risk does not materially exceed 0.10.
4. The worst quality group and severe low-light condition do not worsen in
   risk excess.
5. The direction is consistent across all frozen KMeans seeds.

The bootstrap uses 1,000 deterministic resamples of original-image clusters.
Failure of any endpoint terminates this method route; no additional descriptor,
seed, grouping, neural head, or backbone search is permitted.

## Scope boundary

The SUIM official TEST remains locked. UIIS results, regardless of outcome,
do not authorize changing the frozen SUIM method or using SUIM validation for
another selection step.
