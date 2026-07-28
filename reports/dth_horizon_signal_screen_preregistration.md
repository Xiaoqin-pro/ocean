# DTH Conditional Degradation-Onset Horizon Signal Screen

## Status and scope

This is Gate 1 of a new, independent DTH screen.  It is not DTH head
training, model selection, calibration, or a formal SUIM evaluation.  The
FT-Reliability/TCCR route is frozen as failed at parent commit
`8ccc2d97757b1ba477aed64fa1b79eb4f847abbb` and will not be altered.

Gate 1 may read only the frozen 936-image `risk_head_train.csv` role.  It
must not read `risk_head_development.csv`, SUIM validation, calibration, or
official test splits, or any external dataset.

## Frozen base model

The sole model is the FT Stage-A SegFormer-B0 Variant B final checkpoint:

- Role: frozen Degradation-CE comparator, without FT or retention.
- Path: `outputs/ft_reliability_suim_pilot_v1_2/segformer/formal/B/checkpoints/final.pt`.
- SHA-256: `F9C8D08B39ED04295F63006381384DBD987CAC3B65BFF11A7C8D736DE23430F6`.

The audit aborts if the checkpoint metadata or SHA-256 differs.  No model
parameters may be optimized.

## Coupled pixel trajectories

For every sample and each family `color`, `turbidity`, `lowlight`, and
`blur`, Gate 1 evaluates `clean`, `s1`, `s2`, and `s3`.  Geometry is a
deterministic 384 by 384 resize: crop and flip are prohibited.

For a fixed `(sample_id, family)`, all severities use the same deterministic
trajectory seed and spatial latent pattern.  Only severity parameters change.
The audit verifies parameter directions before inference: blur sigma must
increase; turbidity transmission must decrease; low-light exposure must
decrease; and color attenuation strength must increase.

## Conditional first-failure label

Only pixels eligible on the clean image are supervised:

`eligible = (clean prediction == ground truth) AND (ground truth != ignore)`.

For eligible pixels, `H1` is first wrong at `s1`; `H2` remains correct at
`s1` and is first wrong at `s2`; `H3` remains correct at `s1/s2` and is first
wrong at `s3`; and `censored` remains correct through `s3`.  A later recovery
does not change the first-failure label.  It is reported separately, including
`s1 wrong -> s2 correct` and `s2 wrong -> s3 correct` rates whose denominators
are the lower-severity wrong eligible pixels.

This is conditional degradation-onset prognosis, not an all-pixel current
error detector.  Gate 1 makes no deployment or survival-head claim.

## Outputs

The audit writes atomic scene-level and stratified tables by family, class,
boundary/interior region, and status.  It includes eligible counts; H1/H2/H3/
censored counts and rates; event-scene counts; adjacent C/W transitions;
post-failure recovery; descriptive MSP/entropy/top-gap bins and correlations;
and input, checkpoint, configuration, operator, and output SHA-256 hashes.

## Frozen Gate-1 screen

At least three families must satisfy all of the following:

1. At least 200 scenes contain eligible pixels.
2. At least 100 scenes contain any H1/H2/H3 event.
3. H2 and H3 each occur in at least 30 distinct scenes.
4. The largest H1/H2/H3/censored state is at most 90 percent of eligible pixels.
5. More than two families may not have post-failure recovery above 30 percent.

MSP, entropy, and top-gap descriptions are not stop criteria.  If Gate 1
passes, Gate 2 will be separately preregistered: it will use 936 images for
head training and the already-open 231-image development role for non-blind
development comparisons among confidence, cumulative, categorical, ordinal,
and hazard parameterizations.  Gate 2 is not implemented by this change.
