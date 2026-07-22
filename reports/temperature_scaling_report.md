# SUIM v2 Temperature Scaling

## Protocol

- Frozen checkpoint: `outputs/segformer_b0_suim_v2_scene/checkpoints/best.pt`
- Checkpoint SHA-256: `62CF10BA021B7E24429477C9C9C4690650EB0945D39E19DA0A8DEB3BB1132A5A`
- Splits: calibration for fitting; validation for comparison only.
- Official TEST was not evaluated and the model was not retrained.
- Caches contain 26 AMP-consistent condition/split entries; metrics contain 104 rows.

## Fitted temperatures

| Scope | Temperature |
| --- | ---: |
| clean | 2.1972 |
| pooled | 2.2850 |
| color attenuation | 2.1903 |
| turbidity | 2.2494 |
| low light | 2.3281 |
| blur | 2.3935 |

All temperatures were fitted as `exp(log_temperature)` by minimizing calibration NLL only.

## Validation means across 13 conditions

| Method | NLL | ECE | Brier | AURC |
| --- | ---: | ---: | ---: | ---: |
| Raw | 0.782306 | 0.098327 | 0.260191 | 0.062439 |
| Clean global | 0.528575 | 0.022567 | 0.243977 | 0.063708 |
| Pooled | 0.529494 | 0.025260 | 0.244549 | 0.063839 |
| Per-degradation | 0.529304 | 0.025282 | 0.244433 | 0.063838 |

## Integrity checks

- All 20 project tests passed before this audit update; the added cache-integrity unit test is included in the follow-up test run.
- Temperature fitting did not worsen calibration NLL for any of the six fitting scopes.
- Every fit was finite, stayed away from the configured temperature bounds, and records optimizer iterations plus function evaluations in `experiments/temperature_scaling_fit_history.json`.
- Cache evaluation now fails closed unless every cache has the expected checkpoint SHA-256, degradation-config SHA-256, consistent sample/logit/label counts, and unique sample IDs.
- Positive scalar temperature preserved every cached argmax and all segmentation metrics within each condition/split.
- Raw cache replay matched the frozen degradation pilot within the documented AMP replay tolerance; maximum mIoU difference was `1.66e-5`.

## Interpretation

Temperature scaling strongly improves NLL, ECE and Brier score over raw softmax.  The clean-global temperature has the best aggregate validation NLL and ECE in this first scalar comparison.  Per-degradation scaling obtains lower validation NLL than pooled scaling in 8 of 13 conditions, but does not provide a stable aggregate advantage.  Therefore the evidence supports calibration under degradation, but does not yet justify a more complex learned conditional calibration head.  The next study should analyze condition-level failure patterns and image-level predictors before adding new network parameters.
