# UIIS DeepLabV3-MobileNetV3 Boundary Residual-Risk Analysis

## Protocol

- Evaluation split: UIIS confirmation, 511 original-image clusters under all 13
  frozen image conditions.
- Boundary definition: ground-truth class transition neighbourhood with fixed
  radius `3`; it is used only for evaluation stratification.
- Methods: raw softmax and the already frozen clean-global temperature.
- Statistics: 1,000 original-image cluster-bootstrap replicates per condition
  and method.
- Official SUIM TEST evaluated: **false**. Model retrained: **false**.

## Acceptance

- Aggregate rows: `52 = 13 conditions × 2 methods × 2 regions`.
- Per-image rows: `26,572 = 511 images × 13 conditions × 2 methods × 2 regions`.
- Bootstrap rows: `26 = 13 conditions × 2 methods`.
- All aggregate and bootstrap values are finite, with unique registered keys.
- Boundary diagnostic tests passed (`3 passed`).

## Main result

The mean boundary-minus-interior error-rate gap is `+24.40pp` for both raw and
clean-global probabilities. The equality is expected because scalar temperature
scaling preserves every pixel’s argmax.

For clean raw predictions, the gap is `+24.94pp` with a 95% cluster-bootstrap
CI of `[+23.53pp, +26.24pp]`. Every one of the 13 conditions has a strictly
positive 95% CI lower bound for both raw and clean-global probability outputs.

| Method | Mean boundary − interior error rate | Mean 95% CI |
| --- | ---: | --- |
| Raw | +24.40pp | [+23.02pp, +25.76pp] |
| Clean-global temperature | +24.40pp | [+23.01pp, +25.73pp] |

This independently reproduces the benchmark’s boundary-dominated residual-risk
pattern for the UIIS CNN cell. Temperature scaling changes probability
calibration but not the spatial concentration of segmentation errors.
