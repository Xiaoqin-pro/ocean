# Controlled Degradation Reliability Pilot

Date: 2026-07-22  
Experiment branch: `experiment/degradation-pilot`  
Code commit: `b34e0898d1f5b43d38dc6d84f1e7ed30d20dda43`

## Scope and provenance

- Frozen checkpoint: `outputs/segformer_b0_suim_v2_scene/checkpoints/best.pt`, epoch 80.
- Checkpoint SHA-256: `62CF10BA021B7E24429477C9C9C4690650EB0945D39E19DA0A8DEB3BB1132A5A`.
- Baseline configuration SHA-256: `95C0C4AECACADE9B6D456BD4DB259EAB881D61A9864E883F58F45A557ED4BA26`.
- Degradation configuration SHA-256: `12EB5FBB631DAC1B1C5B5CECA6B6C2CEAF426186F89DFFA7A08330401BCD1286`.
- Conditions: clean plus color attenuation, turbidity/scattering, low light, and Gaussian blur at severities 1--3 (13 conditions total).
- Evaluation splits: validation and calibration only. Official TEST was not accessed.
- Model weights were not updated and no optimizer was instantiated.

The complete 26-row aggregate table is versioned in `experiments/degradation_pilot_metrics.csv`; classwise IoU, Dice, accuracy, and ECE are in `experiments/degradation_pilot_per_class.csv`. Generated images, plots, logits, and weights remain local.

## Clean sanity check

| Split | Clean pilot mIoU | Formal baseline mIoU | Difference | Clean pilot ECE | Formal baseline ECE | Difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.602939 | 0.602939 | 0.000000 | 0.089123 | 0.089123 | 0.000000 |
| Calibration | 0.652563 | 0.652563 | 0.000000 | 0.100071 | 0.100071 | 0.000000 |

The clean condition exactly reproduces the frozen baseline, validating preprocessing, checkpoint selection, metric computation, and the image-only degradation wrapper.

## Main trends

| Split | Clean mIoU / ECE | Color s3 mIoU / ECE | Turbidity s3 mIoU / ECE | Low-light s3 mIoU / ECE | Blur s3 mIoU / ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.6029 / 0.0891 | 0.5837 / 0.0903 | 0.5754 / 0.0956 | 0.4809 / 0.1536 | 0.5365 / 0.1108 |
| Calibration | 0.6526 / 0.1001 | 0.6287 / 0.1134 | 0.6343 / 0.1213 | 0.5473 / 0.1584 | 0.5558 / 0.1489 |

- Severity increases generally reduce mIoU and increase ECE, NLL, Brier score, and AURC. Exact monotonicity is not expected for every metric under every mild condition.
- Low light is the clearest stressor. On validation, low-light s3 lowers mIoU by 0.1220 and raises ECE by 0.0645; AURC rises from 0.0558 to 0.1191.
- Blur is the next strongest reliability stressor. On calibration, blur s3 raises ECE from 0.1001 to 0.1489 and AURC from 0.0684 to 0.0997.
- Mild turbidity can leave or slightly improve mIoU on this finite split, but strong turbidity degrades both segmentation and calibration. This is reported as an empirical split-level effect, not a claim of robustness.

## Interpretation boundary

This pilot supports the next controlled comparison: clean-global, pooled, and per-degradation temperature scaling. Temperature parameters must be fitted only on calibration NLL; validation remains a comparison split. Temperature scaling must preserve the pixelwise argmax and mIoU, and the official TEST remains locked until the method is fixed.
