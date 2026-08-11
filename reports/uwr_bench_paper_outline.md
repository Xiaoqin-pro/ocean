# UWR-Bench: Paper Outline

## Working title

**UWR-Bench: Leakage-Audited Evaluation of Calibration, Error Ranking, and Selective Risk in Underwater Semantic Segmentation**

## 1. Introduction

- Underwater semantic segmentation is exposed to color attenuation, turbidity, low light, and blur.
- Reliability is not a single property: calibrated probabilities, error ranking, and selective-risk behaviour answer different questions.
- The benchmark studies these axes under a fixed, leakage-audited protocol rather than presenting a new calibration method.
- Contributions to state only what is supported after the fixed 2x2 evidence matrix is complete.

## 2. Related Work

### 2.1 Underwater semantic segmentation

### 2.2 Probability calibration and uncertainty ranking

### 2.3 Selective prediction and conformal risk control

### 2.4 Reliability benchmarks and data leakage

## 3. Leakage-Audited Benchmark Protocol

### 3.1 SUIM scene-grouped split

- Describe exact-duplicate removal, near-duplicate scene review, exclusion of uncertain size-repaired masks, and scene-grouped train/validation/calibration partitions.
- Official SUIM TEST is held back for one final evaluation only.

### 3.2 UIIS instance-to-semantic conversion

- Describe deterministic instance-to-semantic conversion and label validation.

### 3.3 Cross-dataset duplicate audit

- Report the conservative UIIS-to-SUIM audit and exclusion process.
- State the final UIIS train/calibration/confirmation split and scene-group isolation.

### 3.4 Evidence roles

| Evidence source | Role |
| --- | --- |
| SUIM SegFormer-B0 | Exploratory development evidence |
| SUIM DeepLabV3-MobileNetV3-Large | Cross-architecture replication |
| UIIS SegFormer-B0 | External fixed-protocol benchmark extension; not a second blind confirmation |
| UIIS DeepLabV3-MobileNetV3-Large | Completed cross-dataset and cross-architecture fixed-protocol benchmark extension |
| SUIM official TEST | Locked final evaluation |

## 4. Reliability Evaluation

### 4.1 Controlled degradations

- Thirteen fixed conditions: clean plus four degradation families at three severities.
- The RGB image changes while its semantic mask remains fixed.

### 4.2 Calibration

- Raw softmax, clean-global temperature scaling, pooled temperature, and per-degradation temperature.
- Fit temperature only on calibration NLL; use evaluation partitions only for reporting.

### 4.3 Error ranking

- Fixed scores: raw MSP, calibrated MSP, entropy, probability margin, logit margin, energy, and local disagreement.
- Report eAURC, error AUPRC, and uncertainty recall in full, boundary, and interior regions.

### 4.4 Boundary/interior residual risk

- Use ground-truth boundaries only as an evaluation stratum.
- Radius 3 is the primary analysis; radius 1 and 5 are sensitivity analyses.
- Use original image identifiers as bootstrap clusters.

### 4.5 Selective risk

- Compare Global CRC, Oracle condition CRC (diagnostic only), and fixed quality-group CRC with global fallback.
- Report alpha values 0.05, 0.10, and 0.15 as fixed sensitivity analyses.

## 5. Results

### 5.1 Degradation sensitivity

### 5.2 Calibration-ranking separation

### 5.3 Boundary-dominated residual risk

### 5.4 Quality-conditioned CRC instability

### 5.5 Model-by-dataset interaction

## 6. Discussion

### 6.1 Why probability calibration does not necessarily improve error ranking

### 6.2 Why image quality is not a sufficient proxy for model difficulty

### 6.3 Consequences of cross-dataset image reuse

### 6.4 Implications for marine perception systems

## 7. Limitations

- State the limited backbone set, synthetic degradation design, and the fact that UIIS confirmation was previously opened for the DARC negative control.

## 8. Conclusion

- Final wording must follow the completed fixed evidence matrix and avoid claims of a successful quality-conditioned CRC method.
