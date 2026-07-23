# DARC-Seg CRC Pilot

## Protocol

- Frozen SegFormer-B0 checkpoint: `62CF10BA021B7E24429477C9C9C4690650EB0945D39E19DA0A8DEB3BB1132A5A`.
- Frozen configuration SHA-256: `263D1DDFEC741CA01A715DA7D6957C4FE7598D98F73DBD490257A4FCE5818DFB`.
- Fit split: calibration; evaluation split: val; 146 original `sample_id` clusters per split.
- Each independent cluster retains all 13 registered degradation versions.
- Official TEST was not accessed and no model was retrained.
- Main selector: raw-MSP uncertainty. Ground-truth boundaries are evaluation strata only.

## Controllers

The pilot compared no rejection, naive empirical selection, image-clustered
global CRC, oracle condition CRC (diagnostic only), and train-only KMeans
quality-group CRC. CRC uses the official finite-sample correction from the
Conformal Risk Control reference implementation and the monotone risk envelope
over a fixed 1%--100% coverage grid.

Three train-only KMeans seeds produced three usable groups. Their calibration
cluster counts were respectively `146/144/100`, `146/144/101`, and
`146/144/101`; all exceed the preregistered minimum of 80.

## Primary validation observations

| Target risk | Global CRC coverage | Oracle coverage gain | Quality-group coverage gain |
| --- | ---: | ---: | ---: |
| 0.05 | 1.0% | 0.23pp | 0.69--0.70pp |
| 0.10 | 31.0% | 3.46pp | 4.52--5.90pp |
| 0.15 | 75.0% | 0.54pp | 2.18--2.81pp |

At `alpha=0.10`, global CRC was conservative (13-condition macro selective
risk 0.0611) and the quality grouping yielded positive 95% paired,
image-clustered bootstrap coverage intervals (about +3.2pp to +7.3pp,
depending on seed). The severe low-light risk did not worsen at that target.

## Gate decision: stop

The DARC-Seg pilot does **not** pass the preregistered expansion gates.

1. Oracle condition CRC has substantial headroom only at `alpha=0.10`; it has
   0.23pp and 0.54pp gains at `alpha=0.05` and `alpha=0.15`. Its across-target
   average is far below the required 3pp.
2. Quality grouping is not uniformly stable across targets and seeds: the
   low-risk target has less than 1pp gain, while the `alpha=0.15` gain remains
   below 3pp. Small worst-condition risk-excess deteriorations also occur in
   some seed/target combinations.
3. Therefore the evidence does not justify a second backbone, an external
   dataset expansion, a learned descriptor, or any use of the locked TEST.

This is a useful negative result: global image-clustered CRC is a valid,
conservative baseline for this preregistered mixture, but there is insufficient
robust oracle headroom for the proposed degradation-aware deployment method.
