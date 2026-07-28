# RCR Gate R1a: Frozen Uncompressed Local-reinference Oracle Result

## Decision

**FAIL — R1b and all RCR method development are closed.**  R1a used the
pre-registered DeepLab local / SegFormer remote pair and the fixed GT-region
oracle protocol: 8-connected local-error components, tight core boxes, 32 px
context, transitive merging of touching contexts, uncompressed current-image
RGB crops, and unconditional replacement of all core-box pixels.

Some accuracy signals were positive, but the complete frozen gate did not
pass:

| R1a criterion | Result | Status |
| --- | ---: | --- |
| Macro mIoU gain | +3.90pp | pass |
| R0 oracle-gain recovery | 46.01% | pass |
| Positive degradation families | 4 | pass |
| Scene bootstrap 95% lower bound | -0.56pp | fail |
| Damage / repair pixels | 51.56% | fail (limit: <50%) |
| Mean context-area ratio | 81.57% | fail (limit: <=75%) |

The selected core boxes did recover local errors in aggregate, but that gain is
not scene-stable under the fixed scene-cluster bootstrap.  More importantly,
the merged 32 px contexts cover most of an image on average and the
unconditional replacement damages enough local-correct pixels to violate the
fixed safety rule.  This does not support a meaningful local-evidence system;
it largely demonstrates that broadly re-observing the image can sometimes
help.

## Access and provenance

- R0 result commit: `85fe7e102f739ef012f150c5f1173ecbb09b94fa`.
- R1a implementation commit at inference: `3a1c82e`.
- R1a output configuration SHA-256:
  `58CBB00865E10B1510F03F9C1B9BC4E173725E019E00252E4DF2B1497ADEF065`.
- Only the already-open 231-image development role was read.
- No model was trained; no codec, byte budget, or communication simulation was
  run.
- Calibration, validation, official SUIM test, and external data were not
  accessed.

The local output files and their SHA-256 manifest remain under
`outputs/rcr_oracle_r1a/` and are not committed.

## Consequence

RCR is permanently closed under the pre-registered decision tree.  Do not
try additional context widths, component filtering, overlap rules, codecs,
confidence gates, crop-quality settings, or a repairability selector on these
assets.  TCCR, DTH, and RCR have now each received their prescribed low-cost
or oracle test; none has justified a new method-development phase.
