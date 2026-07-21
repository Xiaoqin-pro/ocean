# SUIM v2 exact-deduplication split (provisional)

This protocol keeps the official TEST partition immutable. Development samples with an exact SHA-256 image duplicate in official TEST are excluded. The unresolved 55-pixel mask-height repairs and exact duplicate development images with conflicting processed masks are excluded by default.

Remaining exact-image groups are assigned as indivisible units to train, validation, or calibration. This report does not certify scene-level isolation: it must be superseded by the reviewed near-duplicate and scene-group protocol before formal training.

- Seed: `20260721`
- Development before exclusion: 1525
- Development after exclusion: 1466
- Excluded: 59
- train: 1172 (`a36cad902777480da7ffa219bad4e7b432ed574a78e0a1261db9569654d15e6a`)
- val: 147 (`0609c922f3c2b79340e8c184959ab3c43f8dc1c98c571ddc85ba72d77034b383`)
- calibration: 147 (`ed3dad04f1315466a62c9692723a0cf08b6358ea8057e89d589850f753738402`)
- test: 110 (`95c54d05ca54a62636991c2bbb1658599e0ff5326f1851ad526bc2d245d4b5d8`)

See the data-quality reports for exclusions, duplicate-mask consistency, manual quantization review, and the pending scene-group review.
