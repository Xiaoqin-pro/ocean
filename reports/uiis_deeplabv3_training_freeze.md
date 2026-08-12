# UIIS DeepLabV3-MobileNetV3-Large Training Freeze

## Fixed protocol

- Dataset protocol: `UIIS_alpha010_confirmation_v1`.
- Training split: 2,371 images only.
- Calibration and confirmation access during training: false.
- Architecture: DeepLabV3-MobileNetV3-Large with `IMAGENET1K_V2` backbone initialization.
- Input size: 384.
- Batch size: 4.
- Optimizer: AdamW, learning rate `1e-4`, weight decay `1e-4`.
- AMP: enabled.
- Epochs: fixed at 60.
- Checkpoint selection: final epoch only; no validation-based model selection.

## Completion verification

- Completed epoch: 60.
- Global step: 35,580.
- Final train loss: `0.2259635199`.
- Training history rows: 60.
- Non-finite or missing training losses: 0.
- Final checkpoint loads successfully.
- Checkpoint format: `uiis_deeplabv3_fixed_protocol_v1`.
- `official_suim_test_evaluated`: false.
- `calibration_evaluated`: false.
- `confirmation_evaluated`: false.

## Provenance

- Checkpoint: `outputs/deeplabv3_mobilenetv3_uiis_alpha010_confirmation/checkpoints/last.pt`.
- Checkpoint SHA-256: `38EB9EE1AAE4BC28B196017C17E726EE91FE63457FD845A64D6179D3F75FED67`.
- Configuration: `configs/uiis/deeplabv3_mobilenetv3_alpha010_confirmation.yaml`.
- Configuration SHA-256: `89693B3E26699E78379B6B106AE8C265C3E8616687DBB0013D7206135B2D276A`.

This record freezes training only. Calibration, confirmation, temperature scaling, ranking, boundary analysis, CRC sensitivity, and SUIM official TEST evaluation remain outside this stage.
