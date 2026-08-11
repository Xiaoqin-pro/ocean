# FT-Reliability real smoke audit — 2026-07-25

## Frozen implementation

- v1.2 safeguards and protocol closure: 8063703
- Final teacher-artifact fix: 3d07841
- Full test suite before smoke: **114 passed**

## Real technical smoke

| Run | Steps | Outcome | Log interval | Checkpoint SHA-256 |
| --- | ---: | --- | --- | --- |
| A (clean-only) | 50 | completed; finite loss | 23:26:22–23:26:41 | E1A8FC057D479AF508C64C7C34CEAE75EEA432C2842A554B514386A26AB1DF01 |
| B (three-view warm-up) | 50 | completed; finite losses | 23:23:52–23:24:17 | 1DB5080E98ED8D6931ADB5755DAD7175D3A3E5FD8F68204F4E95D68F9D6987A7 |

No NaN or Inf occurred. In the final B smoke batch, clean/s1/s3 cross-entropy
losses were finite (1.2388 / 1.2721 / 1.3979). The observed same-pixel
s1-correct to s3-wrong transition rate was approximately 5.36%. FT and CR
losses were zero by design during the five-epoch shared warm-up.

These observations establish technical executability only. They do not provide
a scientific efficacy result for TCCR.

## Data-access audit

- method_train: read for the authorized smoke runs.
- method_development: not read; only its frozen file hash was checked.
- Formal SUIM validation, calibration, and official TEST: not read.
- External data: not read.

## Interrupted formal A

The attempted formal A run was proactively stopped for a planned power
interruption. Its directory was quarantined as
outputs/aborted_runs/A_power_interruption_20260725_233559.
It is not an experiment result, is not a teacher, and must not be resumed.

## Next stable-power command

Run a new A training from epoch 1, without --resume:

~~~powershell
Set-Location D:\\111\\Desktop\\underwater-calibration
New-Item -ItemType Directory -Force logs\\ft_reliability_v1_2 | Out-Null
.\\venv\\Scripts\\python.exe .\\scripts\\train_ft_reliability_pilot.py --variant A --model segformer 2>&1 |
  Tee-Object logs\\ft_reliability_v1_2\\A_formal_console.log
~~~
