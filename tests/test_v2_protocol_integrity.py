from pathlib import Path

import pytest

from scripts.validate_v2_protocol import validate_v2_protocol


def test_committed_v2_protocol_integrity() -> None:
    root = Path(__file__).resolve().parents[1]
    if not (root / "data" / "suim_processed" / "manifest.csv").is_file():
        pytest.skip("SUIM data are intentionally not bundled with the repository.")
    result = validate_v2_protocol(root)
    assert result["exact_sha_leakage"] is False
    assert result["excluded_samples"] >= 59
