import pandas as pd
import pytest

from scripts.create_grouped_v2_split import validate_group_isolation


def test_group_isolation_accepts_disjoint_groups() -> None:
    validate_group_isolation({
        "train": pd.DataFrame({"group_id": ["a", "a", "b"]}),
        "val": pd.DataFrame({"group_id": ["c"]}),
        "calibration": pd.DataFrame({"group_id": ["d"]}),
    })


def test_group_isolation_rejects_cross_split_group() -> None:
    with pytest.raises(ValueError, match="leaks"):
        validate_group_isolation({
            "train": pd.DataFrame({"group_id": ["a"]}),
            "val": pd.DataFrame({"group_id": ["a"]}),
        })
