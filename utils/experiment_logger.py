from __future__ import annotations

import csv
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "experiment_id", "timestamp", "git_commit", "model", "dataset_split", "seed",
    "image_size", "batch_size", "epochs", "learning_rate", "best_epoch", "val_miou",
    "test_miou", "ece", "nll", "brier", "aurc", "checkpoint_path", "notes",
]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def append_experiment(record: dict[str, Any], path: Path = PROJECT_ROOT / "experiment_log.csv") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {field: "" for field in FIELDS}
    row.update(record)
    row.setdefault("timestamp", datetime.now().astimezone().isoformat(timespec="seconds"))
    row.setdefault("git_commit", git_commit())
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
