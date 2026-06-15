"""Writers for manuscript-compatible long-format tables."""
from __future__ import annotations

import csv
from pathlib import Path


SEED_LEVEL_COLUMNS = ["tier", "task", "variant", "seed", "run_id", "metric", "value", "unit", "config"]
RUN_INDEX_COLUMNS = ["tier", "task", "variant", "seed", "run_id", "config"]


def write_seed_level_results(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_LEVEL_COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_run_index(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dedup: dict[str, dict] = {}
    for row in rows:
        dedup[str(row["run_id"])] = row
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_INDEX_COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(dedup.values())
    return path
