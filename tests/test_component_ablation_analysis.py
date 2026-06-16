from __future__ import annotations

import csv
from pathlib import Path

import pytest

from noisde.analysis import (
    ContrastSpec,
    summarize_component_ablation,
    write_component_ablation_outputs,
)


def row(
    tier: str,
    task: str,
    variant: str,
    seed: str,
    value: str,
    run_id: str,
) -> dict[str, str]:
    return {
        "tier": tier,
        "task": task,
        "variant": variant,
        "seed": seed,
        "run_id": run_id,
        "metric": "Ubar",
        "value": value,
        "unit": "score",
        "config": f"{variant}.yaml",
    }


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tier",
                "task",
                "variant",
                "seed",
                "run_id",
                "metric",
                "value",
                "unit",
                "config",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_component_ablation_summary_collapses_units_and_balances_families(tmp_path: Path) -> None:
    path = tmp_path / "component_ablation.csv"
    rows = [
        row("core_ablation", "T1", "A4", "R001", "0.9", "1"),
        row("core_ablation", "T1", "A4", "R001", "1.0", "1_dup"),
        row("core_ablation", "T1", "B1", "R001", "0.7", "2"),
        row("core_ablation", "T2", "A4", "R001", "0.2", "3"),
        row("core_ablation", "T2", "B1", "R001", "0.1", "4"),
        row("other", "T2", "A4", "R001", "9.0", "5"),
    ]
    write_rows(path, rows)

    summary = summarize_component_ablation(
        path,
        variants=["A4", "B1"],
        contrasts=[ContrastSpec("A4", "B1")],
    )

    by_variant = {row["variant"]: row for row in summary.summary_rows}
    assert float(by_variant["A4"]["family_balanced_mean"]) == pytest.approx(0.575)
    assert by_variant["A4"]["unit_count"] == "2"
    assert by_variant["A4"]["family_count"] == "2"

    assert len(summary.contrast_rows) == 1
    contrast = summary.contrast_rows[0]
    assert contrast["left_variant"] == "A4"
    assert contrast["right_variant"] == "B1"
    assert contrast["paired_unit_count"] == "2"
    assert contrast["positive_unit_count"] == "2"
    assert float(contrast["family_balanced_delta"]) == pytest.approx(0.175)


def test_component_ablation_output_writer_uses_stable_csv_names(tmp_path: Path) -> None:
    path = tmp_path / "component_ablation.csv"
    write_rows(
        path,
        [
            row("core_ablation", "T1", "A4", "R001", "1.0", "1"),
            row("core_ablation", "T1", "B1", "R001", "0.5", "2"),
        ],
    )
    summary = summarize_component_ablation(path, contrasts=[ContrastSpec("A4", "B1")])

    summary_path, contrast_path = write_component_ablation_outputs(summary, tmp_path / "out")

    assert summary_path.name == "component_ablation_family_summary.csv"
    assert contrast_path.name == "component_ablation_family_contrasts.csv"
    assert summary_path.read_text(encoding="utf-8").splitlines()[0].startswith("tier,metric,variant")
    assert contrast_path.read_text(encoding="utf-8").splitlines()[0].startswith("tier,metric,left_variant")
