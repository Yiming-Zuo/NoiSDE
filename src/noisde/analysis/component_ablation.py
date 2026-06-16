"""Family-balanced summaries for component-ablation result tables."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


SUMMARY_COLUMNS = [
    "tier",
    "metric",
    "variant",
    "family_count",
    "unit_count",
    "family_balanced_mean",
    "min_family_mean",
    "max_family_mean",
]

CONTRAST_COLUMNS = [
    "tier",
    "metric",
    "left_variant",
    "right_variant",
    "family_count",
    "paired_unit_count",
    "family_balanced_delta",
    "positive_unit_count",
    "negative_unit_count",
    "zero_unit_count",
]


@dataclass(frozen=True)
class ContrastSpec:
    left: str
    right: str

    @classmethod
    def parse(cls, value: str) -> "ContrastSpec":
        for separator in ("--", ":"):
            if separator in value:
                left, right = value.split(separator, 1)
                left = left.strip()
                right = right.strip()
                if left and right:
                    return cls(left=left, right=right)
        raise ValueError(f"Contrast must use LEFT:RIGHT or LEFT--RIGHT: {value}")


@dataclass(frozen=True)
class UnitValue:
    tier: str
    task: str
    variant: str
    seed: str
    metric: str
    value: float


@dataclass(frozen=True)
class ComponentAblationSummary:
    summary_rows: list[dict[str, str]]
    contrast_rows: list[dict[str, str]]


def _format_float(value: float) -> str:
    return f"{value:.10g}"


def _load_units(
    input_path: Path,
    *,
    tier: str,
    metric: str,
    variants: set[str] | None,
) -> list[UnitValue]:
    grouped: dict[tuple[str, str, str, str, str], list[float]] = {}
    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"tier", "task", "variant", "seed", "metric", "value"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Missing required columns in {input_path}: {names}")
        for row in reader:
            if row["tier"] != tier or row["metric"] != metric:
                continue
            if variants is not None and row["variant"] not in variants:
                continue
            try:
                value = float(row["value"])
            except ValueError:
                continue
            key = (row["tier"], row["task"], row["variant"], row["seed"], row["metric"])
            grouped.setdefault(key, []).append(value)

    units = [
        UnitValue(
            tier=item[0],
            task=item[1],
            variant=item[2],
            seed=item[3],
            metric=item[4],
            value=mean(values),
        )
        for item, values in grouped.items()
    ]
    units.sort(key=lambda item: (item.task, item.variant, item.seed, item.metric))
    return units


def _summary_rows(units: Iterable[UnitValue], *, tier: str, metric: str) -> list[dict[str, str]]:
    by_variant_task: dict[tuple[str, str], list[float]] = {}
    unit_counts: dict[str, int] = {}
    for unit in units:
        by_variant_task.setdefault((unit.variant, unit.task), []).append(unit.value)
        unit_counts[unit.variant] = unit_counts.get(unit.variant, 0) + 1

    by_variant: dict[str, list[float]] = {}
    for (variant, _task), values in by_variant_task.items():
        by_variant.setdefault(variant, []).append(mean(values))

    rows = []
    for variant in sorted(by_variant):
        family_means = by_variant[variant]
        rows.append({
            "tier": tier,
            "metric": metric,
            "variant": variant,
            "family_count": str(len(family_means)),
            "unit_count": str(unit_counts[variant]),
            "family_balanced_mean": _format_float(mean(family_means)),
            "min_family_mean": _format_float(min(family_means)),
            "max_family_mean": _format_float(max(family_means)),
        })
    return rows


def _contrast_rows(
    units: Iterable[UnitValue],
    *,
    tier: str,
    metric: str,
    contrasts: Iterable[ContrastSpec],
) -> list[dict[str, str]]:
    values = {(unit.task, unit.seed, unit.variant): unit.value for unit in units}
    tasks = sorted({unit.task for unit in units})
    seeds_by_task = {
        task: sorted({unit.seed for unit in units if unit.task == task})
        for task in tasks
    }

    rows = []
    for contrast in contrasts:
        family_deltas: list[float] = []
        all_deltas: list[float] = []
        positive = negative = zero = 0
        for task in tasks:
            task_deltas = []
            for seed in seeds_by_task[task]:
                left = values.get((task, seed, contrast.left))
                right = values.get((task, seed, contrast.right))
                if left is None or right is None:
                    continue
                delta = left - right
                task_deltas.append(delta)
                all_deltas.append(delta)
                if delta > 0:
                    positive += 1
                elif delta < 0:
                    negative += 1
                else:
                    zero += 1
            if task_deltas:
                family_deltas.append(mean(task_deltas))
        if not family_deltas:
            continue
        rows.append({
            "tier": tier,
            "metric": metric,
            "left_variant": contrast.left,
            "right_variant": contrast.right,
            "family_count": str(len(family_deltas)),
            "paired_unit_count": str(len(all_deltas)),
            "family_balanced_delta": _format_float(mean(family_deltas)),
            "positive_unit_count": str(positive),
            "negative_unit_count": str(negative),
            "zero_unit_count": str(zero),
        })
    return rows


def summarize_component_ablation(
    input_path: str | Path,
    *,
    tier: str = "core_ablation",
    metric: str = "Ubar",
    variants: Iterable[str] | None = None,
    contrasts: Iterable[ContrastSpec] = (),
) -> ComponentAblationSummary:
    variant_set = set(variants) if variants is not None else None
    units = _load_units(Path(input_path), tier=tier, metric=metric, variants=variant_set)
    return ComponentAblationSummary(
        summary_rows=_summary_rows(units, tier=tier, metric=metric),
        contrast_rows=_contrast_rows(units, tier=tier, metric=metric, contrasts=contrasts),
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_component_ablation_outputs(
    summary: ComponentAblationSummary,
    outdir: str | Path,
) -> tuple[Path, Path]:
    outdir = Path(outdir)
    summary_path = _write_csv(
        outdir / "component_ablation_family_summary.csv",
        SUMMARY_COLUMNS,
        summary.summary_rows,
    )
    contrast_path = _write_csv(
        outdir / "component_ablation_family_contrasts.csv",
        CONTRAST_COLUMNS,
        summary.contrast_rows,
    )
    return summary_path, contrast_path
