#!/usr/bin/env python3
"""Render Figure 1 as raw-observation boxplots.

The visual style follows the APT reference figure
`tex_ref/arXiv-2502.10328v4/dw4_manywell_log_z_plot.pdf`: wide horizontal
Matplotlib layout, bottom legend, Times typography, medium grid, black-edged
colored boxes, and visible whiskers/outliers.

Usage:
    python -m noisde.figures.main_result_overview
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import matplotlib

logging.getLogger("fontTools").setLevel(logging.ERROR)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter
import pandas as pd

from .style import strip_trailing_whitespace, typography_rc

METHODS = [
    ("deterministic", "Deterministic", "#ff1717"),
    ("fixed_noise", "Fixed noise", "#02c702"),
    ("short_remd_mbar", "Short REMD/MBAR", "#007000"),
    ("noisde", "NoiSDE", "#001bff"),
]

HELDOUT_BASIN_PAIRS = [
    "alphaR_to_PII",
    "PII_to_alphaR",
    "beta_to_alphaL",
    "alphaL_to_beta",
    "alphaL_to_PII",
    "PII_to_alphaL",
]

METRICS = [
    {
        "panel": "A",
        "panel_title": "Toy sampling",
        "metric": "ESS_N",
        "metric_label": r"ESS/$N$",
        "source_file": "data/processed/seed_level/toy_sampling_results.csv",
        "tier": "toy_sampling",
        "tasks": None,
        "variant_map": {
            "deterministic": "deterministic",
            "fixed_noise": "fixed_noise",
            "noisde": "pnedb",
        },
    },
    {
        "panel": "A",
        "panel_title": "Toy sampling",
        "metric": "C_mode",
        "metric_label": r"$C_{\mathrm{mode}}$",
        "source_file": "data/processed/seed_level/toy_sampling_results.csv",
        "tier": "toy_sampling",
        "tasks": None,
        "variant_map": {
            "deterministic": "deterministic",
            "fixed_noise": "fixed_noise",
            "noisde": "pnedb",
        },
    },
    {
        "panel": "B",
        "panel_title": "Alanine transport",
        "metric": "A_swap",
        "metric_label": r"$A_{\mathrm{swap}}$",
        "source_file": "data/processed/seed_level/alanine_transport_results.csv",
        "tier": "alanine_transport",
        "tasks": HELDOUT_BASIN_PAIRS,
        "variant_map": {
            "deterministic": "deterministic",
            "fixed_noise": "fixed_noise",
            "short_remd_mbar": "short_remd_mbar",
            "noisde": "pnedb",
        },
    },
    {
        "panel": "B",
        "panel_title": "Alanine transport",
        "metric": "R_high_E",
        "metric_label": r"$R_{\mathrm{high\!-\!E}}$",
        "source_file": "data/processed/seed_level/alanine_transport_results.csv",
        "tier": "alanine_transport",
        "tasks": HELDOUT_BASIN_PAIRS,
        "variant_map": {
            "deterministic": "deterministic",
            "fixed_noise": "fixed_noise",
            "short_remd_mbar": "short_remd_mbar",
            "noisde": "pnedb",
        },
    },
    {
        "panel": "C",
        "panel_title": "Free-energy estimation",
        "metric": "DeltaF_MAE",
        "metric_label": r"$\mathrm{MAE}(\Delta F)$",
        "source_file": "data/processed/seed_level/free_energy_estimation_results.csv",
        "tier": "free_energy",
        "tasks": None,
        "variant_map": {
            "deterministic": "deterministic_transport",
            "fixed_noise": "fixed_noise_transport",
            "short_remd_mbar": "short_mbar",
            "noisde": "pnedb",
        },
    },
    {
        "panel": "C",
        "panel_title": "Free-energy estimation",
        "metric": "CV2_w",
        "source_metric": "Var_w",
        "metric_label": r"$\mathrm{CV}^2(w)$",
        "source_file": "data/processed/seed_level/free_energy_estimation_results.csv",
        "tier": "free_energy",
        "tasks": None,
        "variant_map": {
            "deterministic": "deterministic_transport",
            "fixed_noise": "fixed_noise_transport",
            "short_remd_mbar": "short_mbar",
            "noisde": "pnedb",
        },
    },
]


def load_numeric_rows(root: Path, spec: dict) -> pd.DataFrame:
    df = pd.read_csv(root / spec["source_file"], low_memory=False)
    source_metric = spec.get("source_metric", spec["metric"])
    out = df[(df["tier"] == spec["tier"]) & (df["metric"] == source_metric)].copy()
    if spec["tasks"] is not None:
        out = out[out["task"].isin(spec["tasks"])].copy()
    out["value_numeric"] = pd.to_numeric(out["value"], errors="coerce")
    out = out[out["value_numeric"].notna()].copy()
    out = out[out["variant"].isin(spec["variant_map"].values())].copy()
    reverse_variant = {v: k for k, v in spec["variant_map"].items()}
    out["method"] = out["variant"].map(reverse_variant)
    out["panel"] = spec["panel"]
    out["panel_title"] = spec["panel_title"]
    out["display_metric"] = spec["metric"]
    out["metric_label"] = spec["metric_label"]
    out["source_file"] = spec["source_file"]
    return out


def build_rows(root: Path) -> list[dict]:
    method_labels = {key: label for key, label, _ in METHODS}
    rows: list[dict] = []

    for spec in METRICS:
        df = load_numeric_rows(root, spec)
        expected = set(spec["variant_map"])
        observed = set(df["method"].dropna())
        missing = expected - observed
        if missing:
            missing_labels = ", ".join(sorted(missing))
            print(f"warning: {spec['metric']} missing methods: {missing_labels}")

        for record in df.to_dict("records"):
            method = record["method"]
            rows.append({
                "panel": record["panel"],
                "panel_title": record["panel_title"],
                "metric": record["display_metric"],
                "metric_label": record["metric_label"],
                "method": method,
                "method_label": method_labels[method],
                "source_variant": record["variant"],
                "value": float(record["value_numeric"]),
                "source_file": record["source_file"],
                "tier": record["tier"],
                "task": record["task"],
                "seed": record["seed"],
                "run_id": record["run_id"],
                "unit": record.get("unit", ""),
                "config": record.get("config", ""),
            })
    return rows


def write_source_csv(rows: list[dict], outdir: Path) -> Path:
    path = outdir / "main_result_overview_source.csv"
    fieldnames = [
        "panel",
        "panel_title",
        "metric",
        "metric_label",
        "method",
        "method_label",
        "source_variant",
        "value",
        "source_file",
        "tier",
        "task",
        "seed",
        "run_id",
        "unit",
        "config",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_summary_csv(rows: list[dict], outdir: Path) -> Path:
    path = outdir / "main_result_overview_summary.csv"
    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["panel", "metric", "method", "method_label"], sort=False)["value"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    q = (
        df.groupby(["panel", "metric", "method"], sort=False)["value"]
        .quantile([0.25, 0.75])
        .unstack()
        .reset_index()
        .rename(columns={0.25: "q1", 0.75: "q3"})
    )
    summary = summary.merge(q, on=["panel", "metric", "method"], how="left")
    summary.to_csv(path, index=False)
    return path


def values_by_metric_method(rows: list[dict]) -> dict[tuple[str, str], list[float]]:
    out: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        out.setdefault((row["metric"], row["method"]), []).append(row["value"])
    return out


def strip_trailing_whitespace(path: Path) -> None:
    text = path.read_text()
    path.write_text("\n".join(line.rstrip(" \t") for line in text.splitlines()) + "\n")


TICK_CONFIG = {
    "ESS_N": {
        "ticks": [x / 100 for x in range(10, 51, 10)],
        "ylim": (0.05, 0.55),
        "decimals": 2,
    },
    "C_mode": {
        "ticks": [0.40, 0.55, 0.70, 0.85, 1.00],
        "ylim": (0.40, 1.00),
        "decimals": 2,
    },
    "A_swap": {
        "ticks": [0.00, 0.10, 0.20, 0.30, 0.40],
        "ylim": (0.00, 0.40),
        "decimals": 2,
    },
    "R_high_E": {
        "ticks": [x / 100 for x in range(0, 21, 5)],
        "ylim": (0.00, 0.20),
        "decimals": 2,
    },
    "DeltaF_MAE": {
        "ticks": [0.00, 0.25, 0.50, 0.75, 1.00],
        "ylim": (0.00, 1.00),
        "decimals": 2,
    },
    "CV2_w": {
        "ticks": [0.50, 1.50, 2.50, 3.50, 4.50],
        "ylim": (0.50, 4.50),
        "decimals": 1,
    },
}

RIGHT_TICK_METRICS = {"C_mode", "R_high_E", "CV2_w"}
X_MINOR_TICKS = [0.5, 1.5, 2.5, 3.5, 4.5]


def midpoint_ticks(ticks: list[float]) -> list[float]:
    return [(left + right) / 2 for left, right in zip(ticks, ticks[1:])]


def tick_label(metric: str, value: float, _pos: int | None = None) -> str:
    decimals = TICK_CONFIG[metric]["decimals"]
    return f"{value:.{decimals}f}"


def render(
    rows: list[dict],
    outdir: Path,
    stats_dir: Path | None = None,
    font_preset: str = "times",
) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    stats_dir = outdir if stats_dir is None else stats_dir
    stats_dir.mkdir(parents=True, exist_ok=True)
    written = [write_source_csv(rows, outdir), write_summary_csv(rows, stats_dir)]

    plt.rcParams.update({
        **typography_rc(7.6, font_preset),
        "axes.linewidth": 0.55,
        "axes.edgecolor": "black",
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    method_colors = {key: color for key, _, color in METHODS}
    method_positions = {key: idx + 1 for idx, (key, _, _) in enumerate(METHODS)}
    metric_values = values_by_metric_method(rows)
    fig = plt.figure(figsize=(10.8, 2.80))
    outer = fig.add_gridspec(
        1,
        3,
        left=0.045,
        right=0.99,
        bottom=0.150,
        top=0.715,
        wspace=0.22,
    )
    panel_order = [
        ("A", "Toy sampling"),
        ("B", "Alanine transport"),
        ("C", "Free-energy estimation"),
    ]
    panel_axes: dict[str, list[plt.Axes]] = {}
    axes_by_metric: dict[str, plt.Axes] = {}
    for panel_idx, (panel_key, _panel_title) in enumerate(panel_order):
        inner = outer[panel_idx].subgridspec(1, 2, wspace=0.10)
        panel_specs = [spec for spec in METRICS if spec["panel"] == panel_key]
        panel_axes[panel_key] = []
        for metric_idx, spec in enumerate(panel_specs):
            ax = fig.add_subplot(inner[0, metric_idx])
            panel_axes[panel_key].append(ax)
            axes_by_metric[spec["metric"]] = ax

    for spec in METRICS:
        ax = axes_by_metric[spec["metric"]]
        available_methods = [
            method
            for method, _, _ in METHODS
            if metric_values.get((spec["metric"], method))
        ]
        all_values = [
            value
            for method in available_methods
            for value in metric_values.get((spec["metric"], method), [])
        ]
        y_low = min(all_values)
        y_high = max(all_values)
        y_pad = max((y_high - y_low) * 0.18, y_high * 0.035, 0.008)
        tick_config = TICK_CONFIG[spec["metric"]]
        y_ticks = tick_config["ticks"]
        y_min = min(tick_config["ylim"][0], max(0.0, y_low - y_pad))
        y_max = max(tick_config["ylim"][1], y_high + y_pad)

        ax.set_xlim(0.35, len(METHODS) + 0.65)
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_locator(FixedLocator(y_ticks))
        ax.yaxis.set_minor_locator(FixedLocator(midpoint_ticks(y_ticks)))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, pos, metric=spec["metric"]: tick_label(metric, value, pos))
        )
        ax.set_xticks(list(method_positions.values()))
        ax.set_xticklabels([""] * len(METHODS))
        ax.xaxis.set_minor_locator(FixedLocator(X_MINOR_TICKS))
        ax.tick_params(axis="x", which="both", length=0)
        if spec["metric"] in RIGHT_TICK_METRICS:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.tick_params(
                axis="y",
                labelsize=7.2,
                length=2.5,
                pad=1.5,
                labelleft=False,
                labelright=True,
            )
        else:
            ax.tick_params(
                axis="y",
                labelsize=7.2,
                length=2.5,
                pad=1.5,
                labelleft=True,
                labelright=False,
            )
        ax.tick_params(axis="y", which="minor", length=0)
        ax.grid(True, which="major", color="#858585", alpha=0.48, linewidth=0.36)
        ax.grid(True, which="minor", color="#858585", alpha=0.48, linewidth=0.36)
        ax.set_axisbelow(True)
        ax.set_title(spec["metric_label"], fontsize=8.8, pad=5)
        for spine in ax.spines.values():
            spine.set_color("black")

        for method in available_methods:
            values = metric_values.get((spec["metric"], method), [])
            if not values:
                continue
            parts = ax.boxplot(
                [values],
                positions=[method_positions[method]],
                widths=0.56,
                patch_artist=True,
                showfliers=True,
                whis=1.5,
                manage_ticks=False,
                boxprops={"edgecolor": "black", "linewidth": 0.75},
                whiskerprops={"color": "#777777", "linewidth": 0.55},
                capprops={"color": "#777777", "linewidth": 0.55},
                medianprops={"color": "#e08b2f", "linewidth": 0.75},
                flierprops={
                    "marker": "o",
                    "markersize": 1.45,
                    "markerfacecolor": "white",
                    "markeredgecolor": "#666666",
                    "markeredgewidth": 0.45,
                    "alpha": 0.9,
                },
            )
            for box in parts["boxes"]:
                box.set_facecolor(method_colors[method])

        ax.set_xticks(list(method_positions.values()))
        ax.set_xticklabels([""] * len(METHODS))

    handles = [
        Patch(facecolor=color, edgecolor="black", linewidth=0.75, label=label)
        for _, label, color in METHODS
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=True,
        fontsize=8.1,
        bbox_to_anchor=(0.5, 0.016),
        columnspacing=1.66,
        handlelength=1.44,
        handletextpad=0.44,
        borderpad=0.28,
    )
    legend.get_frame().set_edgecolor("#e0e0e0")
    legend.get_frame().set_linewidth(0.6)
    legend.get_frame().set_facecolor("white")

    for panel_key, panel_title in panel_order:
        positions = [ax.get_position() for ax in panel_axes[panel_key]]
        x0 = min(pos.x0 for pos in positions)
        x1 = max(pos.x1 for pos in positions)
        fig.text(
            (x0 + x1) / 2,
            0.825,
            f"{panel_key}  {panel_title}",
            ha="center",
            va="center",
            fontsize=9.6,
        )

    for ext in ("pdf", "svg"):
        path = outdir / f"main_result_overview.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        if ext == "svg":
            strip_trailing_whitespace(path)
        written.append(path)
    plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument(
        "--outdir",
        default="results/figures/main_result_overview",
        help="Output directory for Figure 1 assets.",
    )
    parser.add_argument(
        "--font-preset",
        choices=["sans", "times"],
        default="times",
        help="Typography preset for figure text.",
    )
    parser.add_argument(
        "--stats-dir",
        default="results/statistics/figures",
        help="Directory for figure summary CSV files.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    stats_dir = Path(args.stats_dir)
    written = render(
        build_rows(root),
        outdir if outdir.is_absolute() else root / outdir,
        stats_dir if stats_dir.is_absolute() else root / stats_dir,
        font_preset=args.font_preset,
    )
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
