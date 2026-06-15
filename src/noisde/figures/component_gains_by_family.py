#!/usr/bin/env python3
"""Render Figure 3 candidate: component gains across benchmark families.

The layout borrows the compact horizontal line-figure grammar of
`preview_main_results/figures_pdf/fig1_noise_family.pdf`, but typography,
palette, legend placement, and grid treatment are aligned with
`results/figures/main_result_overview/main_result_overview.pdf`. The data are
the component-ablation seed-level results, not preview data.

Usage:
    python -m noisde.figures.component_gains_by_family
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

logging.getLogger("fontTools").setLevel(logging.ERROR)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
import pandas as pd

from .style import strip_trailing_whitespace, typography_rc

SOURCE_FILE = "data/processed/seed_level/component_ablation_results.csv"
TIER = "core_ablation"

TASKS = [
    ("T1_double_well", "Double"),
    ("T2_four_well", "Four"),
    ("T3_ring", "Ring"),
    ("alanine_transport", "Ala.\ntrans."),
    ("alanine_temperature_fe", "Ala.\ntemp. FE"),
    ("lj_lambda_fe", "LJ-$\\lambda$\nFE"),
]

METHODS = [
    ("F0", "F0 det.", "#ff1717", (0, (1.2, 1.6)), 1.62, 0.075),
    ("A2", "A2 fixed", "#02c702", (0, (3.2, 2.2)), 1.62, 0.075),
    ("A3", "A3 iso.", "#007000", (0, (5.0, 2.0, 1.2, 2.0)), 1.62, 0.075),
    ("B1", "B1 learned", "#666666", (0, (6.0, 2.2)), 1.62, 0.075),
    ("A4", "A4 NoiSDE", "#001bff", "solid", 1.62, 0.075),
    ("A5", "A5 over-noise", "#e08b2f", (0, (1.0, 1.5)), 1.62, 0.075),
]

PANELS = [
    {
        "key": "A",
        "title": r"A. Utility",
        "metric": "Ubar",
        "ylabel": r"$\bar{\mathcal{U}}_\omega$",
        "ylim": (0.15, 0.90),
        "yticks": [0.20, 0.35, 0.50, 0.65, 0.80],
        "decimals": 2,
    },
    {
        "key": "B",
        "title": r"B. Swap acceptance",
        "metric": "A_swap",
        "ylabel": r"$A_{\mathrm{swap}}$",
        "ylim": (0.00, 0.40),
        "yticks": [0.00, 0.10, 0.20, 0.30, 0.40],
        "decimals": 2,
    },
    {
        "key": "C",
        "title": r"C. High-energy access",
        "metric": "R_high_E",
        "ylabel": r"$R_{\mathrm{high\!-\!E}}$",
        "ylim": (0.00, 0.40),
        "yticks": [0.00, 0.10, 0.20, 0.30, 0.40],
        "decimals": 2,
    },
]

T_CRIT_95 = {
    9: 2.2621571627409915,
    19: 2.093024054408263,
}
X_STEP = 1.00


def midpoint_ticks(ticks: list[float]) -> list[float]:
    return [(left + right) / 2 for left, right in zip(ticks, ticks[1:])]


def load_source(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / SOURCE_FILE, low_memory=False)
    df = df[df["tier"] == TIER].copy()
    df = df[df["task"].isin([task for task, _ in TASKS])].copy()
    df = df[df["variant"].isin([method for method, *_ in METHODS])].copy()
    df = df[df["metric"].isin([panel["metric"] for panel in PANELS])].copy()
    df["value_numeric"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[df["value_numeric"].notna()].copy()
    return df


def t_critical_95(count: int) -> float:
    df = count - 1
    return T_CRIT_95.get(df, 1.96)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["task", "variant", "metric"], sort=False)["value_numeric"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    summary["sem"] = summary["std"] / summary["count"].pow(0.5)
    summary["ci95_half_width"] = summary.apply(
        lambda row: t_critical_95(int(row["count"])) * row["sem"],
        axis=1,
    )
    summary["ci95_low"] = summary["mean"] - summary["ci95_half_width"]
    summary["ci95_high"] = summary["mean"] + summary["ci95_half_width"]
    return summary


def write_source_csv(df: pd.DataFrame, outdir: Path) -> Path:
    path = outdir / "component_gains_by_family_source.csv"
    columns = [
        "tier",
        "task",
        "variant",
        "seed",
        "run_id",
        "metric",
        "value",
        "unit",
        "config",
    ]
    df[columns].to_csv(path, index=False, lineterminator="\n")
    return path


def write_summary_csv(summary: pd.DataFrame, outdir: Path) -> Path:
    path = outdir / "component_gains_by_family_summary.csv"
    task_labels = dict(TASKS)
    method_labels = {method: label for method, label, *_ in METHODS}
    out = summary.copy()
    out["task_label"] = out["task"].map(task_labels)
    out["method_label"] = out["variant"].map(method_labels)
    columns = [
        "task",
        "task_label",
        "variant",
        "method_label",
        "metric",
        "count",
        "mean",
        "std",
        "ci95_low",
        "ci95_high",
        "min",
        "max",
    ]
    out[columns].to_csv(path, index=False, lineterminator="\n")
    return path


def stat_lookup(summary: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, float]]:
    rows: dict[tuple[str, str, str], dict[str, float]] = {}
    for record in summary.to_dict("records"):
        rows[(record["task"], record["variant"], record["metric"])] = {
            "mean": float(record["mean"]),
            "low": float(record["ci95_low"]),
            "high": float(record["ci95_high"]),
        }
    return rows


def render(summary: pd.DataFrame, outdir: Path, font_preset: str = "times") -> list[Path]:
    matplotlib.rcParams.update({
        **typography_rc(7.6, font_preset),
        "axes.linewidth": 0.55,
        "axes.edgecolor": "black",
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    lookup = stat_lookup(summary)
    center = (len(TASKS) - 1) / 2
    x = [(idx - center) * X_STEP + center for idx in range(len(TASKS))]
    xlabels = [label for _, label in TASKS]

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.35), constrained_layout=False)
    fig.subplots_adjust(left=0.052, right=0.990, top=0.790, bottom=0.278, wspace=0.250)

    for ax, panel in zip(axes, PANELS):
        metric = panel["metric"]
        for method, _, color, linestyle, linewidth, alpha in METHODS:
            means: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for task, _ in TASKS:
                stats = lookup[(task, method, metric)]
                means.append(stats["mean"])
                lows.append(stats["low"])
                highs.append(stats["high"])

            ax.fill_between(x, lows, highs, color=color, alpha=alpha, linewidth=0, zorder=1)
            ax.plot(
                x,
                means,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                marker="o",
                markersize=3.45,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=0.0,
                zorder=3,
            )

        ax.set_title(panel["title"], fontsize=9.4, pad=6.0)
        ax.set_ylabel(panel["ylabel"], fontsize=8.4, labelpad=4)
        ax.set_xlim(-0.22, len(TASKS) - 0.78)
        ax.set_ylim(*panel["ylim"])
        ax.set_yticks(panel["yticks"])
        ax.yaxis.set_minor_locator(FixedLocator(midpoint_ticks(panel["yticks"])))
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=7.2, linespacing=1.02)
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        ax.tick_params(axis="x", pad=3.2, length=2.5, width=0.55)
        ax.tick_params(axis="y", labelsize=7.2, pad=1.5, length=2.5, width=0.55)
        ax.tick_params(axis="y", which="minor", length=0)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:.{panel['decimals']}f}")
        )
        ax.grid(True, which="major", color="#858585", alpha=0.48, linewidth=0.36)
        ax.grid(True, which="minor", axis="y", color="#858585", alpha=0.48, linewidth=0.36)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
            spine.set_color("black")

    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker="o",
            markersize=3.45,
            label=label,
        )
        for method, label, color, linestyle, linewidth, _ in METHODS
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        frameon=True,
        fontsize=8.1,
        handlelength=2.00,
        handletextpad=0.40,
        columnspacing=1.18,
        borderpad=0.28,
        bbox_to_anchor=(0.5, 0.100),
    )
    legend.get_frame().set_edgecolor("#e0e0e0")
    legend.get_frame().set_linewidth(0.6)
    legend.get_frame().set_facecolor("white")

    outputs = [
        outdir / "component_gains_by_family.pdf",
        outdir / "component_gains_by_family.svg",
    ]
    for path in outputs:
        if path.suffix == ".png":
            fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
        else:
            fig.savefig(path, facecolor="white", bbox_inches="tight")
            if path.suffix == ".svg":
                strip_trailing_whitespace(path)
    plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/figures/component_gains_by_family"),
        help="Output directory for component-gain figure assets.",
    )
    parser.add_argument(
        "--font-preset",
        choices=["sans", "times"],
        default="times",
        help="Typography preset for figure text.",
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=Path("results/statistics/figures"),
        help="Directory for figure summary CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    outdir = args.outdir if args.outdir.is_absolute() else root / args.outdir
    stats_dir = args.stats_dir if args.stats_dir.is_absolute() else root / args.stats_dir
    outdir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    source = load_source(root)
    summary = summarize(source)
    write_source_csv(source, outdir)
    write_summary_csv(summary, stats_dir)
    render(summary, outdir, font_preset=args.font_preset)


if __name__ == "__main__":
    main()
