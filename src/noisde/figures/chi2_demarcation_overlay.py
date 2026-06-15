#!/usr/bin/env python3
"""Render the chi2 demarcation diagnostic from raw observables.

The figure follows the diagnostic-distribution form of the APT DW4 ablation
reference: each panel overlays raw observable densities for all checkpoints,
the same-chi2 decile, the chi2-selected endpoints, and the utility-selected
endpoints.

Usage:
    python -m noisde.figures.chi2_demarcation_overlay
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

logging.getLogger("fontTools").setLevel(logging.ERROR)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd

from .style import strip_trailing_whitespace, typography_rc

SOURCE_FILE = "data/external/raw_observables_chi2_demarcation.csv"
SOURCE_TIER = "chi2_demarcation"
SOURCE_TASKS = {"T2_four_well", "T3_ring"}
SELECTED_DECILE = "D02"
N_BINS = 112
CHUNK_SIZE = 750_000

GROUPS = [
    ("all", "all checkpoints"),
    ("decile", r"same $\chi^2$ decile D02"),
    ("chi2", r"$\chi^2$-selected"),
    ("utility", "utility-selected"),
]

COLORS = {
    "all": "#000000",
    "decile": "#001bff",
    "chi2": "#ff1717",
    "utility": "#007000",
}

LINEWIDTHS = {
    "all": 1.00,
    "decile": 1.25,
    "chi2": 1.25,
    "utility": 1.25,
}

ALPHAS = {
    "all": 0.55,
    "decile": 0.82,
    "chi2": 0.82,
    "utility": 0.82,
}


@dataclass(frozen=True)
class Panel:
    key: str
    title: str
    observable: str
    xlabel: str
    xlim: tuple[float, float]
    xticks: tuple[float, ...]
    decimals: int


PANELS = [
    Panel(
        key="A",
        title="Endpoint chi2",
        observable="chi2_contribution",
        xlabel=r"Endpoint $\chi^2$ contribution",
        xlim=(0.0, 6.0),
        xticks=(0.0, 1.5, 3.0, 4.5, 6.0),
        decimals=1,
    ),
    Panel(
        key="B",
        title="Mode residual",
        observable="mode_residual_normalized",
        xlabel=r"Mode residual",
        xlim=(0.0, 1.0),
        xticks=(0.0, 0.25, 0.50, 0.75, 1.00),
        decimals=2,
    ),
    Panel(
        key="C",
        title="Swap log acceptance",
        observable="swap_log_alpha",
        xlabel=r"Swap log acceptance",
        xlim=(-3.0, 1.2),
        xticks=(-3.0, -2.0, -1.0, 0.0, 1.0),
        decimals=1,
    ),
    Panel(
        key="D",
        title="High-energy margin",
        observable="energy_minus_high_threshold",
        xlabel=r"Energy above threshold",
        xlim=(-6.0, 2.5),
        xticks=(-6.0, -4.0, -2.0, 0.0, 2.0),
        decimals=1,
    ),
]


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def midpoint_ticks(ticks: tuple[float, ...] | list[float]) -> list[float]:
    return [(left + right) / 2 for left, right in zip(ticks, ticks[1:])]


def step_xy(edges: np.ndarray, density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.repeat(edges, 2)[1:-1], np.repeat(density, 2)


def panel_edges() -> dict[str, np.ndarray]:
    return {
        panel.observable: np.linspace(panel.xlim[0], panel.xlim[1], N_BINS + 1)
        for panel in PANELS
    }


def empty_histograms() -> dict[str, dict[str, np.ndarray]]:
    return {
        panel.observable: {
            group_key: np.zeros(N_BINS, dtype=np.float64)
            for group_key, _ in GROUPS
        }
        for panel in PANELS
    }


def empty_counts() -> dict[str, dict[str, int]]:
    return {
        panel.observable: {group_key: 0 for group_key, _ in GROUPS}
        for panel in PANELS
    }


def update_minmax(
    stats: dict[str, dict[str, float]],
    observable: str,
    values: pd.Series,
) -> None:
    if values.empty:
        return
    current = stats[observable]
    current["min"] = min(current["min"], float(values.min()))
    current["max"] = max(current["max"], float(values.max()))


def load_histograms(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_path = root / SOURCE_FILE
    observables = {panel.observable for panel in PANELS}
    edges_by_observable = panel_edges()
    histograms = empty_histograms()
    total_counts = empty_counts()
    in_range_counts = empty_counts()
    minmax = {
        panel.observable: {"min": float("inf"), "max": float("-inf")}
        for panel in PANELS
    }

    usecols = [
        "tier",
        "task",
        "same_chi2_decile_id",
        "is_chi2_opt",
        "is_utility_opt",
        "observable",
        "value",
    ]
    for chunk in pd.read_csv(source_path, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False):
        chunk = chunk[
            (chunk["tier"] == SOURCE_TIER)
            & (chunk["task"].isin(SOURCE_TASKS))
            & (chunk["observable"].isin(observables))
        ].copy()
        if chunk.empty:
            continue
        chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
        chunk = chunk[chunk["value"].notna()]
        if chunk.empty:
            continue

        chi2_mask = truthy(chunk["is_chi2_opt"])
        utility_mask = truthy(chunk["is_utility_opt"])
        decile_mask = chunk["same_chi2_decile_id"].astype(str).eq(SELECTED_DECILE)

        masks = {
            "all": pd.Series(True, index=chunk.index),
            "decile": decile_mask,
            "chi2": chi2_mask,
            "utility": utility_mask,
        }
        for panel in PANELS:
            observable_mask = chunk["observable"].eq(panel.observable)
            if not observable_mask.any():
                continue
            observable_values = chunk.loc[observable_mask, "value"]
            update_minmax(minmax, panel.observable, observable_values)
            edges = edges_by_observable[panel.observable]
            for group_key, _ in GROUPS:
                values = chunk.loc[observable_mask & masks[group_key], "value"]
                if values.empty:
                    continue
                total_counts[panel.observable][group_key] += len(values)
                counts, _ = np.histogram(values.to_numpy(dtype=np.float64), bins=edges)
                histograms[panel.observable][group_key] += counts
                in_range_counts[panel.observable][group_key] += int(counts.sum())

    plot_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for panel in PANELS:
        edges = edges_by_observable[panel.observable]
        bin_width = float(edges[1] - edges[0])
        for group_key, group_label in GROUPS:
            counts = histograms[panel.observable][group_key]
            total = total_counts[panel.observable][group_key]
            density = counts / (total * bin_width) if total else counts
            for index, value in enumerate(density):
                plot_rows.append({
                    "panel": panel.key,
                    "observable": panel.observable,
                    "group": group_key,
                    "group_label": group_label,
                    "bin_left": edges[index],
                    "bin_right": edges[index + 1],
                    "bin_center": (edges[index] + edges[index + 1]) / 2,
                    "count": int(counts[index]),
                    "density": float(value),
                    "total_count": int(total),
                    "in_range_count": int(in_range_counts[panel.observable][group_key]),
                })
            summary_rows.append({
                "panel": panel.key,
                "observable": panel.observable,
                "group": group_key,
                "total_count": int(total),
                "in_range_count": int(in_range_counts[panel.observable][group_key]),
                "in_range_fraction": (
                    in_range_counts[panel.observable][group_key] / total if total else np.nan
                ),
                "raw_min": minmax[panel.observable]["min"],
                "raw_max": minmax[panel.observable]["max"],
                "xlim_low": panel.xlim[0],
                "xlim_high": panel.xlim[1],
            })
    return pd.DataFrame(plot_rows), pd.DataFrame(summary_rows)


def write_plot_data(
    plot_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    outdir: Path,
    stats_dir: Path,
) -> None:
    plot_df.to_csv(
        outdir / "chi2_demarcation_overlay_source.csv",
        index=False,
        lineterminator="\n",
    )
    stats_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(
        stats_dir / "chi2_demarcation_overlay_summary.csv",
        index=False,
        lineterminator="\n",
    )


def render(plot_df: pd.DataFrame, outdir: Path, font_preset: str = "times") -> list[Path]:
    matplotlib.rcParams.update({
        **typography_rc(7.35, font_preset),
        "axes.linewidth": 0.55,
        "axes.edgecolor": "black",
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.35), constrained_layout=False)
    fig.subplots_adjust(
        left=0.084,
        right=0.985,
        top=0.920,
        bottom=0.148,
        wspace=0.205,
        hspace=0.385,
    )

    for ax, panel in zip(axes.flat, PANELS):
        panel_df = plot_df[plot_df["observable"] == panel.observable]
        y_max = 0.0
        edges = np.linspace(panel.xlim[0], panel.xlim[1], N_BINS + 1)
        for group_key, _ in GROUPS:
            group_df = panel_df[panel_df["group"] == group_key]
            density = group_df["density"].to_numpy(dtype=np.float64)
            x, y = step_xy(edges, density)
            y_max = max(y_max, float(np.nanmax(y)) if len(y) else 0.0)
            ax.plot(
                x,
                y,
                color=COLORS[group_key],
                linewidth=LINEWIDTHS[group_key],
                alpha=ALPHAS[group_key],
                zorder={"all": 1, "decile": 2, "chi2": 3, "utility": 4}[group_key],
            )

        y_top = y_max * 1.16 if y_max > 0 else 1.0
        ax.set_title(f"{panel.key}. {panel.title}", fontsize=8.75, pad=5.2)
        ax.set_xlabel(panel.xlabel, fontsize=8.3, labelpad=3.4)
        ax.set_ylabel("Density", fontsize=8.15, labelpad=3.4)
        ax.set_xlim(*panel.xlim)
        ax.set_ylim(0, y_top)
        ax.set_xticks(panel.xticks)
        ax.xaxis.set_minor_locator(FixedLocator(midpoint_ticks(panel.xticks)))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:.{panel.decimals}f}")
        )
        y_ticks = np.linspace(0, y_max, 4)
        ax.yaxis.set_major_locator(FixedLocator(y_ticks))
        ax.yaxis.set_minor_locator(FixedLocator(midpoint_ticks(list(y_ticks))))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: "0" if abs(value) < 1e-9 else f"{value:.1f}")
        )
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=7.05,
            pad=1.8,
            length=2.5,
            width=0.55,
        )
        ax.tick_params(axis="both", which="minor", length=0)
        ax.grid(True, which="major", color="#858585", alpha=0.48, linewidth=0.36)
        ax.grid(True, which="minor", color="#858585", alpha=0.48, linewidth=0.36)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
            spine.set_color("black")

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[group_key],
            linewidth=1.55 if group_key != "all" else 1.25,
            alpha=ALPHAS[group_key],
            label=group_label,
        )
        for group_key, group_label in GROUPS
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=True,
        fontsize=7.85,
        handlelength=1.72,
        handletextpad=0.44,
        columnspacing=1.28,
        borderpad=0.28,
        bbox_to_anchor=(0.5, 0.030),
    )
    legend.get_frame().set_edgecolor("#e0e0e0")
    legend.get_frame().set_linewidth(0.6)
    legend.get_frame().set_facecolor("white")

    outputs = [
        outdir / "chi2_demarcation_overlay.pdf",
        outdir / "chi2_demarcation_overlay.svg",
    ]
    for path in outputs:
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
        default=Path("results/figures/chi2_demarcation_overlay"),
        help="Output directory for chi2 demarcation overlay assets.",
    )
    parser.add_argument(
        "--font-preset",
        choices=["sans", "times"],
        default="times",
        help="Typography preset for figure text.",
    )
    parser.add_argument(
        "--source-data",
        type=Path,
        default=None,
        help="Optional pre-binned chi2_demarcation_overlay_source.csv for re-rendering only.",
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

    if args.source_data is None:
        plot_df, summary_df = load_histograms(root)
        write_plot_data(plot_df, summary_df, outdir, stats_dir)
    else:
        source_data = args.source_data if args.source_data.is_absolute() else root / args.source_data
        plot_df = pd.read_csv(source_data)
    render(plot_df, outdir, font_preset=args.font_preset)


if __name__ == "__main__":
    main()
