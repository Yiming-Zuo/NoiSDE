#!/usr/bin/env python3
"""Summarize component-ablation seed-level tables."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noisde.analysis import (
    ContrastSpec,
    summarize_component_ablation,
    write_component_ablation_outputs,
)


DEFAULT_VARIANTS = [
    "F0",
    "F1",
    "F2",
    "A2",
    "A3_prime",
    "A3",
    "B1",
    "A4_prime",
    "A4_FM_selected",
    "A4",
    "A5",
]

DEFAULT_CONTRASTS = [
    ContrastSpec("A4", "B1"),
    ContrastSpec("A4", "A4_prime"),
    ContrastSpec("A4_prime", "A3"),
    ContrastSpec("A4", "A4_FM_selected"),
]


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/processed/seed_level/component_ablation_results.csv",
        help="Long-format component-ablation seed-level CSV.",
    )
    parser.add_argument("--tier", default="core_ablation")
    parser.add_argument("--metric", default="Ubar")
    parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        help="Variant to include. Repeat to override the default manuscript set.",
    )
    parser.add_argument(
        "--contrast",
        action="append",
        dest="contrasts",
        help="Paired contrast as LEFT:RIGHT or LEFT--RIGHT. Repeat to override defaults.",
    )
    parser.add_argument(
        "--outdir",
        default="results/statistics/component_ablation",
        help="Output directory for summary CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contrasts = (
        [ContrastSpec.parse(item) for item in args.contrasts]
        if args.contrasts
        else DEFAULT_CONTRASTS
    )
    variants = args.variants if args.variants else DEFAULT_VARIANTS
    summary = summarize_component_ablation(
        project_path(args.input),
        tier=args.tier,
        metric=args.metric,
        variants=variants,
        contrasts=contrasts,
    )
    summary_path, contrast_path = write_component_ablation_outputs(
        summary,
        project_path(args.outdir),
    )
    print(f"wrote {summary_path.relative_to(ROOT)}")
    print(f"wrote {contrast_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
