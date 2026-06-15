#!/usr/bin/env python3
"""Unified figure-generation entry point for the NoiSDE repository."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Install with: pip install -r requirements.txt") from exc


FIGURE_CONFIGS = {
    "main_result_overview": "configs/figures/main_result_overview.yaml",
    "chi2_demarcation_overlay": "configs/figures/chi2_demarcation_overlay.yaml",
    "component_gains_by_family": "configs/figures/component_gains_by_family.yaml",
}


def load_config(root: Path, config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "command" not in config:
        raise ValueError(f"Invalid figure config: {path}")
    return config


def run_config(root: Path, config_path: str | Path, dry_run: bool) -> None:
    config = load_config(root, config_path)
    command = [str(part) for part in config["command"]]
    print("+ " + " ".join(command))
    if dry_run:
        return
    env = os.environ.copy()
    src_path = str(root / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    subprocess.run(command, cwd=root, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Regenerate every supported figure.")
    group.add_argument("--figure", choices=sorted(FIGURE_CONFIGS), help="Figure key to regenerate.")
    group.add_argument("--config", help="Path to a figure YAML config.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if args.all:
        for key in FIGURE_CONFIGS:
            print(f"[figure] {key}")
            run_config(root, FIGURE_CONFIGS[key], args.dry_run)
        return
    if args.figure:
        run_config(root, FIGURE_CONFIGS[args.figure], args.dry_run)
        return
    run_config(root, args.config, args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
