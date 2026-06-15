#!/usr/bin/env python3
"""Run a PyMBAR free-energy estimate from an NPZ reduced-potential matrix."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noisde.benchmarks import MBARFreeEnergyEstimator
from noisde.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/free_energy/default_mbar.yaml")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    config = load_yaml(project_path(args.config))
    mbar_config = config["mbar"]
    input_path = project_path(mbar_config["input_npz"])
    if not input_path.exists():
        raise FileNotFoundError(f"Missing MBAR input matrix: {input_path}")
    data = np.load(input_path)
    u_kn = np.asarray(data[mbar_config.get("u_kn_key", "u_kn")], dtype=np.float64)
    n_k = np.asarray(data[mbar_config.get("n_k_key", "N_k")], dtype=np.int64)
    result = MBARFreeEnergyEstimator().compute(u_kn, n_k)

    output_path = project_path(args.out or mbar_config["output_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["state_i", "state_j", "delta_f", "d_delta_f"], lineterminator="\n")
        writer.writeheader()
        for i in range(result.delta_f.shape[0]):
            for j in range(result.delta_f.shape[1]):
                writer.writerow({
                    "state_i": i,
                    "state_j": j,
                    "delta_f": result.delta_f[i, j],
                    "d_delta_f": result.d_delta_f[i, j],
                })
    print(f"wrote {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
