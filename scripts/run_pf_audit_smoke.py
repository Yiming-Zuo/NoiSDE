#!/usr/bin/env python3
"""Run PF companion sign-convention smoke audits."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noisde.config import load_yaml
from noisde.pf_companion import identity_map_audit, linear_gaussian_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/pf_companion_diagnostic/smoke.yaml")
    parser.add_argument("--outdir", default="tmp/pf_smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(ROOT / args.config)
    seed = int(config.get("seed", 20260524))
    batch_size = int(config.get("batch_size", 8))
    state_dim = int(config.get("state_dim", 2))
    torch.manual_seed(seed)

    x = torch.randn(batch_size, state_dim, requires_grad=True)
    matrix = torch.diag(torch.linspace(0.2, -0.1, state_dim))
    rows = []
    for metric, value in identity_map_audit(x).items():
        rows.append({"audit": "identity_map", "metric": metric, "value": value})
    for metric, value in linear_gaussian_audit(x, matrix).items():
        rows.append({"audit": "linear_gaussian", "metric": metric, "value": value})

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "pf_audit_smoke.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audit", "metric", "value"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {path.relative_to(ROOT)}")
    for row in rows:
        print(f"{row['audit']}.{row['metric']}={row['value']:.6g}")


if __name__ == "__main__":
    main()
