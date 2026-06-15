"""Writers for PF-companion audit artifacts."""
from __future__ import annotations

import csv
from pathlib import Path


PF_AUDIT_COLUMNS = [
    "audit_id",
    "window_id",
    "seed",
    "variant",
    "pf_mode",
    "hutchinson_M",
    "ess_n",
    "log_weight_995",
    "overlap_min",
    "r_highE",
    "deltaF_abs_err_kcal_mol",
    "pf_gate_pass",
    "mbar_bootstrap_gate_pass",
    "reference_quality_gate_pass",
    "overall_pass_fail",
    "downgrade_flag",
    "allowed_claim",
]


def write_pf_audit_log(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PF_AUDIT_COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path
