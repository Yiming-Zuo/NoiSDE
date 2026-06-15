"""Density-dependent diagnostic helpers."""
from __future__ import annotations

import math

import torch


def reduced_fep(log_weights: torch.Tensor) -> torch.Tensor:
    return -torch.logsumexp(log_weights, dim=0) + math.log(log_weights.numel())


def summarize_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    if not rows:
        return out
    keys = sorted({key for row in rows for key in row})
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            out[f"{key}_mean"] = sum(values) / len(values)
    return out
