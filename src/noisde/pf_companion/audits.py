"""PF companion sign-convention audits."""
from __future__ import annotations

import torch

from .divergence import exact_divergence


def identity_map_audit(x: torch.Tensor) -> dict[str, float]:
    v = torch.zeros_like(x, requires_grad=True)
    div = exact_divergence(v + x * 0.0, x.requires_grad_(True))
    return {"max_abs_divergence": float(div.abs().max().detach().cpu())}


def linear_gaussian_audit(x: torch.Tensor, matrix: torch.Tensor) -> dict[str, float]:
    x_req = x.detach().requires_grad_(True)
    y = x_req @ matrix.T
    div = exact_divergence(y, x_req)
    expected = torch.trace(matrix).expand_as(div)
    err = (div - expected).abs()
    return {
        "max_abs_error": float(err.max().detach().cpu()),
        "mean_abs_error": float(err.mean().detach().cpu()),
    }
