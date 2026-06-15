"""Divergence estimators for PF companion flows."""
from __future__ import annotations

import torch


def exact_divergence(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Computes tr(dy/dx) exactly by looping over state dimensions."""

    if y.shape != x.shape:
        raise ValueError("y and x must have the same shape")
    div = x.new_zeros(x.shape[0])
    for dim in range(x.shape[1]):
        grad = torch.autograd.grad(
            y[:, dim].sum(),
            x,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        div = div + grad[:, dim]
    return div


def hutchinson_divergence(
    vector_field,
    x: torch.Tensor,
    num_probes: int = 1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Hutchinson trace estimate for large state dimensions."""

    estimates = []
    for _ in range(num_probes):
        probe = torch.randint(
            0,
            2,
            x.shape,
            device=x.device,
            generator=generator,
        ).to(dtype=x.dtype)
        probe = probe.mul(2.0).sub(1.0)
        y = vector_field(x)
        dot = (y * probe).sum()
        grad = torch.autograd.grad(dot, x, create_graph=True, retain_graph=True)[0]
        estimates.append((grad * probe).sum(dim=-1))
    return torch.stack(estimates, dim=0).mean(dim=0)


def diffusion_tensor_divergence(tensor: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Computes (div a)_i = sum_j partial_j a_ij for small dimensions."""

    if tensor.ndim != 3:
        raise ValueError("tensor must have shape [batch, dim, dim]")
    batch, dim, _ = tensor.shape
    out = x.new_zeros(batch, dim)
    for i in range(dim):
        acc = x.new_zeros(batch)
        for j in range(dim):
            grad = torch.autograd.grad(
                tensor[:, i, j].sum(),
                x,
                create_graph=True,
                retain_graph=True,
            )[0]
            acc = acc + grad[:, j]
        out[:, i] = acc
    return out
