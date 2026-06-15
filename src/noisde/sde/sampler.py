"""Euler-Maruyama bridge rollout."""
from __future__ import annotations

import torch

from noisde.types import PathSample, ReducedPotentialFn


class EulerMaruyamaSampler:
    """Reparameterized sampler for the learned bridge."""

    def __init__(self, num_steps: int = 32) -> None:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        self.num_steps = num_steps

    def rollout(
        self,
        model,
        x0: torch.Tensor,
        context: torch.Tensor | None,
        reduced_potential_fn: ReducedPotentialFn,
        generator: torch.Generator | None = None,
    ) -> PathSample:
        states = [x0]
        increments = []
        dt = 1.0 / self.num_steps
        sqrt_dt = dt ** 0.5
        x = x0
        times = torch.linspace(0.0, 1.0, self.num_steps + 1, device=x0.device, dtype=x0.dtype)
        for step in range(self.num_steps):
            t = times[step].expand(x.shape[0])
            x_req = x.requires_grad_(True)
            r, force = reduced_potential_and_force(reduced_potential_fn, x_req, context)
            drift = model.drift(x_req, t, context, r, force)
            factor = model.diffusion_factor(x_req, t, context, r.detach(), force.detach())
            noise = torch.randn(
                x.shape[0],
                factor.shape[-1],
                device=x.device,
                dtype=x.dtype,
                generator=generator,
            )
            dx_noise = torch.bmm(factor, noise.unsqueeze(-1)).squeeze(-1) * sqrt_dt
            x = x + drift * dt + dx_noise
            states.append(x)
            increments.append(noise)
        return PathSample(
            times=times,
            states=torch.stack(states, dim=1),
            increments=torch.stack(increments, dim=1),
            context=context,
        )


def reduced_potential_and_force(
    reduced_potential_fn: ReducedPotentialFn,
    x: torch.Tensor,
    context: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(reduced_potential_fn, "reduced_and_force"):
        reduced, force = reduced_potential_fn.reduced_and_force(x, context)
        return reduced.to(device=x.device, dtype=x.dtype), force.to(device=x.device, dtype=x.dtype)
    with torch.enable_grad():
        x = x.requires_grad_(True)
        reduced = reduced_potential_fn(x, context)
        grad = torch.autograd.grad(reduced.sum(), x, create_graph=True, retain_graph=True)[0]
    return reduced, grad
