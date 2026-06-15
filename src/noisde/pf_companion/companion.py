"""Detached-feature probability-flow companion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from noisde.pf_companion.divergence import diffusion_tensor_divergence, exact_divergence
from noisde.sde.sampler import reduced_potential_and_force
from noisde.types import ReducedPotentialFn


@dataclass(frozen=True)
class PFCompanionConfig:
    mode: Literal["score", "current"] = "score"
    num_steps: int = 32
    use_exact_divergence: bool = True


class ProbabilityFlowCompanion:
    """PF companion used for density-dependent diagnostics."""

    def __init__(self, config: PFCompanionConfig | None = None) -> None:
        self.config = config or PFCompanionConfig()

    def velocity(
        self,
        model,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None,
        reduced_potential_fn: ReducedPotentialFn,
    ) -> torch.Tensor:
        x_req = x.requires_grad_(True)
        reduced, force = reduced_potential_and_force(reduced_potential_fn, x_req, context)
        if self.config.mode == "current":
            return model.current_velocity(x_req, t, context, reduced.detach(), force.detach())
        drift = model.drift(x_req, t, context, reduced, force)
        tensor = model.diffusion_tensor(x_req, t, context, reduced.detach(), force.detach())
        div_tensor = diffusion_tensor_divergence(tensor, x_req)
        score = model.score(x_req, t, context)
        score_term = torch.bmm(tensor, score.unsqueeze(-1)).squeeze(-1)
        return drift - 0.5 * div_tensor - 0.5 * score_term

    def integrate_log_density(
        self,
        model,
        z0: torch.Tensor,
        context: torch.Tensor | None,
        reduced_potential_fn: ReducedPotentialFn,
        log_q0_fn,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dt = 1.0 / self.config.num_steps
        z = z0
        log_q = log_q0_fn(z0, context)
        for step in range(self.config.num_steps):
            t = z.new_full((z.shape[0],), step * dt)

            def field(inp: torch.Tensor) -> torch.Tensor:
                return self.velocity(model, inp, t, context, reduced_potential_fn)

            z_req = z.detach().requires_grad_(True)
            v = field(z_req)
            div = exact_divergence(v, z_req)
            log_q = log_q - div * dt
            z = z + v * dt
        return z, log_q

    def log_weights(
        self,
        model,
        z0: torch.Tensor,
        context: torch.Tensor | None,
        r_a_fn: ReducedPotentialFn,
        r_b_fn: ReducedPotentialFn,
        log_q0_fn,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z1, log_q1 = self.integrate_log_density(model, z0, context, r_b_fn, log_q0_fn)
        r_b = r_b_fn(z1, context)
        log_w = -r_b - log_q1
        return z1, log_q1, log_w
