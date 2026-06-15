"""NoiSDE training objective."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from noisde.sde import EulerMaruyamaSampler
from noisde.sde.sampler import reduced_potential_and_force
from noisde.types import Batch, ReducedPotentialFn
from noisde.utility import UtilityNormalizer, smooth_positive_noise_utility


@dataclass(frozen=True)
class NoiSDELossConfig:
    lambda_end: float = 1.0
    lambda_energy: float = 0.1
    lambda_pn: float = 1.0
    lambda_reg: float = 1.0e-3
    path_steps: int = 16
    energy_threshold: float = 10.0
    endpoint_kernel_bandwidth: float = 1.0
    utility_weights: dict[str, float] = field(default_factory=dict)
    normalizer: UtilityNormalizer = field(default_factory=UtilityNormalizer)


class NoiSDEObjective(nn.Module):
    """Computes FM, endpoint, energy, positive-noise, and magnitude losses."""

    def __init__(
        self,
        r_a_fn: ReducedPotentialFn,
        r_b_fn: ReducedPotentialFn,
        config: NoiSDELossConfig | None = None,
    ) -> None:
        super().__init__()
        self.r_a_fn = r_a_fn
        self.r_b_fn = r_b_fn
        self.config = config or NoiSDELossConfig()
        self.sampler = EulerMaruyamaSampler(self.config.path_steps)

    def forward(self, model, batch: Batch, generator: torch.Generator | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x0, x1, context = batch.x0, batch.x1, batch.context
        fm_loss = self.flow_matching_loss(model, x0, x1, context, generator)
        path = self.sampler.rollout(model, x0, context, self.r_b_fn, generator)
        endpoint_loss = gaussian_mmd(path.x1, x1, self.config.endpoint_kernel_bandwidth)
        flat_path = path.states.reshape(-1, path.states.shape[-1])
        flat_context = _repeat_context(context, path.states.shape[1]) if context is not None else None
        path_reduced = self.r_b_fn(flat_path, flat_context).reshape(path.states.shape[:2])
        energy_loss = torch.relu(path_reduced - self.config.energy_threshold).mean()
        r_a_x0 = self.r_a_fn(x0, context)
        r_b_x1 = self.r_b_fn(path.x1, context)
        log_weights = -r_b_x1 + r_a_x0
        utility, coords = smooth_positive_noise_utility(
            path,
            r_a_x0,
            r_b_x1,
            path_reduced,
            log_weights,
            self.config.normalizer,
            self.config.utility_weights,
            high_energy_threshold=self.config.energy_threshold,
        )
        use_pn = getattr(getattr(model, "variant_spec", None), "use_positive_noise_loss", True)
        pn_loss = -utility if use_pn else utility.new_zeros(())
        magnitude_loss = self.magnitude_regularizer(model, path, context)
        total = (
            fm_loss
            + self.config.lambda_end * endpoint_loss
            + self.config.lambda_energy * energy_loss
            + self.config.lambda_pn * pn_loss
            + self.config.lambda_reg * magnitude_loss
        )
        parts = {
            "loss": total,
            "L_FM": fm_loss.detach(),
            "L_end": endpoint_loss.detach(),
            "L_E": energy_loss.detach(),
            "L_PN": pn_loss.detach(),
            "L_M": magnitude_loss.detach(),
            "utility": utility.detach(),
        }
        parts.update({f"U_{key}": value.detach() for key, value in coords.items()})
        return total, parts

    def flow_matching_loss(
        self,
        model,
        x0: torch.Tensor,
        x1: torch.Tensor,
        context: torch.Tensor | None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        batch_size = x0.shape[0]
        t = torch.rand(batch_size, device=x0.device, dtype=x0.dtype, generator=generator)
        x_t = (1.0 - t[:, None]) * x0 + t[:, None] * x1
        target_velocity = x1 - x0
        x_t_req = x_t.detach().requires_grad_(True)
        reduced, force = reduced_potential_and_force(self.r_b_fn, x_t_req, context)
        pred = model.drift(x_t_req, t, context, reduced, force)
        return (pred - target_velocity).square().mean()

    def magnitude_regularizer(self, model, path, context: torch.Tensor | None) -> torch.Tensor:
        states = path.states[:, :-1].reshape(-1, path.states.shape[-1])
        times = path.times[:-1].repeat(path.states.shape[0])
        flat_context = _repeat_context(context, path.states.shape[1] - 1) if context is not None else None
        states_req = states.detach().requires_grad_(True)
        reduced, force = reduced_potential_and_force(self.r_b_fn, states_req, flat_context)
        factor = model.diffusion_factor(states_req, times, flat_context, reduced.detach(), force.detach())
        return factor.square().mean()


def gaussian_mmd(left: torch.Tensor, right: torch.Tensor, bandwidth: float) -> torch.Tensor:
    gamma = 1.0 / (2.0 * bandwidth * bandwidth)
    k_xx = torch.exp(-gamma * torch.cdist(left, left).square()).mean()
    k_yy = torch.exp(-gamma * torch.cdist(right, right).square()).mean()
    k_xy = torch.exp(-gamma * torch.cdist(left, right).square()).mean()
    return k_xx + k_yy - 2.0 * k_xy


def _repeat_context(context: torch.Tensor | None, repeats: int) -> torch.Tensor | None:
    if context is None:
        return None
    return context[:, None, :].expand(-1, repeats, -1).reshape(-1, context.shape[-1])
