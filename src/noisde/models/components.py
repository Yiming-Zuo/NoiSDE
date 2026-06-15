"""Neural components for the NoiSDE bridge."""
from __future__ import annotations

import torch
from torch import nn


def _as_time_column(t: torch.Tensor, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if t.ndim == 0:
        return t.expand(batch_size, 1).to(device=device, dtype=dtype)
    if t.ndim == 1:
        return t[:, None].to(device=device, dtype=dtype)
    return t.to(device=device, dtype=dtype)


def make_mlp(input_dim: int, output_dim: int, hidden_dim: int, depth: int, activation: type[nn.Module] = nn.SiLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(last, hidden_dim))
        layers.append(activation())
        last = hidden_dim
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class FeatureBuilder(nn.Module):
    """Builds the common state, time, context, and energy feature vector."""

    def __init__(self, state_dim: int, context_dim: int, include_energy_features: bool = True) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.include_energy_features = include_energy_features

    @property
    def output_dim(self) -> int:
        energy_dim = 1 + self.state_dim if self.include_energy_features else 0
        return self.state_dim + 1 + self.context_dim + energy_dim

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None = None,
        reduced_potential: torch.Tensor | None = None,
        force: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        t_col = _as_time_column(t, batch_size, x.device, x.dtype)
        if context is None:
            context = x.new_zeros(batch_size, self.context_dim)
        if context.ndim == 1:
            context = context[None, :].expand(batch_size, -1)
        parts = [x, t_col, context.to(device=x.device, dtype=x.dtype)]
        if self.include_energy_features:
            if reduced_potential is None:
                reduced_potential = x.new_zeros(batch_size)
            if force is None:
                force = x.new_zeros(batch_size, self.state_dim)
            parts.extend([reduced_potential.reshape(batch_size, 1).to(x.dtype), force.to(x.dtype)])
        return torch.cat(parts, dim=-1)


class DriftBackbone(nn.Module):
    """Deterministic bridge velocity v_theta."""

    def __init__(self, state_dim: int, context_dim: int, hidden_dim: int = 128, depth: int = 3) -> None:
        super().__init__()
        self.features = FeatureBuilder(state_dim, context_dim, include_energy_features=False)
        self.net = make_mlp(self.features.output_dim, state_dim, hidden_dim, depth)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(self.features(x, t, context))


class EnergyGuidance(nn.Module):
    """Energy-guided drift correction g_psi."""

    def __init__(self, state_dim: int, context_dim: int, hidden_dim: int = 128, depth: int = 2) -> None:
        super().__init__()
        self.features = FeatureBuilder(state_dim, context_dim, include_energy_features=True)
        self.net = make_mlp(self.features.output_dim, state_dim, hidden_dim, depth)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None,
        reduced_potential: torch.Tensor,
        force: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(self.features(x, t, context, reduced_potential, force))


class EnergyConditionedDiffusionFactor(nn.Module):
    """Low-rank energy-conditioned diffusion factor M_phi."""

    def __init__(
        self,
        state_dim: int,
        context_dim: int,
        rank: int,
        hidden_dim: int = 128,
        depth: int = 3,
        max_scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.rank = rank
        self.max_scale = max_scale
        self.features = FeatureBuilder(state_dim, context_dim, include_energy_features=True)
        self.net = make_mlp(self.features.output_dim, state_dim * rank, hidden_dim, depth)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None,
        reduced_potential: torch.Tensor,
        force: torch.Tensor,
    ) -> torch.Tensor:
        raw = self.net(self.features(x, t, context, reduced_potential, force))
        factor = torch.tanh(raw).reshape(x.shape[0], self.state_dim, self.rank)
        return self.max_scale * factor


class ScoreNetwork(nn.Module):
    """Auxiliary score network for PF companion Mode A."""

    def __init__(self, state_dim: int, context_dim: int, hidden_dim: int = 128, depth: int = 3) -> None:
        super().__init__()
        self.features = FeatureBuilder(state_dim, context_dim, include_energy_features=False)
        self.net = make_mlp(self.features.output_dim, state_dim, hidden_dim, depth)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(self.features(x, t, context))


class CurrentVelocityNetwork(nn.Module):
    """Direct companion current-velocity estimator for PF companion Mode B."""

    def __init__(self, state_dim: int, context_dim: int, hidden_dim: int = 128, depth: int = 3) -> None:
        super().__init__()
        self.features = FeatureBuilder(state_dim, context_dim, include_energy_features=True)
        self.net = make_mlp(self.features.output_dim, state_dim, hidden_dim, depth)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None,
        reduced_potential: torch.Tensor | None = None,
        force: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.net(self.features(x, t, context, reduced_potential, force))
