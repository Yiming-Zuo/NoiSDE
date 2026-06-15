"""Assembled NoiSDE bridge model."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .components import (
    CurrentVelocityNetwork,
    DriftBackbone,
    EnergyConditionedDiffusionFactor,
    EnergyGuidance,
    ScoreNetwork,
)
from .variants import VariantSpec, get_variant


@dataclass(frozen=True)
class NoiSDEModelConfig:
    state_dim: int
    context_dim: int = 0
    rank: int = 2
    hidden_dim: int = 128
    depth: int = 3
    diffusion_floor: float = 1.0e-4
    diffusion_max_scale: float = 2.0
    energy_guidance_scale: float = 1.0
    variant: str = "A4"
    fixed_noise_scale: float = 0.1
    over_noise_scale: float = 3.0


class NoiSDEModel(nn.Module):
    """Energy-guided bridge with learned low-rank diffusion tensor."""

    def __init__(self, config: NoiSDEModelConfig) -> None:
        super().__init__()
        self.config = config
        self.variant_spec: VariantSpec = get_variant(config.variant)
        self.drift_backbone = DriftBackbone(config.state_dim, config.context_dim, config.hidden_dim, config.depth)
        self.energy_guidance = EnergyGuidance(config.state_dim, config.context_dim, config.hidden_dim, max(1, config.depth - 1))
        self.diffusion_factor_net = EnergyConditionedDiffusionFactor(
            config.state_dim,
            config.context_dim,
            config.rank,
            config.hidden_dim,
            config.depth,
            config.diffusion_max_scale,
        )
        self.score_net = ScoreNetwork(config.state_dim, config.context_dim, config.hidden_dim, config.depth)
        self.current_velocity_net = CurrentVelocityNetwork(config.state_dim, config.context_dim, config.hidden_dim, config.depth)

    @property
    def state_dim(self) -> int:
        return self.config.state_dim

    def drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None,
        reduced_potential: torch.Tensor,
        force: torch.Tensor,
    ) -> torch.Tensor:
        backbone = self.drift_backbone(x, t, context)
        if not self.variant_spec.use_energy_guidance:
            return backbone
        guidance = self.energy_guidance(x, t, context, reduced_potential, force)
        return backbone + self.config.energy_guidance_scale * guidance

    def diffusion_factor(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None,
        reduced_potential: torch.Tensor,
        force: torch.Tensor,
    ) -> torch.Tensor:
        kind = self.variant_spec.diffusion_kind
        if kind == "none":
            return x.new_zeros(x.shape[0], self.state_dim, self.config.rank)
        if kind == "fixed":
            return self._isotropic_factor(x, x.new_full((x.shape[0],), self.config.fixed_noise_scale))
        if self.variant_spec.use_diffusion_energy_features:
            feature_reduced = reduced_potential
            feature_force = force
        else:
            feature_reduced = torch.zeros_like(reduced_potential)
            feature_force = torch.zeros_like(force)
        raw = self.diffusion_factor_net(x, t, context, feature_reduced, feature_force)
        if self.config.variant == "A5":
            raw = raw * self.config.over_noise_scale
        if kind == "learned_iso":
            scale = raw.square().mean(dim=(1, 2)).sqrt().clamp_min(self.config.diffusion_floor ** 0.5)
            return self._isotropic_factor(x, scale)
        if kind == "learned_diagonal":
            scale = raw.square().mean(dim=-1).sqrt().clamp_min(self.config.diffusion_floor ** 0.5)
            return torch.diag_embed(scale)
        return raw

    def diffusion_tensor(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None,
        reduced_potential: torch.Tensor,
        force: torch.Tensor,
    ) -> torch.Tensor:
        if self.variant_spec.diffusion_kind == "none":
            return x.new_zeros(x.shape[0], self.state_dim, self.state_dim)
        factor = self.diffusion_factor(x, t, context, reduced_potential, force)
        tensor = factor @ factor.transpose(-1, -2)
        eye = torch.eye(self.state_dim, device=x.device, dtype=x.dtype).expand(x.shape[0], -1, -1)
        return tensor + self.config.diffusion_floor * eye

    def score(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor | None) -> torch.Tensor:
        return self.score_net(x, t, context)

    def current_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None,
        reduced_potential: torch.Tensor,
        force: torch.Tensor,
    ) -> torch.Tensor:
        return self.current_velocity_net(x, t, context, reduced_potential, force)

    def _isotropic_factor(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return torch.diag_embed(scale[:, None].expand(-1, self.state_dim))
