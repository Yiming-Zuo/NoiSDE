"""Analytic toy Boltzmann tasks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from noisde.types import Batch


ToyKind = Literal["double_well", "four_well", "ring"]


@dataclass(frozen=True)
class ToyTaskConfig:
    kind: ToyKind = "double_well"
    state_dim: int = 2
    beta: float = 1.0
    batch_size: int = 64
    context_dim: int = 1


class ToyBridgeTask:
    """Analytic source-target bridge task for smoke and method development."""

    def __init__(self, config: ToyTaskConfig) -> None:
        self.config = config

    def source_reduced_potential(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        return 0.5 * x.square().sum(dim=-1)

    def target_reduced_potential(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if self.config.kind == "double_well":
            first = (x[:, 0].square() - 1.0).square()
            rest = 0.5 * x[:, 1:].square().sum(dim=-1)
            return self.config.beta * (first + rest)
        if self.config.kind == "ring":
            radius = x.norm(dim=-1)
            return self.config.beta * ((radius - 2.0).square() / 0.18)
        return self.config.beta * four_well_energy(x)

    def sample_source(self, batch_size: int | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
        n = batch_size or self.config.batch_size
        return torch.randn(n, self.config.state_dim, generator=generator)

    def sample_target_proxy(self, batch_size: int | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
        n = batch_size or self.config.batch_size
        if self.config.kind == "double_well":
            signs = torch.randint(0, 2, (n, 1), generator=generator, dtype=torch.float32).mul(2.0).sub(1.0)
            noise = 0.2 * torch.randn(n, self.config.state_dim, generator=generator)
            centers = torch.zeros(n, self.config.state_dim)
            centers[:, :1] = signs
            return centers + noise
        if self.config.kind == "ring":
            theta = 2.0 * torch.pi * torch.rand(n, generator=generator)
            x = torch.stack([2.0 * torch.cos(theta), 2.0 * torch.sin(theta)], dim=-1)
            if self.config.state_dim > 2:
                x = torch.cat([x, torch.zeros(n, self.config.state_dim - 2)], dim=-1)
            return x + 0.1 * torch.randn(n, self.config.state_dim, generator=generator)
        centers = torch.tensor([[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5]], dtype=torch.float32)
        ids = torch.randint(0, centers.shape[0], (n,), generator=generator)
        x = centers[ids] + 0.25 * torch.randn(n, 2, generator=generator)
        if self.config.state_dim > 2:
            x = torch.cat([x, torch.zeros(n, self.config.state_dim - 2)], dim=-1)
        return x

    def batch_stream(self, seed: int = 20260524):
        generator = torch.Generator().manual_seed(seed)
        while True:
            x0 = self.sample_source(generator=generator)
            x1 = self.sample_target_proxy(generator=generator)
            context = torch.zeros(x0.shape[0], self.config.context_dim)
            yield Batch(x0=x0, x1=x1, context=context)


def four_well_energy(x: torch.Tensor) -> torch.Tensor:
    centers = x.new_tensor([[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5]])
    logits = -torch.cdist(x[:, :2], centers).square() / 0.35
    return -torch.logsumexp(logits, dim=-1)
