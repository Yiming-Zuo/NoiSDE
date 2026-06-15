"""Shared lightweight data containers."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import torch


Tensor = torch.Tensor
ReducedPotentialFn = Callable[[Tensor, Tensor | None], Tensor]
ForceFn = Callable[[Tensor, Tensor | None], Tensor]


class PotentialProvider(Protocol):
    """Callable potential object that may also provide explicit gradients."""

    def __call__(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        ...

    def reduced_and_force(self, x: Tensor, context: Tensor | None = None) -> tuple[Tensor, Tensor]:
        ...


@dataclass(frozen=True)
class ThermodynamicContext:
    """Tensor-backed source-target condition descriptor."""

    value: Tensor
    name: str = "context"


@dataclass
class Batch:
    """Endpoint-pair batch used by the training objective."""

    x0: Tensor
    x1: Tensor
    context: Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PathSample:
    """Euler-Maruyama path sample."""

    times: Tensor
    states: Tensor
    increments: Tensor
    context: Tensor | None = None

    @property
    def x0(self) -> Tensor:
        return self.states[:, 0]

    @property
    def x1(self) -> Tensor:
        return self.states[:, -1]


@dataclass
class CheckpointRecord:
    """Validation record consumed by Pareto-knee selection."""

    step: int
    path: Path | None
    metrics: dict[str, float]
    score_vector: dict[str, float]


@dataclass(frozen=True)
class SafetyGates:
    """Hard validation gates from the manuscript checkpoint rule."""

    r_high_e_max: float
    d_e_max: float
