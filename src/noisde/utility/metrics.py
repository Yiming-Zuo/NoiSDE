"""Hard and relaxed positive-noise utility metrics."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from noisde.types import PathSample


@dataclass(frozen=True)
class UtilityNormalizer:
    """Maps raw metrics to [0, 1] utility coordinates."""

    bad: dict[str, float] = field(default_factory=dict)
    ref: dict[str, float] = field(default_factory=dict)

    def normalize(self, metric: str, value: torch.Tensor, higher_is_better: bool) -> torch.Tensor:
        default_bad = 0.0 if higher_is_better else 1.0
        default_ref = 1.0 if higher_is_better else 0.0
        bad = float(self.bad.get(metric, default_bad))
        ref = float(self.ref.get(metric, default_ref))
        denom = ref - bad
        if abs(denom) < 1.0e-8:
            denom = 1.0e-8
        return ((value - bad) / denom).clamp(0.0, 1.0)


def effective_sample_size(log_weights: torch.Tensor) -> torch.Tensor:
    weights = torch.softmax(log_weights, dim=0)
    return 1.0 / (weights.square().sum() * log_weights.numel())


def coefficient_of_variation_squared(log_weights: torch.Tensor) -> torch.Tensor:
    weights = torch.exp(log_weights - log_weights.max())
    mean = weights.mean()
    return weights.var(unbiased=False) / mean.square().clamp_min(1.0e-12)


def soft_mode_coverage(x: torch.Tensor, centers: torch.Tensor | None, temperature: float = 0.1) -> torch.Tensor:
    if centers is None or centers.numel() == 0:
        return torch.ones((), device=x.device, dtype=x.dtype)
    dist = torch.cdist(x, centers.to(device=x.device, dtype=x.dtype))
    assign = torch.softmax(-dist / temperature, dim=-1).mean(dim=0)
    occupied = 1.0 - torch.exp(-assign * centers.shape[0])
    return occupied.mean()


def high_energy_rate(reduced: torch.Tensor, threshold: float, temperature: float = 0.1) -> torch.Tensor:
    return torch.sigmoid((reduced - threshold) / temperature).mean()


def swap_acceptance_proxy(r_a_x0: torch.Tensor, r_b_x1: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return torch.sigmoid(-(r_b_x1 - r_a_x0).abs() / temperature).mean()


def smooth_positive_noise_utility(
    path: PathSample,
    r_a_x0: torch.Tensor,
    r_b_x1: torch.Tensor,
    path_reduced: torch.Tensor,
    log_weights: torch.Tensor,
    normalizer: UtilityNormalizer,
    weights: dict[str, float] | None = None,
    mode_centers: torch.Tensor | None = None,
    high_energy_threshold: float = 10.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiable relaxed utility used by L_PN."""

    weights = weights or {
        "ESS_N": 1.0,
        "A_swap": 1.0,
        "C_mode": 1.0,
        "D_E": 1.0,
        "CV2_w": 1.0,
        "R_high_E": 1.0,
    }
    raw = {
        "ESS_N": effective_sample_size(log_weights),
        "A_swap": swap_acceptance_proxy(r_a_x0, r_b_x1),
        "C_mode": soft_mode_coverage(path.x1, mode_centers),
        "D_E": torch.relu(path_reduced - high_energy_threshold).mean(),
        "CV2_w": coefficient_of_variation_squared(log_weights),
        "R_high_E": high_energy_rate(path_reduced, high_energy_threshold),
    }
    higher = {"ESS_N", "A_swap", "C_mode"}
    coords = {
        key: normalizer.normalize(key, value, key in higher)
        for key, value in raw.items()
    }
    total_weight = sum(float(weights.get(key, 0.0)) for key in coords) or 1.0
    utility = sum(float(weights.get(key, 0.0)) * coords[key] for key in coords) / total_weight
    return utility, coords


def compute_hard_metrics(
    path: PathSample,
    r_a_x0: torch.Tensor,
    r_b_x1: torch.Tensor,
    path_reduced: torch.Tensor,
    log_weights: torch.Tensor,
    mode_centers: torch.Tensor | None = None,
    high_energy_threshold: float = 10.0,
) -> dict[str, float]:
    with torch.no_grad():
        return {
            "ESS_N": float(effective_sample_size(log_weights).detach().cpu()),
            "A_swap": float(swap_acceptance_proxy(r_a_x0, r_b_x1).detach().cpu()),
            "C_mode": float(soft_mode_coverage(path.x1, mode_centers).detach().cpu()),
            "D_E": float(torch.relu(path_reduced - high_energy_threshold).mean().detach().cpu()),
            "CV2_w": float(coefficient_of_variation_squared(log_weights).detach().cpu()),
            "R_high_E": float((path_reduced > high_energy_threshold).to(torch.float32).mean().detach().cpu()),
        }
