"""Ablation-variant registry for manuscript component attribution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DiffusionKind = Literal["none", "fixed", "learned_iso", "learned_diagonal", "learned_anisotropic"]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    use_energy_guidance: bool
    diffusion_kind: DiffusionKind
    use_positive_noise_loss: bool
    description: str
    use_diffusion_energy_features: bool = True


VARIANTS: dict[str, VariantSpec] = {
    "F0": VariantSpec("F0", False, "none", False, "deterministic backbone"),
    "F1": VariantSpec("F1", True, "none", False, "energy-guided drift only"),
    "F2": VariantSpec("F2", False, "learned_anisotropic", True, "positive noise without energy-guided drift"),
    "A2": VariantSpec("A2", True, "fixed", False, "fixed isotropic noise sweep"),
    "A3_prime": VariantSpec("A3_prime", True, "learned_iso", False, "learned isotropic noise without utility", False),
    "A3": VariantSpec("A3", True, "learned_iso", True, "learned isotropic noise with utility", False),
    "A4_prime": VariantSpec("A4_prime", True, "learned_anisotropic", False, "full architecture without utility"),
    "A4": VariantSpec("A4", True, "learned_anisotropic", True, "full NoiSDE"),
    "A5": VariantSpec("A5", True, "learned_anisotropic", True, "over-noise stress variant"),
    "B1": VariantSpec("B1", True, "learned_diagonal", False, "non-PN learnable stochastic control"),
}


def get_variant(name: str) -> VariantSpec:
    try:
        return VARIANTS[name]
    except KeyError as exc:
        allowed = ", ".join(sorted(VARIANTS))
        raise ValueError(f"Unknown NoiSDE variant {name!r}. Allowed: {allowed}") from exc
