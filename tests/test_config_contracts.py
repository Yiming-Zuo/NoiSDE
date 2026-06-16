from __future__ import annotations

from pathlib import Path

from noisde.config import load_yaml
from noisde.models import VARIANTS


ROOT = Path(__file__).resolve().parents[1]


def test_all_yaml_configs_load_as_mappings() -> None:
    paths = sorted((ROOT / "configs").rglob("*.yaml"))

    assert paths
    for path in paths:
        data = load_yaml(path)
        assert isinstance(data, dict), path


def test_core_ablation_matrix_matches_variant_registry() -> None:
    data = load_yaml(ROOT / "configs/experiments/core_ablation/variant_matrix.yaml")
    variants = data["variants"]

    assert set(variants) <= set(VARIANTS)
    for name, item in variants.items():
        assert item["model_variant"] == name
        assert item["utility_loss"] == VARIANTS[name].use_positive_noise_loss


def test_core_ablation_smoke_uses_declared_variant_matrix() -> None:
    matrix = load_yaml(ROOT / "configs/experiments/core_ablation/variant_matrix.yaml")
    smoke = load_yaml(ROOT / "configs/experiments/core_ablation/smoke_variant_matrix.yaml")

    assert smoke["variant_matrix"] == "configs/experiments/core_ablation/variant_matrix.yaml"
    assert set(smoke["variants"]) == set(matrix["variants"])


def test_fixed_noise_grid_is_positive_and_ordered() -> None:
    data = load_yaml(ROOT / "configs/experiments/core_ablation/variant_matrix.yaml")
    grid = data["variants"]["A2"]["fixed_noise_grid"]

    assert grid == sorted(grid)
    assert all(value > 0 for value in grid)
