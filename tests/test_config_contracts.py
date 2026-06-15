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
