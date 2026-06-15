from __future__ import annotations

import importlib.util

import pytest


def test_openmm_adapter_reports_missing_dependency() -> None:
    if importlib.util.find_spec("openmm") is not None:
        pytest.skip("OpenMM is installed in this environment")

    from noisde.benchmarks import OpenMMReducedPotential, OpenMMTaskConfig

    with pytest.raises(ImportError, match="OpenMM is required"):
        OpenMMReducedPotential(OpenMMTaskConfig(structure_path="missing.pdb"))


def test_pymbar_adapter_reports_missing_dependency() -> None:
    if importlib.util.find_spec("pymbar") is not None:
        pytest.skip("PyMBAR is installed in this environment")

    from noisde.benchmarks import MBARFreeEnergyEstimator

    with pytest.raises(ImportError, match="PyMBAR is required"):
        MBARFreeEnergyEstimator()
