"""OpenMM-backed molecular reduced-potential adapter."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from noisde.types import Batch


@dataclass(frozen=True)
class OpenMMTaskConfig:
    structure_path: Path
    forcefield_files: tuple[str, ...] = ("amber14-all.xml", "implicit/gbn2.xml")
    temperature_kelvin: float = 300.0
    nonbonded_method: str = "NoCutoff"
    constraints: str = "HBonds"
    solute_dielectric: float = 1.0
    solvent_dielectric: float = 78.5


class OpenMMReducedPotential:
    """Evaluates reduced potentials through OpenMM when installed."""

    def __init__(self, config: OpenMMTaskConfig) -> None:
        try:
            from openmm import LangevinMiddleIntegrator, unit
            from openmm import app
            from openmm.app import ForceField, HBonds, NoCutoff, PDBFile, Simulation
        except ImportError as exc:
            raise ImportError("OpenMM is required for molecular tasks. Install openmm in the project environment.") from exc

        self.config = config
        self.unit = unit
        pdb = PDBFile(str(config.structure_path))
        forcefield = ForceField(*config.forcefield_files)
        nonbonded = NoCutoff if config.nonbonded_method == "NoCutoff" else getattr(app, config.nonbonded_method)
        constraints = HBonds if config.constraints == "HBonds" else None
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=nonbonded,
            constraints=constraints,
            soluteDielectric=config.solute_dielectric,
            solventDielectric=config.solvent_dielectric,
        )
        integrator = LangevinMiddleIntegrator(
            config.temperature_kelvin * unit.kelvin,
            1.0 / unit.picosecond,
            0.002 * unit.picoseconds,
        )
        self.simulation = Simulation(pdb.topology, system, integrator)
        self.simulation.context.setPositions(pdb.positions)
        self.beta = 1.0 / (unit.MOLAR_GAS_CONSTANT_R * config.temperature_kelvin * unit.kelvin)

    def __call__(self, positions_nm: np.ndarray) -> float:
        quantity = positions_nm * self.unit.nanometer
        self.simulation.context.setPositions(quantity)
        state = self.simulation.context.getState(energy=True)
        energy = state.getPotentialEnergy()
        reduced = energy * self.beta
        return float(reduced.value_in_unit(self.unit.dimensionless))

    def energy_and_gradient(self, positions_nm: np.ndarray) -> tuple[float, np.ndarray]:
        """Return reduced potential and gradient with respect to positions in nm."""

        quantity = positions_nm * self.unit.nanometer
        self.simulation.context.setPositions(quantity)
        state = self.simulation.context.getState(energy=True, forces=True)
        energy = state.getPotentialEnergy()
        forces = state.getForces(asNumpy=True)
        reduced = energy * self.beta
        reduced_gradient = -forces * self.beta
        gradient_value = reduced_gradient.value_in_unit(1 / self.unit.nanometer)
        return float(reduced.value_in_unit(self.unit.dimensionless)), np.asarray(gradient_value, dtype=np.float64)

    def reduced_and_force(self, x: torch.Tensor, context: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Torch-facing external potential hook.

        `x` is flattened as `[batch, 3 * n_atoms]` in nanometers. The returned
        second tensor is the gradient of the reduced potential with respect to
        flattened positions, matching the internal `force` feature convention.
        """

        values = []
        gradients = []
        flat = x.detach().cpu().numpy()
        for row in flat:
            positions = row.reshape(-1, 3)
            value, grad = self.energy_and_gradient(positions)
            values.append(value)
            gradients.append(grad.reshape(-1))
        reduced = torch.as_tensor(values, device=x.device, dtype=x.dtype)
        force = torch.as_tensor(np.stack(gradients, axis=0), device=x.device, dtype=x.dtype)
        return reduced, force


@dataclass(frozen=True)
class MolecularSampleSet:
    """Precomputed molecular coordinates and optional context."""

    positions_nm: torch.Tensor
    context: torch.Tensor | None = None
    reduced_potential: torch.Tensor | None = None
    reduced_gradient: torch.Tensor | None = None

    @classmethod
    def from_npz(cls, path: Path, context_dim: int = 0) -> "MolecularSampleSet":
        data = np.load(path)
        positions = torch.as_tensor(data["positions_nm"], dtype=torch.float32)
        context = None
        if "context" in data:
            context = torch.as_tensor(data["context"], dtype=torch.float32)
        elif context_dim:
            context = torch.zeros(positions.shape[0], context_dim)
        reduced = torch.as_tensor(data["reduced_potential"], dtype=torch.float32) if "reduced_potential" in data else None
        grad = torch.as_tensor(data["reduced_gradient"], dtype=torch.float32) if "reduced_gradient" in data else None
        return cls(positions.reshape(positions.shape[0], -1), context, reduced, grad.reshape(grad.shape[0], -1) if grad is not None else None)


class MolecularBridgeTask:
    """Batches precomputed source/target molecular coordinates for training."""

    def __init__(self, source: MolecularSampleSet, target: MolecularSampleSet, batch_size: int = 32) -> None:
        if source.positions_nm.shape[1] != target.positions_nm.shape[1]:
            raise ValueError("source and target positions must have the same flattened dimension")
        self.source = source
        self.target = target
        self.batch_size = batch_size

    @property
    def state_dim(self) -> int:
        return int(self.source.positions_nm.shape[1])

    def batch_stream(self, seed: int = 20260524):
        generator = torch.Generator().manual_seed(seed)
        while True:
            src_idx = torch.randint(0, self.source.positions_nm.shape[0], (self.batch_size,), generator=generator)
            tgt_idx = torch.randint(0, self.target.positions_nm.shape[0], (self.batch_size,), generator=generator)
            context = self.target.context[tgt_idx] if self.target.context is not None else None
            yield Batch(
                x0=self.source.positions_nm[src_idx],
                x1=self.target.positions_nm[tgt_idx],
                context=context,
                metadata={"source_index": src_idx, "target_index": tgt_idx},
            )


class TabulatedPotentialProvider:
    """Nearest-neighbour provider for precomputed reduced potentials/gradients."""

    def __init__(self, samples: MolecularSampleSet) -> None:
        if samples.reduced_potential is None or samples.reduced_gradient is None:
            raise ValueError("samples must include reduced_potential and reduced_gradient")
        self.samples = samples

    def __call__(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        idx = self._nearest_indices(x)
        return self.samples.reduced_potential.to(device=x.device, dtype=x.dtype)[idx]

    def reduced_and_force(self, x: torch.Tensor, context: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self._nearest_indices(x)
        reduced = self.samples.reduced_potential.to(device=x.device, dtype=x.dtype)[idx]
        gradient = self.samples.reduced_gradient.to(device=x.device, dtype=x.dtype)[idx]
        return reduced, gradient

    def _nearest_indices(self, x: torch.Tensor) -> torch.Tensor:
        refs = self.samples.positions_nm.to(device=x.device, dtype=x.dtype)
        return torch.cdist(x.detach(), refs).argmin(dim=1)


def make_harmonic_molecular_samples(
    sample_count: int,
    atom_count: int,
    center_shift: float,
    context_dim: int = 2,
    seed: int = 20260524,
    stiffness: float = 1.0,
) -> MolecularSampleSet:
    """Generate a deterministic tabulated molecular-like harmonic sample set."""

    generator = torch.Generator().manual_seed(seed)
    state_dim = atom_count * 3
    center = torch.zeros(state_dim)
    center[0::3] = center_shift
    positions = center + 0.05 * torch.randn(sample_count, state_dim, generator=generator)
    displacement = positions - center
    reduced = 0.5 * stiffness * displacement.square().sum(dim=-1)
    gradient = stiffness * displacement
    context = torch.zeros(sample_count, context_dim) if context_dim else None
    if context is not None:
        context[:, 0] = center_shift
        if context_dim > 1:
            context[:, 1] = stiffness
    return MolecularSampleSet(
        positions_nm=positions,
        context=context,
        reduced_potential=reduced,
        reduced_gradient=gradient,
    )
