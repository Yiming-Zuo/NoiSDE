from __future__ import annotations

from pathlib import Path

import pytest
import torch

from noisde.benchmarks import (
    MolecularBridgeTask,
    MolecularSampleSet,
    PFReweightingEvaluator,
    TabulatedPotentialProvider,
    ToyBridgeTask,
    ToyTaskConfig,
    make_harmonic_molecular_samples,
    pf_audit_row,
)
from noisde.evaluation import BridgeEvaluationConfig, BridgeEvaluator, reduced_fep
from noisde.export import PF_AUDIT_COLUMNS, write_pf_audit_log, write_run_index, write_seed_level_results
from noisde.models import VARIANTS, NoiSDEModel, NoiSDEModelConfig
from noisde.pf_companion import PFCompanionConfig, ProbabilityFlowCompanion, identity_map_audit, linear_gaussian_audit
from noisde.training import NoiSDELossConfig, NoiSDEObjective, Trainer, TrainerConfig
from noisde.training.trainer import score_vector_from_metrics
from noisde.types import CheckpointRecord, SafetyGates
from noisde.utility import UtilityNormalizer, select_pareto_knee


def test_toy_training_step_is_executable() -> None:
    task = ToyBridgeTask(ToyTaskConfig(kind="double_well", state_dim=2, batch_size=8))
    model = NoiSDEModel(NoiSDEModelConfig(state_dim=2, context_dim=1, rank=2, hidden_dim=16, depth=1))
    objective = NoiSDEObjective(
        task.source_reduced_potential,
        task.target_reduced_potential,
        NoiSDELossConfig(path_steps=3, energy_threshold=8.0),
    )
    trainer = Trainer(model, objective, TrainerConfig(steps=2, lr=1.0e-3))

    history = trainer.fit(task.batch_stream(seed=7))

    assert len(history) == 2
    assert torch.isfinite(torch.tensor(history[-1]["loss"]))


def test_ablation_variants_change_diffusion_contract() -> None:
    x = torch.randn(3, 2)
    t = torch.zeros(3)
    context = torch.zeros(3, 1)
    reduced = torch.ones(3)
    force = torch.zeros_like(x)

    deterministic = NoiSDEModel(NoiSDEModelConfig(state_dim=2, context_dim=1, rank=2, hidden_dim=8, depth=1, variant="F0"))
    fixed = NoiSDEModel(NoiSDEModelConfig(state_dim=2, context_dim=1, rank=2, hidden_dim=8, depth=1, variant="A2", fixed_noise_scale=0.25))
    isotropic = NoiSDEModel(NoiSDEModelConfig(state_dim=2, context_dim=1, rank=2, hidden_dim=8, depth=1, variant="A3"))
    full = NoiSDEModel(NoiSDEModelConfig(state_dim=2, context_dim=1, rank=2, hidden_dim=8, depth=1, variant="A4"))
    diagonal = NoiSDEModel(NoiSDEModelConfig(state_dim=2, context_dim=1, rank=2, hidden_dim=8, depth=1, variant="B1"))

    assert "A4" in VARIANTS
    assert deterministic.diffusion_factor(x, t, context, reduced, force).abs().sum() == 0
    assert deterministic.diffusion_tensor(x, t, context, reduced, force).abs().sum() == 0
    assert fixed.diffusion_factor(x, t, context, reduced, force)[0, 0, 0] == pytest.approx(0.25)
    assert isotropic.diffusion_factor(x, t, context, reduced, force).diagonal(dim1=-2, dim2=-1).shape == (3, 2)
    assert full.diffusion_factor(x, t, context, reduced, force).shape == (3, 2, 2)
    diagonal_factor = diagonal.diffusion_factor(x, t, context, reduced, force)
    assert diagonal_factor.shape == (3, 2, 2)
    assert torch.allclose(diagonal_factor, torch.diag_embed(diagonal_factor.diagonal(dim1=-2, dim2=-1)))


def test_pareto_knee_uses_safety_gates() -> None:
    records = [
        CheckpointRecord(1, None, {"R_high_E": 0.01, "D_E": 0.05}, {"ESS_N": 0.5, "C_mode": 0.6}),
        CheckpointRecord(2, None, {"R_high_E": 0.02, "D_E": 0.04}, {"ESS_N": 0.7, "C_mode": 0.7}),
        CheckpointRecord(3, None, {"R_high_E": 0.80, "D_E": 0.02}, {"ESS_N": 1.0, "C_mode": 1.0}),
    ]

    selected = select_pareto_knee(records, SafetyGates(r_high_e_max=0.1, d_e_max=0.1))

    assert selected.step == 2


def test_score_vector_from_metrics_normalizes_direction() -> None:
    normalizer = UtilityNormalizer(
        bad={"ESS_N": 0.0, "R_high_E": 1.0},
        ref={"ESS_N": 1.0, "R_high_E": 0.0},
    )
    score = score_vector_from_metrics({"ESS_N": 0.5, "R_high_E": 0.2}, normalizer)

    assert score["ESS_N"] == pytest.approx(0.5)
    assert score["R_high_E"] == pytest.approx(0.8)


def test_pf_companion_audits_are_exact_for_simple_fields() -> None:
    x = torch.randn(4, 2, requires_grad=True)
    matrix = torch.tensor([[0.3, 0.1], [0.0, -0.2]], dtype=x.dtype)

    identity = identity_map_audit(x)
    linear = linear_gaussian_audit(x, matrix)

    assert identity["max_abs_divergence"] == 0.0
    assert linear["max_abs_error"] < 1.0e-6


def test_bridge_evaluator_returns_checkpoint_coordinates() -> None:
    task = ToyBridgeTask(ToyTaskConfig(kind="ring", state_dim=2, batch_size=6))
    model = NoiSDEModel(NoiSDEModelConfig(state_dim=2, context_dim=1, rank=2, hidden_dim=12, depth=1))
    batch = next(task.batch_stream(seed=8))
    evaluator = BridgeEvaluator(
        task.source_reduced_potential,
        task.target_reduced_potential,
        BridgeEvaluationConfig(path_steps=2, high_energy_threshold=8.0),
    )

    result = evaluator.evaluate(model, batch)

    assert {"ESS_N", "A_swap", "C_mode", "D_E", "CV2_w", "R_high_E"}.issubset(result.metrics)
    assert result.score_vector
    assert all(torch.isfinite(torch.tensor(value)) for value in result.metrics.values())


def test_pf_companion_integrates_density_on_toy_model() -> None:
    task = ToyBridgeTask(ToyTaskConfig(kind="double_well", state_dim=2, batch_size=4))
    model = NoiSDEModel(NoiSDEModelConfig(state_dim=2, context_dim=1, rank=2, hidden_dim=8, depth=1))
    batch = next(task.batch_stream(seed=9))
    pf = ProbabilityFlowCompanion(PFCompanionConfig(mode="current", num_steps=2))

    def log_q0(x: torch.Tensor, context: torch.Tensor | None) -> torch.Tensor:
        return -0.5 * x.square().sum(dim=-1)

    z1, log_q1, log_w = pf.log_weights(
        model,
        batch.x0,
        batch.context,
        task.source_reduced_potential,
        task.target_reduced_potential,
        log_q0,
    )

    assert z1.shape == batch.x0.shape
    assert torch.isfinite(log_q1).all()
    assert torch.isfinite(log_w).all()


def test_export_writers_use_manuscript_columns(tmp_path: Path) -> None:
    rows = [
        {
            "tier": "toy_sampling",
            "task": "T1_double_well",
            "variant": "noisde",
            "seed": "R001",
            "run_id": "toy__noisde__R001",
            "metric": "ESS_N",
            "value": 0.5,
            "unit": "fraction",
            "config": "configs/experiments/toy_sampling/T1/noisde.yaml",
        }
    ]

    seed_path = write_seed_level_results(rows, tmp_path / "seed.csv")
    index_path = write_run_index(rows, tmp_path / "run_index.csv")

    assert seed_path.read_text(encoding="utf-8").splitlines()[0] == "tier,task,variant,seed,run_id,metric,value,unit,config"
    assert index_path.read_text(encoding="utf-8").splitlines()[0] == "tier,task,variant,seed,run_id,config"


def test_reduced_fep_is_finite() -> None:
    value = reduced_fep(torch.tensor([0.0, -1.0, -2.0]))

    assert torch.isfinite(value)


def test_molecular_tabulated_provider_batches_and_returns_forces() -> None:
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    reduced = torch.tensor([0.1, 0.2])
    gradient = torch.ones_like(positions) * 0.5
    samples = MolecularSampleSet(positions_nm=positions, reduced_potential=reduced, reduced_gradient=gradient)
    task = MolecularBridgeTask(samples, samples, batch_size=2)
    provider = TabulatedPotentialProvider(samples)
    batch = next(task.batch_stream(seed=11))

    values, forces = provider.reduced_and_force(batch.x0)

    assert batch.x0.shape == (2, 6)
    assert values.shape == (2,)
    assert forces.shape == batch.x0.shape


def test_molecular_tabulated_training_step_is_executable() -> None:
    source = make_harmonic_molecular_samples(8, atom_count=2, center_shift=0.0, context_dim=2, seed=12)
    target = make_harmonic_molecular_samples(8, atom_count=2, center_shift=0.2, context_dim=2, seed=13)
    task = MolecularBridgeTask(source, target, batch_size=4)
    source_provider = TabulatedPotentialProvider(source)
    target_provider = TabulatedPotentialProvider(target)
    model = NoiSDEModel(
        NoiSDEModelConfig(
            state_dim=6,
            context_dim=2,
            rank=2,
            hidden_dim=8,
            depth=1,
            diffusion_max_scale=0.05,
        )
    )
    objective = NoiSDEObjective(
        source_provider,
        target_provider,
        NoiSDELossConfig(path_steps=2, energy_threshold=2.0, lambda_energy=0.0),
    )
    trainer = Trainer(model, objective, TrainerConfig(steps=1, lr=1.0e-4, max_grad_norm=1.0))

    history = trainer.fit(task.batch_stream(seed=14))

    assert len(history) == 1
    assert torch.isfinite(torch.tensor(history[0]["loss"]))


def test_pf_reweighting_evaluator_reports_density_metrics() -> None:
    result = PFReweightingEvaluator().evaluate(torch.tensor([0.0, -0.5, -1.0, -1.5]))
    row = pf_audit_row(result, "audit-1", "w0", "R001", "noisde", "current", 1)

    assert result.sample_count == 4
    assert result.ess_n > 0
    assert result.cv2_w >= 0
    assert row["allowed_claim"] in {"density_dependent", "proposal_quality_only"}


def test_pf_audit_writer_uses_declared_columns(tmp_path: Path) -> None:
    result = PFReweightingEvaluator().evaluate(torch.tensor([0.0, -0.5, -1.0, -1.5]))
    row = pf_audit_row(result, "audit-1", "w0", "R001", "noisde", "current", 1)

    path = write_pf_audit_log([row], tmp_path / "pf.csv")

    assert path.read_text(encoding="utf-8").splitlines()[0] == ",".join(PF_AUDIT_COLUMNS)
