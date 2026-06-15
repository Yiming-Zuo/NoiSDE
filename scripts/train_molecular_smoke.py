#!/usr/bin/env python3
"""Run a tabulated molecular NoiSDE training smoke job."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noisde.benchmarks import MolecularBridgeTask, MolecularSampleSet, TabulatedPotentialProvider, make_harmonic_molecular_samples
from noisde.config import load_yaml
from noisde.evaluation import BridgeEvaluationConfig, BridgeEvaluator
from noisde.export import write_run_index, write_seed_level_results
from noisde.models import NoiSDEModel, NoiSDEModelConfig
from noisde.training import NoiSDELossConfig, NoiSDEObjective, Trainer, TrainerConfig
from noisde.types import SafetyGates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/alanine_transport/smoke_tabulated.yaml")
    parser.add_argument("--outdir", default="tmp/molecular_smoke")
    return parser.parse_args()


def load_or_generate_samples(config: dict, role: str, fallback_seed: int) -> MolecularSampleSet:
    section = config[role]
    if section.get("npz"):
        return MolecularSampleSet.from_npz(ROOT / section["npz"], context_dim=int(config["task"].get("context_dim", 0)))
    return make_harmonic_molecular_samples(
        sample_count=int(config["task"]["sample_count"]),
        atom_count=int(config["task"]["atom_count"]),
        center_shift=float(section["center_shift"]),
        context_dim=int(config["task"].get("context_dim", 0)),
        seed=int(section.get("seed", fallback_seed)),
        stiffness=float(section.get("stiffness", 1.0)),
    )


def main() -> None:
    args = parse_args()
    config = load_yaml(ROOT / args.config)
    trainer_data = dict(config["trainer"])
    if trainer_data.get("checkpoint_dir") is not None:
        trainer_data["checkpoint_dir"] = ROOT / trainer_data["checkpoint_dir"]
    trainer_cfg = TrainerConfig(**trainer_data)
    torch.manual_seed(trainer_cfg.seed)

    source = load_or_generate_samples(config, "source", trainer_cfg.seed)
    target = load_or_generate_samples(config, "target", trainer_cfg.seed + 1)
    task = MolecularBridgeTask(source, target, batch_size=int(config["task"]["batch_size"]))
    source_provider = TabulatedPotentialProvider(source)
    target_provider = TabulatedPotentialProvider(target)

    model_cfg = NoiSDEModelConfig(**config["model"])
    loss_cfg = NoiSDELossConfig(**config["loss"])
    model = NoiSDEModel(model_cfg)
    objective = NoiSDEObjective(source_provider, target_provider, loss_cfg)
    trainer = Trainer(model, objective, trainer_cfg)
    history = trainer.fit(task.batch_stream(seed=trainer_cfg.seed))

    rows = []
    for item in history:
        step = int(item["step"])
        run_id = f"molecular_smoke__tabulated__noisde__step_{step:04d}"
        for metric, value in item.items():
            if metric == "step":
                continue
            rows.append({
                "tier": "molecular_smoke",
                "task": "tabulated_harmonic",
                "variant": "noisde",
                "seed": f"seed_{trainer_cfg.seed}",
                "run_id": run_id,
                "metric": metric,
                "value": value,
                "unit": "training",
                "config": args.config,
            })

    evaluator = BridgeEvaluator(
        source_provider,
        target_provider,
        BridgeEvaluationConfig(**config.get("validation", {})),
    )
    validation_batch = next(task.batch_stream(seed=trainer_cfg.seed + 1))
    evaluation = evaluator.evaluate(model, validation_batch)
    record = trainer.save_checkpoint(
        step=int(history[-1]["step"]),
        metrics=evaluation.metrics,
        score_vector=evaluation.score_vector,
    )
    selected = trainer.select_checkpoint(SafetyGates(**config["selection"]["safety_gates"]))
    validation_run_id = f"molecular_smoke__tabulated__noisde__validation_{record.step:04d}"
    for metric, value in evaluation.metrics.items():
        rows.append({
            "tier": "molecular_smoke_validation",
            "task": "tabulated_harmonic",
            "variant": "noisde",
            "seed": f"seed_{trainer_cfg.seed + 1}",
            "run_id": validation_run_id,
            "metric": metric,
            "value": value,
            "unit": "validation",
            "config": args.config,
        })

    outdir = ROOT / args.outdir
    metric_path = write_seed_level_results(rows, outdir / "molecular_smoke_metrics.csv")
    index_path = write_run_index(rows, outdir / "molecular_smoke_run_index.csv")
    print(f"wrote {metric_path.relative_to(ROOT)}")
    print(f"wrote {index_path.relative_to(ROOT)}")
    print(f"final_loss={history[-1]['loss']:.6f}")
    print(f"selected_checkpoint_step={selected.step}")
    print("validation_metrics=" + ",".join(f"{key}:{value:.6f}" for key, value in sorted(evaluation.metrics.items())))


if __name__ == "__main__":
    main()
