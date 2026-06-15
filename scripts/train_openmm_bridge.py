#!/usr/bin/env python3
"""Train NoiSDE with an OpenMM reduced-potential oracle and NPZ sample stores."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noisde.benchmarks import MolecularBridgeTask, MolecularSampleSet, OpenMMReducedPotential, OpenMMTaskConfig
from noisde.config import load_yaml
from noisde.evaluation import BridgeEvaluationConfig, BridgeEvaluator
from noisde.export import write_run_index, write_seed_level_results
from noisde.models import NoiSDEModel, NoiSDEModelConfig
from noisde.training import NoiSDELossConfig, NoiSDEObjective, Trainer, TrainerConfig
from noisde.types import SafetyGates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/alanine_transport/default_alanine_gbn2.yaml")
    parser.add_argument("--outdir", default="tmp/openmm_bridge")
    return parser.parse_args()


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_samples(config: dict, role: str) -> MolecularSampleSet:
    section = config[role]
    path = project_path(section["npz"])
    if not path.exists():
        raise FileNotFoundError(f"Missing {role} sample store: {path}")
    return MolecularSampleSet.from_npz(path, context_dim=int(config["task"].get("context_dim", 0)))


def openmm_config(config: dict, key: str) -> OpenMMTaskConfig:
    data = dict(config[key])
    data["structure_path"] = project_path(data["structure_path"])
    data["forcefield_files"] = tuple(data.get("forcefield_files", ("amber14-all.xml", "implicit/gbn2.xml")))
    return OpenMMTaskConfig(**data)


def trainer_config(config: dict) -> TrainerConfig:
    data = dict(config["trainer"])
    if data.get("checkpoint_dir") is not None:
        data["checkpoint_dir"] = project_path(data["checkpoint_dir"])
    return TrainerConfig(**data)


def resolved_model_config(config: dict, task: MolecularBridgeTask) -> NoiSDEModelConfig:
    data = dict(config["model"])
    if data.get("state_dim") == "auto":
        data["state_dim"] = task.state_dim
    if data.get("context_dim") == "auto":
        data["context_dim"] = 0 if task.target.context is None else int(task.target.context.shape[1])
    return NoiSDEModelConfig(**data)


def main() -> None:
    args = parse_args()
    config = load_yaml(project_path(args.config))
    trainer_cfg = trainer_config(config)
    torch.manual_seed(trainer_cfg.seed)

    source = load_samples(config, "source_samples")
    target = load_samples(config, "target_samples")
    task = MolecularBridgeTask(source, target, batch_size=int(config["task"]["batch_size"]))
    source_oracle = OpenMMReducedPotential(openmm_config(config, "source_potential"))
    target_oracle = OpenMMReducedPotential(openmm_config(config, config.get("target_potential_key", "target_potential")))

    model = NoiSDEModel(resolved_model_config(config, task))
    objective = NoiSDEObjective(source_oracle, target_oracle, NoiSDELossConfig(**config["loss"]))
    trainer = Trainer(model, objective, trainer_cfg)
    history = trainer.fit(task.batch_stream(seed=trainer_cfg.seed))

    rows = []
    for item in history:
        step = int(item["step"])
        run_id = f"openmm_bridge__{config['task']['name']}__{config['model']['variant']}__step_{step:06d}"
        for metric, value in item.items():
            if metric == "step":
                continue
            rows.append({
                "tier": "openmm_bridge",
                "task": config["task"]["name"],
                "variant": config["model"]["variant"],
                "seed": f"seed_{trainer_cfg.seed}",
                "run_id": run_id,
                "metric": metric,
                "value": value,
                "unit": "training",
                "config": args.config,
            })

    evaluator = BridgeEvaluator(source_oracle, target_oracle, BridgeEvaluationConfig(**config.get("validation", {})))
    evaluation = evaluator.evaluate(model, next(task.batch_stream(seed=trainer_cfg.seed + 1)))
    trainer.save_checkpoint(int(history[-1]["step"]), evaluation.metrics, evaluation.score_vector)
    selected = trainer.select_checkpoint(SafetyGates(**config["selection"]["safety_gates"]))
    validation_run_id = f"openmm_bridge__{config['task']['name']}__{config['model']['variant']}__validation"
    for metric, value in evaluation.metrics.items():
        rows.append({
            "tier": "openmm_bridge_validation",
            "task": config["task"]["name"],
            "variant": config["model"]["variant"],
            "seed": f"seed_{trainer_cfg.seed + 1}",
            "run_id": validation_run_id,
            "metric": metric,
            "value": value,
            "unit": "validation",
            "config": args.config,
        })

    outdir = project_path(args.outdir)
    metric_path = write_seed_level_results(rows, outdir / "openmm_bridge_metrics.csv")
    index_path = write_run_index(rows, outdir / "openmm_bridge_run_index.csv")
    print(f"wrote {metric_path.relative_to(ROOT)}")
    print(f"wrote {index_path.relative_to(ROOT)}")
    print(f"final_loss={history[-1]['loss']:.6f}")
    print(f"selected_checkpoint_step={selected.step}")


if __name__ == "__main__":
    main()
