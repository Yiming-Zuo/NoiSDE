#!/usr/bin/env python3
"""Run a small internal NoiSDE toy-training smoke job."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noisde.benchmarks import ToyBridgeTask, ToyTaskConfig
from noisde.config import load_yaml
from noisde.evaluation import BridgeEvaluationConfig, BridgeEvaluator
from noisde.export import write_run_index, write_seed_level_results
from noisde.models import NoiSDEModel, NoiSDEModelConfig
from noisde.training import NoiSDELossConfig, NoiSDEObjective, Trainer, TrainerConfig
from noisde.types import SafetyGates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/toy_sampling/smoke_double_well.yaml")
    parser.add_argument("--outdir", default="tmp/toy_smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(ROOT / args.config)
    task_cfg = ToyTaskConfig(**config["task"])
    model_cfg = NoiSDEModelConfig(**config["model"])
    loss_cfg = NoiSDELossConfig(**config["loss"])
    trainer_data = dict(config["trainer"])
    if trainer_data.get("checkpoint_dir") is not None:
        trainer_data["checkpoint_dir"] = ROOT / trainer_data["checkpoint_dir"]
    trainer_cfg = TrainerConfig(**trainer_data)

    torch.manual_seed(trainer_cfg.seed)
    task = ToyBridgeTask(task_cfg)
    model = NoiSDEModel(model_cfg)
    objective = NoiSDEObjective(task.source_reduced_potential, task.target_reduced_potential, loss_cfg)
    trainer = Trainer(model, objective, trainer_cfg)
    history = trainer.fit(task.batch_stream(seed=trainer_cfg.seed))

    rows = []
    for item in history:
        step = int(item["step"])
        run_id = f"toy_smoke__{task_cfg.kind}__noisde__step_{step:04d}"
        for metric, value in item.items():
            if metric == "step":
                continue
            rows.append({
                "tier": "toy_smoke",
                "task": task_cfg.kind,
                "variant": "noisde",
                "seed": f"seed_{trainer_cfg.seed}",
                "run_id": run_id,
                "metric": metric,
                "value": value,
                "unit": "training",
                "config": args.config,
            })

    validation = config.get("validation", {})
    evaluator = BridgeEvaluator(
        task.source_reduced_potential,
        task.target_reduced_potential,
        BridgeEvaluationConfig(**validation),
    )
    validation_batch = next(task.batch_stream(seed=trainer_cfg.seed + 1))
    evaluation = evaluator.evaluate(model, validation_batch)
    record = trainer.save_checkpoint(
        step=int(history[-1]["step"]),
        metrics=evaluation.metrics,
        score_vector=evaluation.score_vector,
    )
    gates = SafetyGates(**config["selection"]["safety_gates"])
    selected = trainer.select_checkpoint(gates)

    validation_run_id = f"toy_smoke__{task_cfg.kind}__noisde__validation_{record.step:04d}"
    for metric, value in evaluation.metrics.items():
        rows.append({
            "tier": "toy_smoke_validation",
            "task": task_cfg.kind,
            "variant": "noisde",
            "seed": f"seed_{trainer_cfg.seed + 1}",
            "run_id": validation_run_id,
            "metric": metric,
            "value": value,
            "unit": "validation",
            "config": args.config,
        })

    outdir = ROOT / args.outdir
    path = write_seed_level_results(rows, outdir / "toy_smoke_training_metrics.csv")
    index_path = write_run_index(rows, outdir / "toy_smoke_run_index.csv")
    print(f"wrote {path.relative_to(ROOT)}")
    print(f"wrote {index_path.relative_to(ROOT)}")
    print(f"final_loss={history[-1]['loss']:.6f}")
    print(f"selected_checkpoint_step={selected.step}")
    print("validation_metrics=" + ",".join(f"{key}:{value:.6f}" for key, value in sorted(evaluation.metrics.items())))


if __name__ == "__main__":
    main()
