#!/usr/bin/env python3
"""Run a component-ablation smoke over the manuscript variant matrix."""
from __future__ import annotations

import argparse
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
from noisde.types import CheckpointRecord, SafetyGates
from noisde.utility import select_pareto_knee


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/core_ablation/smoke_variant_matrix.yaml")
    parser.add_argument("--outdir", default="tmp/ablation_smoke")
    return parser.parse_args()


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def build_trainer_config(config: dict) -> TrainerConfig:
    trainer_data = dict(config["trainer"])
    if trainer_data.get("checkpoint_dir") is not None:
        trainer_data["checkpoint_dir"] = project_path(trainer_data["checkpoint_dir"])
    return TrainerConfig(**trainer_data)


def variant_candidates(name: str, matrix_item: dict) -> list[tuple[str, dict]]:
    if name != "A2":
        return [(name, {})]
    return [
        (f"A2_sigma_{str(scale).replace('.', 'p')}", {"fixed_noise_scale": float(scale)})
        for scale in matrix_item.get("fixed_noise_grid", [0.1])
    ]


def main() -> None:
    args = parse_args()
    config = load_yaml(project_path(args.config))
    matrix = load_yaml(project_path(config["variant_matrix"]))
    requested = config.get("variants") or list(matrix["variants"])
    task = ToyBridgeTask(ToyTaskConfig(**config["task"]))
    loss_cfg = NoiSDELossConfig(**config["loss"])
    trainer_cfg = build_trainer_config(config)
    evaluator = BridgeEvaluator(
        task.source_reduced_potential,
        task.target_reduced_potential,
        BridgeEvaluationConfig(**config.get("validation", {})),
    )
    gates = SafetyGates(**config["selection"]["safety_gates"])

    rows = []
    selected_records: dict[str, CheckpointRecord] = {}
    selected_candidate_labels: dict[str, str] = {}

    for index, variant_name in enumerate(requested):
        matrix_item = matrix["variants"][variant_name]
        candidate_records: list[CheckpointRecord] = []
        candidate_labels: dict[int, str] = {}
        for candidate_index, (candidate_label, overrides) in enumerate(variant_candidates(variant_name, matrix_item)):
            seed = trainer_cfg.seed + index * 100 + candidate_index
            torch.manual_seed(seed)
            model_data = dict(config["model"])
            model_data["variant"] = matrix_item["model_variant"]
            model_data.update(overrides)
            if "over_noise_scale" in matrix_item:
                model_data["over_noise_scale"] = float(matrix_item["over_noise_scale"])
            model = NoiSDEModel(NoiSDEModelConfig(**model_data))
            trainer = Trainer(model, NoiSDEObjective(task.source_reduced_potential, task.target_reduced_potential, loss_cfg), trainer_cfg)
            history = trainer.fit(task.batch_stream(seed=seed))
            run_prefix = f"ablation_smoke__{task.config.kind}__{candidate_label}__seed_{seed}"
            for item in history:
                step = int(item["step"])
                run_id = f"{run_prefix}__step_{step:04d}"
                for metric, value in item.items():
                    if metric == "step":
                        continue
                    rows.append({
                        "tier": "core_ablation_smoke",
                        "task": task.config.kind,
                        "variant": candidate_label,
                        "seed": f"seed_{seed}",
                        "run_id": run_id,
                        "metric": metric,
                        "value": value,
                        "unit": "training",
                        "config": args.config,
                    })

            validation_batch = next(task.batch_stream(seed=seed + 1))
            evaluation = evaluator.evaluate(model, validation_batch)
            record = CheckpointRecord(
                step=index * 1000 + candidate_index,
                path=None,
                metrics=evaluation.metrics,
                score_vector=evaluation.score_vector,
            )
            candidate_records.append(record)
            candidate_labels[record.step] = candidate_label
            validation_run_id = f"{run_prefix}__validation"
            for metric, value in evaluation.metrics.items():
                rows.append({
                    "tier": "core_ablation_smoke_validation",
                    "task": task.config.kind,
                    "variant": candidate_label,
                    "seed": f"seed_{seed + 1}",
                    "run_id": validation_run_id,
                    "metric": metric,
                    "value": value,
                    "unit": "validation",
                    "config": args.config,
                })

        selected = select_pareto_knee(candidate_records, gates)
        selected_records[variant_name] = selected
        selected_candidate_labels[variant_name] = candidate_labels[selected.step]
        rows.append({
            "tier": "core_ablation_smoke_selection",
            "task": task.config.kind,
            "variant": selected_candidate_labels[variant_name],
            "seed": "selection",
            "run_id": f"ablation_smoke__{task.config.kind}__{variant_name}__selected",
            "metric": "selected_candidate",
            "value": 1.0,
            "unit": "flag",
            "config": args.config,
        })

    outdir = project_path(args.outdir)
    metric_path = write_seed_level_results(rows, outdir / "ablation_smoke_metrics.csv")
    index_path = write_run_index(rows, outdir / "ablation_smoke_run_index.csv")
    print(f"wrote {metric_path.relative_to(ROOT)}")
    print(f"wrote {index_path.relative_to(ROOT)}")
    print("selected_candidates=" + ",".join(f"{name}:{selected_candidate_labels[name]}" for name in requested))
    print("selected_ess_n=" + ",".join(f"{name}:{selected_records[name].metrics['ESS_N']:.6f}" for name in requested))


if __name__ == "__main__":
    main()
