#!/usr/bin/env python3
"""Run a PF companion reweighting smoke evaluation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noisde.benchmarks import PFReweightingEvaluator, ToyBridgeTask, ToyTaskConfig, pf_audit_row
from noisde.config import load_yaml
from noisde.export import write_pf_audit_log, write_run_index, write_seed_level_results
from noisde.models import NoiSDEModel, NoiSDEModelConfig
from noisde.pf_companion import PFCompanionConfig, ProbabilityFlowCompanion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/free_energy/smoke_pf_reweighting.yaml")
    parser.add_argument("--outdir", default="tmp/pf_reweighting_smoke")
    return parser.parse_args()


def standard_normal_log_q0(x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
    return -0.5 * x.square().sum(dim=-1)


def main() -> None:
    args = parse_args()
    config = load_yaml(ROOT / args.config)
    seed = int(config["seed"])
    torch.manual_seed(seed)

    task = ToyBridgeTask(ToyTaskConfig(**config["task"]))
    model = NoiSDEModel(NoiSDEModelConfig(**config["model"]))
    pf = ProbabilityFlowCompanion(PFCompanionConfig(**config["pf_companion"]))
    x0 = task.sample_source(batch_size=int(config["sample_count"]), generator=torch.Generator().manual_seed(seed))
    context = torch.zeros(x0.shape[0], task.config.context_dim) if task.config.context_dim else None
    _, _, log_weights = pf.log_weights(
        model,
        x0,
        context,
        task.source_reduced_potential,
        task.target_reduced_potential,
        standard_normal_log_q0,
    )
    result = PFReweightingEvaluator().evaluate(log_weights)

    run_id = f"pf_reweighting_smoke__{task.config.kind}__{config['variant']}__seed_{seed}"
    metric_rows = []
    for metric, value in {
        "Delta_f": result.delta_f,
        "ESS_N": result.ess_n,
        "CV2_w": result.cv2_w,
        "log_weight_995": result.log_weight_995,
        "sample_count": result.sample_count,
    }.items():
        metric_rows.append({
            "tier": "pf_reweighting_smoke",
            "task": task.config.kind,
            "variant": config["variant"],
            "seed": f"seed_{seed}",
            "run_id": run_id,
            "metric": metric,
            "value": value,
            "unit": "reduced",
            "config": args.config,
        })
    audit = pf_audit_row(
        result,
        audit_id=run_id,
        window_id=task.config.kind,
        seed=f"seed_{seed}",
        variant=config["variant"],
        pf_mode=pf.config.mode,
        hutchinson_m=int(config.get("hutchinson_M", 1)),
        ess_threshold=float(config["gates"]["ess_n_min"]),
        log_weight_995_max=float(config["gates"]["log_weight_995_max"]),
    )

    outdir = ROOT / args.outdir
    metric_path = write_seed_level_results(metric_rows, outdir / "pf_reweighting_metrics.csv")
    index_path = write_run_index(metric_rows, outdir / "pf_reweighting_run_index.csv")
    audit_path = write_pf_audit_log([audit], outdir / "pf_reweighting_audit.csv")
    print(f"wrote {metric_path.relative_to(ROOT)}")
    print(f"wrote {index_path.relative_to(ROOT)}")
    print(f"wrote {audit_path.relative_to(ROOT)}")
    print(f"delta_f={result.delta_f:.6f}")
    print(f"ess_n={result.ess_n:.6f}")
    print(f"allowed_claim={audit['allowed_claim']}")


if __name__ == "__main__":
    main()
