"""Minimal research trainer for NoiSDE."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from noisde.types import Batch, CheckpointRecord, SafetyGates
from noisde.utility import UtilityNormalizer, select_pareto_knee


@dataclass(frozen=True)
class TrainerConfig:
    steps: int = 1000
    lr: float = 1.0e-3
    checkpoint_dir: Path | None = None
    seed: int = 20260524
    max_grad_norm: float | None = None


class Trainer:
    """Small trainer that keeps the algorithmic contract executable."""

    def __init__(self, model, objective, config: TrainerConfig | None = None) -> None:
        self.model = model
        self.objective = objective
        self.config = config or TrainerConfig()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr)
        self.generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        self.history: list[dict[str, float]] = []
        self.checkpoints: list[CheckpointRecord] = []

    def fit(self, batches: Iterable[Batch]) -> list[dict[str, float]]:
        iterator = iter(batches)
        self.model.train()
        for step in range(1, self.config.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(batches)
                batch = next(iterator)
            self.optimizer.zero_grad()
            loss, parts = self.objective(self.model, batch, self.generator)
            loss.backward()
            if self.config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            row = {key: float(value.detach().cpu()) for key, value in parts.items()}
            row["step"] = float(step)
            self.history.append(row)
        return self.history

    def save_checkpoint(self, step: int, metrics: dict[str, float], score_vector: dict[str, float]) -> CheckpointRecord:
        path = None
        if self.config.checkpoint_dir is not None:
            self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = self.config.checkpoint_dir / f"checkpoint_{step:06d}.pt"
            torch.save({"step": step, "model": self.model.state_dict(), "metrics": metrics}, path)
        record = CheckpointRecord(step=step, path=path, metrics=metrics, score_vector=score_vector)
        self.checkpoints.append(record)
        return record

    def select_checkpoint(self, gates: SafetyGates) -> CheckpointRecord:
        return select_pareto_knee(self.checkpoints, gates)


def score_vector_from_metrics(metrics: dict[str, float], normalizer: UtilityNormalizer) -> dict[str, float]:
    higher = {"ESS_N", "A_swap", "C_mode"}
    out = {}
    for key in ("ESS_N", "A_swap", "C_mode", "D_E", "CV2_w", "R_high_E"):
        if key in metrics:
            value = torch.tensor(float(metrics[key]))
            out[key] = float(normalizer.normalize(key, value, key in higher))
    return out
