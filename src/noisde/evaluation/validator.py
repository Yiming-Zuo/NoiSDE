"""Hard validation metrics for checkpoint selection."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from noisde.sde import EulerMaruyamaSampler
from noisde.types import Batch, ReducedPotentialFn
from noisde.utility import UtilityNormalizer, compute_hard_metrics


@dataclass(frozen=True)
class BridgeEvaluationConfig:
    path_steps: int = 16
    high_energy_threshold: float = 10.0
    normalizer: UtilityNormalizer = field(default_factory=UtilityNormalizer)


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    score_vector: dict[str, float]


class BridgeEvaluator:
    """Evaluates hard validation coordinates used by the Pareto-knee rule."""

    def __init__(
        self,
        r_a_fn: ReducedPotentialFn,
        r_b_fn: ReducedPotentialFn,
        config: BridgeEvaluationConfig | None = None,
        mode_centers: torch.Tensor | None = None,
    ) -> None:
        self.r_a_fn = r_a_fn
        self.r_b_fn = r_b_fn
        self.config = config or BridgeEvaluationConfig()
        self.mode_centers = mode_centers
        self.sampler = EulerMaruyamaSampler(self.config.path_steps)

    def evaluate(self, model, batch: Batch) -> EvaluationResult:
        model.eval()
        with torch.enable_grad():
            path = self.sampler.rollout(model, batch.x0, batch.context, self.r_b_fn)
            flat_path = path.states.reshape(-1, path.states.shape[-1])
            flat_context = _repeat_context(batch.context, path.states.shape[1]) if batch.context is not None else None
            path_reduced = self.r_b_fn(flat_path, flat_context).reshape(path.states.shape[:2])
            r_a_x0 = self.r_a_fn(batch.x0, batch.context)
            r_b_x1 = self.r_b_fn(path.x1, batch.context)
            log_weights = -r_b_x1 + r_a_x0
            metrics = compute_hard_metrics(
                path,
                r_a_x0,
                r_b_x1,
                path_reduced,
                log_weights,
                self.mode_centers,
                self.config.high_energy_threshold,
            )
        return EvaluationResult(metrics=metrics, score_vector=self._score(metrics))

    def _score(self, metrics: dict[str, float]) -> dict[str, float]:
        higher = {"ESS_N", "A_swap", "C_mode"}
        out = {}
        for key, value in metrics.items():
            if key in {"ESS_N", "A_swap", "C_mode", "D_E", "CV2_w", "R_high_E"}:
                tensor_value = torch.tensor(float(value))
                out[key] = float(self.config.normalizer.normalize(key, tensor_value, key in higher))
        return out


def _repeat_context(context: torch.Tensor | None, repeats: int) -> torch.Tensor | None:
    if context is None:
        return None
    return context[:, None, :].expand(-1, repeats, -1).reshape(-1, context.shape[-1])
