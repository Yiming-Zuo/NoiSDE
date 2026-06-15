"""Safe Pareto-knee checkpoint selection."""
from __future__ import annotations

from noisde.types import CheckpointRecord, SafetyGates


def _dominates(left: dict[str, float], right: dict[str, float]) -> bool:
    keys = sorted(set(left) & set(right))
    return all(left[key] >= right[key] for key in keys) and any(left[key] > right[key] for key in keys)


def safe_records(records: list[CheckpointRecord], gates: SafetyGates) -> list[CheckpointRecord]:
    return [
        record
        for record in records
        if record.metrics.get("R_high_E", float("inf")) <= gates.r_high_e_max
        and record.metrics.get("D_E", float("inf")) <= gates.d_e_max
    ]


def pareto_front(records: list[CheckpointRecord]) -> list[CheckpointRecord]:
    front: list[CheckpointRecord] = []
    for candidate in records:
        if not any(_dominates(other.score_vector, candidate.score_vector) for other in records if other is not candidate):
            front.append(candidate)
    return front


def select_pareto_knee(records: list[CheckpointRecord], gates: SafetyGates) -> CheckpointRecord:
    safe = safe_records(records, gates)
    if not safe:
        raise ValueError("No checkpoint satisfies the safety gates")
    front = pareto_front(safe)
    return min(
        front,
        key=lambda record: sum((1.0 - float(value)) ** 2 for value in record.score_vector.values()) ** 0.5,
    )
