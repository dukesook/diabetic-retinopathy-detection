from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolicDecision:
    left_level: int
    right_level: int
    patient_level: int
    referable_dr: int
    objective_cost: float


def _negative_log_cost(probability: float) -> float:
    return -math.log(max(probability, 1e-12))


def solve_patient_decision(
    left_probs: list[float],
    right_probs: list[float],
    referral_threshold: int = 2,
    referral_bonus: float = 0.0,
) -> SymbolicDecision:
    if len(left_probs) != 5 or len(right_probs) != 5:
        raise ValueError("left_probs and right_probs must each contain 5 class probabilities")

    best: SymbolicDecision | None = None
    for left_level in range(5):
        for right_level in range(5):
            patient_level = max(left_level, right_level)
            referable_dr = int(patient_level >= referral_threshold)
            cost = _negative_log_cost(left_probs[left_level]) + _negative_log_cost(right_probs[right_level])
            cost -= referral_bonus * referable_dr

            candidate = SymbolicDecision(
                left_level=left_level,
                right_level=right_level,
                patient_level=patient_level,
                referable_dr=referable_dr,
                objective_cost=cost,
            )
            if best is None or candidate.objective_cost < best.objective_cost:
                best = candidate

    if best is None:
        raise RuntimeError("no feasible decision found")
    return best
