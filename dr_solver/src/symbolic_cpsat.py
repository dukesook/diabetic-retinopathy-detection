"""Symbolic component: patient-level decision encoded in OR-Tools CP-SAT.

Variables
    x_L[g], x_R[g]  Boolean one-hot grade per eye, g in 0..4
    left_level, right_level, patient_level  in 0..4
    referable_dr    Boolean
    mismatch        Boolean, true iff |left_level - right_level| >= 2

Hard constraints (Oct 1 slide 6)
    exactly one grade per eye
    patient_level = max(left_level, right_level)
    referable_dr = 1  iff  patient_level >= 2

Objective (minimize, integer weights = round(SCALE * value))
    sum_g  -log p_L[g] * x_L[g]  +  sum_g -log p_R[g] * x_R[g]     (CNN evidence)
    + lam  * mismatch                                             (bilateral consistency, soft)
    - beta * referable_dr                                         (referral cost term, Oct 1)
    + mu   * sum_{a,b} W[a][b] * y[a][b]                          (learned pair compatibility, soft)

y[a][b] is a Boolean that is true iff left = a and right = b (linked by hard channeling
constraints). W[a][b] = -log( P(a,b) / (P(a) P(b)) ) is estimated from TRAINING labels.
With mu = 1 this is the Bayes rule for combining two per-eye posteriors under the
assumption that the two images are independent given the two grades:
    P(a, b | imgL, imgR)  is proportional to  p(a | imgL) p(b | imgR) P(a,b) / (P(a) P(b)).

With lam = 0 the two eyes are independent, so the optimum is per-eye argmax + max
(the learned-only rule) unless beta pushes a borderline patient to "refer".
The lam term is what couples the two eyes.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from ortools.sat.python import cp_model

SCALE = 10_000
G = range(5)
REF_T = 2
MISMATCH_GAP = 2


@dataclass(frozen=True)
class Decision:
    left_level: int
    right_level: int
    patient_level: int
    referable_dr: int
    mismatch: int
    objective: int          # integer objective that CP-SAT minimized
    status: str
    solve_ms: float


def int_costs(probs) -> list[int]:
    return [int(round(SCALE * -math.log(max(float(p), 1e-12)))) for p in probs]


def int_weight(x: float) -> int:
    return int(round(SCALE * x))


def pair_weights(W, mu: float) -> list[list[int]] | None:
    if W is None or mu == 0:
        return None
    return [[int_weight(mu * float(W[a][b])) for b in G] for a in G]


def pair_matrix_from_labels(left_levels, right_levels, alpha: float = 1.0):
    """W[a][b] = -log(P(a,b) / (P(a)P(b))) from training patients, add-alpha smoothing."""
    import numpy as np

    C = np.full((5, 5), alpha, dtype=float)
    np.add.at(C, (np.asarray(left_levels, int), np.asarray(right_levels, int)), 1)
    J = C / C.sum()
    m = (J.sum(0) + J.sum(1)) / 2          # pooled per-eye marginal
    return (-np.log(J / np.outer(m, m))).tolist()


def objective_of(l: int, r: int, cl: list[int], cr: list[int], lam: float, beta: float,
                 pw: list[list[int]] | None = None) -> int:
    """Same integer objective, evaluated for one (left, right) assignment."""
    pat = max(l, r)
    return (cl[l] + cr[r] + int_weight(lam) * int(abs(l - r) >= MISMATCH_GAP)
            - int_weight(beta) * int(pat >= REF_T) + (pw[l][r] if pw else 0))


def solve_patient(left_probs, right_probs, lam: float = 0.0, beta: float = 0.0,
                  W=None, mu: float = 0.0) -> Decision:
    if len(left_probs) != 5 or len(right_probs) != 5:
        raise ValueError("need 5 class probabilities per eye")
    cl, cr = int_costs(left_probs), int_costs(right_probs)
    pw = pair_weights(W, mu)

    m = cp_model.CpModel()
    xL = [m.NewBoolVar(f"xL{g}") for g in G]
    xR = [m.NewBoolVar(f"xR{g}") for g in G]
    m.AddExactlyOne(xL)
    m.AddExactlyOne(xR)

    left = m.NewIntVar(0, 4, "left_level")
    right = m.NewIntVar(0, 4, "right_level")
    pat = m.NewIntVar(0, 4, "patient_level")
    m.Add(left == sum(g * xL[g] for g in G))
    m.Add(right == sum(g * xR[g] for g in G))
    m.AddMaxEquality(pat, [left, right])

    ref = m.NewBoolVar("referable_dr")
    m.Add(pat >= REF_T).OnlyEnforceIf(ref)
    m.Add(pat <= REF_T - 1).OnlyEnforceIf(ref.Not())

    diff = m.NewIntVar(-4, 4, "diff")
    absdiff = m.NewIntVar(0, 4, "absdiff")
    m.Add(diff == left - right)
    m.AddAbsEquality(absdiff, diff)
    mis = m.NewBoolVar("mismatch")
    m.Add(absdiff >= MISMATCH_GAP).OnlyEnforceIf(mis)
    m.Add(absdiff <= MISMATCH_GAP - 1).OnlyEnforceIf(mis.Not())

    pair_term = 0
    if pw is not None:
        # y[a][b] <-> (left = a AND right = b), via channeling with the one-hot rows/columns
        y = [[m.NewBoolVar(f"y{a}{b}") for b in G] for a in G]
        for a in G:
            m.Add(sum(y[a][b] for b in G) == xL[a])
        for b in G:
            m.Add(sum(y[a][b] for a in G) == xR[b])
        pair_term = sum(pw[a][b] * y[a][b] for a in G for b in G)

    m.Minimize(sum(cl[g] * xL[g] for g in G) + sum(cr[g] * xR[g] for g in G)
               + int_weight(lam) * mis - int_weight(beta) * ref + pair_term)

    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1          # tiny model; one worker is fastest and deterministic
    solver.parameters.max_time_in_seconds = 10.0
    t0 = time.perf_counter()
    st = solver.Solve(m)
    ms = (time.perf_counter() - t0) * 1000
    name = solver.StatusName(st)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT status {name}")
    return Decision(
        left_level=int(solver.Value(left)), right_level=int(solver.Value(right)),
        patient_level=int(solver.Value(pat)), referable_dr=int(solver.Value(ref)),
        mismatch=int(solver.Value(mis)), objective=int(round(solver.ObjectiveValue())),
        status=name, solve_ms=ms,
    )


def solve_bruteforce(left_probs, right_probs, lam: float = 0.0, beta: float = 0.0,
                     W=None, mu: float = 0.0) -> Decision:
    """Reference: enumerate all 25 (left, right) pairs with the identical integer objective.
    Used only in tests to check that the CP-SAT encoding is correct."""
    cl, cr = int_costs(left_probs), int_costs(right_probs)
    pw = pair_weights(W, mu)
    best = min(((objective_of(l, r, cl, cr, lam, beta, pw), l, r) for l in G for r in G))
    obj, l, r = best
    pat = max(l, r)
    return Decision(l, r, pat, int(pat >= REF_T), int(abs(l - r) >= MISMATCH_GAP), obj,
                    "BRUTEFORCE", 0.0)
