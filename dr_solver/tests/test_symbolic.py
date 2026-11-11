"""Correctness checks for the CP-SAT symbolic component.  Run:  pytest -q"""
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.symbolic_cpsat import int_costs, objective_of, solve_bruteforce, solve_patient  # noqa: E402
from src.symbolic_decision import solve_patient_decision  # original Oct 1 helper  # noqa: E402

N = 1000


def rand_probs(rng):
    # mix of confident and uncertain vectors
    alpha = rng.choice([0.2, 1.0, 5.0])
    return list(rng.dirichlet([alpha] * 5))


@pytest.fixture(scope="module")
def cases():
    rng = np.random.default_rng(0)
    return [(rand_probs(rng), rand_probs(rng)) for _ in range(N)]


def test_lam0_beta0_equals_argmax_max(cases):
    """(a) Without the coupling term, the optimizer reduces to the learned-only rule."""
    for pl, pr in cases:
        d = solve_patient(pl, pr, lam=0, beta=0)
        cl, cr = int_costs(pl), int_costs(pr)
        al, ar = int(np.argmax(pl)), int(np.argmax(pr))
        # compare objective (rounding can create exact ties between grades)
        assert d.objective == cl[al] + cr[ar]
        assert d.patient_level == max(d.left_level, d.right_level)


def test_matches_original_oct1_helper(cases):
    """Same answer as the original symbolic_decision.py (float objective, up to rounding)."""
    for pl, pr in cases:
        for beta in (0.0, 0.5, 1.0):
            ref = solve_patient_decision(pl, pr, referral_bonus=beta)
            d = solve_patient(pl, pr, lam=0, beta=beta)
            cl, cr = int_costs(pl), int_costs(pr)
            ours_float = objective_of(d.left_level, d.right_level, cl, cr, 0, beta) / 10_000
            assert abs(ours_float - ref.objective_cost) < 1e-3


@pytest.mark.parametrize("lam,beta", [(0.5, 0.0), (1.0, 0.5), (2.0, 1.0), (4.0, 0.0)])
def test_cpsat_equals_bruteforce(cases, lam, beta):
    """(b) CP-SAT optimum equals exhaustive search with the identical objective."""
    for pl, pr in cases[:300]:
        d = solve_patient(pl, pr, lam=lam, beta=beta)
        b = solve_bruteforce(pl, pr, lam=lam, beta=beta)
        assert d.objective == b.objective
        assert d.status == "OPTIMAL"


def test_constraints_hold(cases):
    """(c) Hard constraints always satisfied."""
    for pl, pr in cases[:300]:
        d = solve_patient(pl, pr, lam=1.0, beta=0.5)
        assert d.patient_level == max(d.left_level, d.right_level)
        assert d.referable_dr == int(d.patient_level >= 2)
        assert d.mismatch == int(abs(d.left_level - d.right_level) >= 2)


def test_coupling_changes_a_known_case():
    """Hand example: left eye clearly grade 0, right eye torn between 0 and 3.
    Argmax picks right=3 (mismatch); with lam the solver prefers right=0."""
    left = [0.96, 0.01, 0.01, 0.01, 0.01]
    right = [0.40, 0.05, 0.05, 0.45, 0.05]
    assert solve_patient(left, right, lam=0).right_level == 3
    assert solve_patient(left, right, lam=1.0).right_level == 0


# ---- learned pair compatibility (C7) -------------------------------------------------
from src.symbolic_cpsat import pair_matrix_from_labels  # noqa: E402


@pytest.fixture(scope="module")
def W():
    rng = np.random.default_rng(1)
    base = rng.choice(5, 3000, p=[.7, .08, .15, .04, .03])
    left = np.clip(base + rng.choice([-1, 0, 0, 0, 1], 3000), 0, 4)
    right = np.clip(base + rng.choice([-1, 0, 0, 0, 1], 3000), 0, 4)
    return pair_matrix_from_labels(left, right)


@pytest.mark.parametrize("mu,lam,beta", [(1.0, 0, 0), (0.5, 0, 0.5), (1.5, 1.0, 0.0)])
def test_pair_term_cpsat_equals_bruteforce(cases, W, mu, lam, beta):
    for pl, pr in cases[:300]:
        d = solve_patient(pl, pr, lam=lam, beta=beta, W=W, mu=mu)
        b = solve_bruteforce(pl, pr, lam=lam, beta=beta, W=W, mu=mu)
        assert d.objective == b.objective
        assert d.status == "OPTIMAL"


def test_pair_term_is_bayes_rule(cases, W):
    """mu = 1: the solver picks argmax of p(a|L) p(b|R) P(a,b) / (P(a)P(b))."""
    Wn = np.array(W)
    for pl, pr in cases[:300]:
        d = solve_patient(pl, pr, W=W, mu=1.0)
        score = np.log(np.maximum(np.outer(pl, pr), 1e-12)) - Wn
        a, b = np.unravel_index(score.argmax(), score.shape)
        cl, cr = int_costs(pl), int_costs(pr)
        from src.symbolic_cpsat import objective_of, pair_weights
        pw = pair_weights(W, 1.0)
        assert objective_of(d.left_level, d.right_level, cl, cr, 0, 0, pw) <= \
            objective_of(int(a), int(b), cl, cr, 0, 0, pw)
