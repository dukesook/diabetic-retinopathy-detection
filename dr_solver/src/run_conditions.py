"""Run every condition on the same test patients and save raw per-patient decisions.
All tuning (threshold t, lam, beta, mu) uses VAL only. The test split is only predicted.

Examples:
    python -m src.run_conditions --seed 42                     # probs from run "seed42"
    python -m src.run_conditions --tag b3_ens                  # probs from ensemble "b3_ens"
    python -m src.run_conditions --tag b3_ens --select_by recall_at_spec
Run name = tag (plus "_recall" when --select_by recall_at_spec).
Writes outputs/decisions_{C1..C7}_{run}.csv, results/val_grid_{run}.csv,
results/chosen_params_{run}.json
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from .common import DATA, OUTPUTS, REFERRAL_THRESHOLD, RESULTS, ensure_dirs, to_patient_table
from .metrics import qwk, referral_metrics
from .symbolic_cpsat import pair_matrix_from_labels, solve_patient

PCOLS = [f"p{k}" for k in range(5)]
LAM_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
BETA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0]
MU_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def load_patients(path) -> pd.DataFrame:
    eyes = pd.read_csv(path, dtype={"patient_id": str})
    pt = to_patient_table(eyes, ["level"] + PCOLS)
    pt["level_left"] = pt["level_left"].astype(int)
    pt["level_right"] = pt["level_right"].astype(int)
    pt["true_patient"] = pt[["level_left", "level_right"]].max(1)
    pt["true_ref"] = (pt["true_patient"] >= REFERRAL_THRESHOLD).astype(int)
    return pt


def probs(pt, eye) -> np.ndarray:
    return pt[[f"{c}_{eye}" for c in PCOLS]].to_numpy(dtype=float)


def ref_score(pt) -> np.ndarray:
    """Continuous referral score: max over eyes of P(grade >= 2)."""
    return np.maximum(probs(pt, "left")[:, 2:].sum(1), probs(pt, "right")[:, 2:].sum(1))


def _solve_one(pair, lam, beta, W, mu):
    d = solve_patient(pair[0], pair[1], lam=lam, beta=beta, W=W, mu=mu)
    return d.left_level, d.right_level, d.patient_level, d.referable_dr, d.solve_ms


def run_solver(PL, PR, lam, beta, workers, W=None, mu=0.0) -> np.ndarray:
    pairs = list(zip(PL.tolist(), PR.tolist()))
    fn = partial(_solve_one, lam=lam, beta=beta, W=W, mu=mu)
    if workers > 1:
        with Pool(workers) as pool:
            res = pool.map(fn, pairs, chunksize=64)
    else:
        res = [fn(p) for p in pairs]
    return np.array(res)


def frame(pt, left, right, ref, score=None, solve_ms=None) -> pd.DataFrame:
    out = pt[["patient_id", "level_left", "level_right", "true_patient", "true_ref"]].copy()
    out["pred_left"] = np.asarray(left, dtype=int)
    out["pred_right"] = np.asarray(right, dtype=int)
    out["pred_patient"] = np.maximum(out["pred_left"], out["pred_right"])
    out["pred_ref"] = np.asarray(ref, dtype=int)
    out["score"] = score if score is not None else np.nan
    out["solve_ms"] = solve_ms if solve_ms is not None else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None, help="probability run name; default seed{seed}")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--select_by", default="f1", choices=["f1", "recall_at_spec"],
                    help="val criterion: referral F1, or recall subject to specificity >= --min_spec")
    ap.add_argument("--min_spec", type=float, default=0.90)
    ap.add_argument("--limit", type=int, default=None, help="limit patients (smoke test)")
    a = ap.parse_args()
    ensure_dirs()
    tag = a.tag or f"seed{a.seed}"
    run = tag if a.select_by == "f1" else f"{tag}_recall"

    val = load_patients(OUTPUTS / f"probs_{tag}_val.csv")
    test = load_patients(OUTPUTS / f"probs_{tag}_test.csv")
    if a.limit:
        val, test = val.head(a.limit), test.head(a.limit)
    print(f"run {run}: val patients {len(val)}  test patients {len(test)}")

    def val_crit(y, p):
        m = referral_metrics(y, p)
        if a.select_by == "f1":
            return m["f1"]
        return m["recall"] if m["specificity"] >= a.min_spec else -1

    def save(cond, df):
        df.to_csv(OUTPUTS / f"decisions_{cond}_{run}.csv", index=False)

    # ---- training-set knowledge used by the symbolic layer ------------------------------
    splits = pd.read_csv(DATA / "splits.csv", dtype={"patient_id": str})
    tr = splits[splits.split == "train"]
    prior = np.bincount(tr["level"], minlength=5) / len(tr)
    trp = tr.pivot(index="patient_id", columns="eye", values="level").dropna()
    W = pair_matrix_from_labels(trp["left"].astype(int), trp["right"].astype(int))

    # ---- C1 learned-only rule ---------------------------------------------------------
    def c1(pt):
        L, R = probs(pt, "left").argmax(1), probs(pt, "right").argmax(1)
        return L, R, (np.maximum(L, R) >= REFERRAL_THRESHOLD).astype(int)

    L, R, ref = c1(test)
    save("C1", frame(test, L, R, ref, ref_score(test)))

    # ---- C2 learned-only + threshold tuned on val ------------------------------------
    sv, st = ref_score(val), ref_score(test)
    t_grid = np.round(np.arange(0.02, 0.96, 0.01), 2)
    t_scores = [val_crit(val["true_ref"], sv > t) for t in t_grid]
    t_best = float(t_grid[int(np.argmax(t_scores))])
    save("C2", frame(test, L, R, (st > t_best).astype(int), st))
    print(f"C2 threshold t = {t_best} (val {a.select_by} {max(t_scores):.4f})")

    # ---- C3 symbolic-only: training-set class priors instead of CNN evidence ---------
    PP = np.tile(prior, (len(test), 1))
    res = run_solver(PP, PP, 0.0, 0.0, a.workers)
    save("C3", frame(test, res[:, 0], res[:, 1], res[:, 3], None, res[:, 4]))

    PLv, PRv = probs(val, "left"), probs(val, "right")
    PLt, PRt = probs(test, "left"), probs(test, "right")

    def grid_row(kind, lam, beta, mu, r):
        m = referral_metrics(val["true_ref"], r[:, 3])
        return {"kind": kind, "lam": lam, "beta": beta, "mu": mu,
                "crit": val_crit(val["true_ref"], r[:, 3]),
                "val_f1": m["f1"], "val_recall": m["recall"], "val_specificity": m["specificity"],
                "val_eye_qwk": qwk(np.concatenate([val.level_left, val.level_right]),
                                   np.concatenate([r[:, 0], r[:, 1]])),
                "val_patient_qwk": qwk(val["true_patient"], r[:, 2])}

    # ---- grid: bilateral mismatch (lam) x beta ----------------------------------------
    rows = []
    for lam in LAM_GRID:
        for beta in BETA_GRID:
            rows.append(grid_row("mismatch", lam, beta, 0.0, run_solver(PLv, PRv, lam, beta, a.workers)))
            print(rows[-1], flush=True)
    # ---- grid: learned pair compatibility (mu) x beta ---------------------------------
    for mu in MU_GRID:
        for beta in BETA_GRID:
            rows.append(grid_row("pair", 0.0, beta, mu,
                                 run_solver(PLv, PRv, 0.0, beta, a.workers, W=W, mu=mu)))
            print(rows[-1], flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(RESULTS / f"val_grid_{run}.csv", index=False)

    # ties broken toward the simplest model (smaller weights)
    gm = grid[grid.kind == "mismatch"]
    g0 = gm[gm.lam == 0].sort_values(["crit", "beta"], ascending=[False, True]).iloc[0]
    g5 = gm.sort_values(["crit", "lam", "beta"], ascending=[False, True, True]).iloc[0]
    gp = grid[grid.kind == "pair"].sort_values(["crit", "mu", "beta"], ascending=[False, True, True]).iloc[0]

    r = run_solver(PLt, PRt, 0.0, float(g0.beta), a.workers)
    save("C4", frame(test, r[:, 0], r[:, 1], r[:, 3], None, r[:, 4]))
    r = run_solver(PLt, PRt, float(g5.lam), float(g5.beta), a.workers)
    save("C5", frame(test, r[:, 0], r[:, 1], r[:, 3], None, r[:, 4]))
    r = run_solver(PLt, PRt, 0.0, float(gp.beta), a.workers, W=W, mu=float(gp.mu))
    save("C7", frame(test, r[:, 0], r[:, 1], r[:, 3], None, r[:, 4]))

    # ---- C6 oracle: true per-eye labels into the solver ------------------------------
    eye1 = np.eye(5) * (1 - 4e-6) + 1e-6
    OL, OR = eye1[test["level_left"].to_numpy()], eye1[test["level_right"].to_numpy()]
    r = run_solver(OL, OR, 0.0, 0.0, a.workers)
    save("C6", frame(test, r[:, 0], r[:, 1], r[:, 3], None, r[:, 4]))

    chosen = {"run": run, "probs_tag": tag, "select_by": a.select_by,
              "min_spec": a.min_spec if a.select_by != "f1" else None,
              "C2_threshold": t_best,
              "C4_lam": 0.0, "C4_beta": float(g0.beta),
              "C5_lam": float(g5.lam), "C5_beta": float(g5.beta),
              "C7_mu": float(gp.mu), "C7_beta": float(gp.beta),
              "val_crit": {"C2": float(max(t_scores)), "C4": float(g0.crit),
                           "C5": float(g5.crit), "C7": float(gp.crit)},
              "train_prior": prior.tolist(),
              "pair_matrix_W": [[round(x, 4) for x in row] for row in W]}
    with open(RESULTS / f"chosen_params_{run}.json", "w") as f:
        json.dump(chosen, f, indent=2)
    print(json.dumps({k: v for k, v in chosen.items() if k != "pair_matrix_W"}, indent=2))


if __name__ == "__main__":
    main()
