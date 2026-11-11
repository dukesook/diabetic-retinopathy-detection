"""Evaluate saved decisions. Reads only outputs/decisions_*.csv (+ probs for failure cases).

Examples:
    python -m src.evaluate --runs seed42                       # one run  -> results/seed42/
    python -m src.evaluate --runs seed42 seed1 seed2 --name b0_3seeds
    python -m src.evaluate --runs b3_ens                        # ensemble
The first run is the main one (CIs, tests, plots); more runs add mean +/- std across runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from .common import OUTPUTS, RESULTS, ensure_dirs  # noqa: E402
from .metrics import auc, qwk, referral_metrics  # noqa: E402

NAMES = {
    "C1": "Learned-only (argmax + max rule)",
    "C2": "Learned-only + tuned threshold",
    "C3": "Symbolic-only (class priors)",
    "C4": "Combined, Oct 1 constraints (lam=0)",
    "C5": "Combined + bilateral mismatch",
    "C7": "Combined + learned pair compatibility",
    "C6": "Oracle labels into solver (check)",
}
METRICS = ["eye_qwk", "patient_qwk", "patient_acc", "ref_precision", "ref_recall",
           "ref_f1", "ref_specificity", "ref_fnr"]
SHOW = ["eye_qwk", "patient_qwk", "ref_recall", "ref_precision", "ref_f1", "ref_fnr", "ref_specificity"]
PAIRED = ["eye_qwk", "patient_qwk", "ref_f1", "ref_recall", "ref_fnr"]


def point_metrics(d: pd.DataFrame) -> dict:
    eye_t = np.concatenate([d.level_left, d.level_right])
    eye_p = np.concatenate([d.pred_left, d.pred_right])
    r = referral_metrics(d.true_ref, d.pred_ref)
    return {"eye_qwk": qwk(eye_t, eye_p), "patient_qwk": qwk(d.true_patient, d.pred_patient),
            "patient_acc": float((d.true_patient == d.pred_patient).mean()),
            "ref_precision": r["precision"], "ref_recall": r["recall"], "ref_f1": r["f1"],
            "ref_specificity": r["specificity"], "ref_fnr": r["fnr"]}


def boot_metrics(d: pd.DataFrame, idx_sets) -> dict:
    arr = d.reset_index(drop=True)
    out = {m: [] for m in METRICS}
    for idx in idx_sets:
        pm = point_metrics(arr.iloc[idx])
        for m in METRICS:
            out[m].append(pm[m])
    return {m: np.array(v) for m, v in out.items()}


def ci(v):
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def mcnemar(a_correct, b_correct) -> dict:
    a_correct, b_correct = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    b = int((a_correct & ~b_correct).sum())
    c = int((~a_correct & b_correct).sum())
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return {"A_right_B_wrong": b, "A_wrong_B_right": c, "mcnemar_p": p}


def fmt(v, c=None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.3f}" if c is None else f"{v:.3f} [{c[0]:.3f}, {c[1]:.3f}]"


def confusion_plot(ax, d, title):
    M = np.zeros((5, 5), int)
    np.add.at(M, (d.true_patient.to_numpy(), d.pred_patient.to_numpy()), 1)
    ax.imshow(M, cmap="Blues")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, M[i, j], ha="center", va="center",
                    color="white" if M[i, j] > M.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("predicted patient grade"); ax.set_ylabel("true patient grade")
    ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_title(title, fontsize=10)


def roc_points(y, s):
    order = np.argsort(-s)
    y = np.asarray(y)[order]
    tpr = np.concatenate([[0], np.cumsum(y) / max(y.sum(), 1)])
    fpr = np.concatenate([[0], np.cumsum(1 - y) / max((1 - y).sum(), 1)])
    return fpr, tpr


def load_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=None, help="run names, e.g. seed42 b3_ens")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="legacy: same as --runs seed{S}")
    ap.add_argument("--name", default=None, help="output folder under results/ (default: first run)")
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()
    ensure_dirs()
    runs = a.runs or [f"seed{s}" for s in (a.seeds or [42])]
    R0 = runs[0]
    outdir = RESULTS / (a.name or R0)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    dec = {r: {c: pd.read_csv(OUTPUTS / f"decisions_{c}_{r}.csv", dtype={"patient_id": str})
               for c in NAMES if (OUTPUTS / f"decisions_{c}_{r}.csv").exists()} for r in runs}
    conds = list(dec[R0].keys())
    n = len(dec[R0][conds[0]])
    idx_sets = [rng.integers(0, n, n) for _ in range(a.n_boot)]  # same resamples for every condition
    boots = {c: boot_metrics(dec[R0][c], idx_sets) for c in conds}

    # ---- main table -------------------------------------------------------------------
    rows = []
    for c in conds:
        d = dec[R0][c]
        pm = point_metrics(d)
        row = {"condition": c, "name": NAMES[c]}
        for m in METRICS:
            row[m] = pm[m]
            row[m + "_lo"], row[m + "_hi"] = ci(boots[c][m])
        if len(runs) > 1:
            per = [point_metrics(dec[r][c]) for r in runs if c in dec[r]]
            for m in METRICS:
                v = [p[m] for p in per]
                row[m + "_runs_mean"], row[m + "_runs_std"] = float(np.mean(v)), float(np.std(v, ddof=1))
        row["solver_ms_median"] = float(d.solve_ms.median()) if d.solve_ms.notna().any() else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(outdir / "main_table.csv", index=False)

    lines = [f"Run {R0}: test set {n} patients, 95% bootstrap CI ({a.n_boot} resamples)\n",
             "| Condition | " + " | ".join(SHOW) + " |", "|" + "---|" * (len(SHOW) + 1)]
    for r in rows:
        lines.append(f"| {r['condition']} {r['name']} | " + " | ".join(
            fmt(r[m], (r[m + '_lo'], r[m + '_hi'])) for m in SHOW) + " |")
    if len(runs) > 1:
        lines += ["", f"Across runs {runs} (mean +/- std):", "",
                  "| Condition | " + " | ".join(SHOW) + " |", "|" + "---|" * (len(SHOW) + 1)]
        for r in rows:
            if f"{SHOW[0]}_runs_mean" in r:
                lines.append(f"| {r['condition']} | " + " | ".join(
                    f"{r[m+'_runs_mean']:.3f} +/- {r[m+'_runs_std']:.3f}" for m in SHOW) + " |")

    # ---- paired comparisons: each combined system vs each baseline --------------------
    correct = {c: (dec[R0][c].pred_ref == dec[R0][c].true_ref).to_numpy() for c in conds}
    comps = []
    for A in ("C5", "C7"):
        for B in ("C1", "C2", "C4", "C5"):
            if A in conds and B in conds and A != B:
                row = {"A": A, "B": B, **mcnemar(correct[A], correct[B])}
                pa, pb = point_metrics(dec[R0][A]), point_metrics(dec[R0][B])
                for m in PAIRED:
                    diff = boots[A][m] - boots[B][m]
                    row[f"{m}_diff"] = pa[m] - pb[m]
                    row[f"{m}_diff_lo"], row[f"{m}_diff_hi"] = ci(diff)
                comps.append(row)
    comp_df = pd.DataFrame(comps)
    comp_df.to_csv(outdir / "paired_comparisons.csv", index=False)
    if len(comp_df):
        lines += ["", "Paired differences A - B (95% paired bootstrap CI; * = CI excludes 0), "
                  "McNemar p on referral decisions:", "",
                  "| A vs B | " + " | ".join(PAIRED) + " | McNemar p |", "|" + "---|" * (len(PAIRED) + 2)]
        for r in comps:
            cells = []
            for m in PAIRED:
                lo, hi = r[f"{m}_diff_lo"], r[f"{m}_diff_hi"]
                star = "*" if (lo > 0 or hi < 0) else ""
                cells.append(f"{r[f'{m}_diff']:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}")
            lines.append(f"| {r['A']} vs {r['B']} | " + " | ".join(cells) + f" | {r['mcnemar_p']:.2g} |")
    (outdir / "main_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    c1 = dec[R0]["C1"]
    cnn_auc = auc(c1.true_ref, c1.score)

    # ---- how often does the symbolic layer change anything? ---------------------------
    changes = {}
    for c in ("C4", "C5", "C7"):
        if c in conds:
            d = dec[R0][c]
            changes[c] = {
                "patients_different_patient_grade_vs_C1": int((d.pred_patient != c1.pred_patient).sum()),
                "patients_different_referral_vs_C1": int((d.pred_ref != c1.pred_ref).sum()),
                "eyes_different_grade_vs_C1": int((d.pred_left != c1.pred_left).sum()
                                                  + (d.pred_right != c1.pred_right).sum()),
            }

    # ---- failure analysis C1 vs best combined system ---------------------------------
    params = load_json(RESULTS / f"chosen_params_{R0}.json")
    probs_tag = params.get("probs_tag", R0)
    fr, counts = [], {}
    pr = pd.read_csv(OUTPUTS / f"probs_{probs_tag}_test.csv", dtype={"patient_id": str})
    pcols = [f"p{k}" for k in range(5)]
    pr["probs"] = pr[pcols].round(3).astype(str).agg(" ".join, axis=1)
    pw = pr.pivot(index="patient_id", columns="eye", values="probs")
    for S in ("C5", "C7"):
        if S not in conds:
            continue
        dS = dec[R0][S]
        win = c1.patient_id[~correct["C1"] & correct[S]]
        lose = c1.patient_id[correct["C1"] & ~correct[S]]
        counts[S] = {"right_where_C1_wrong": len(win), "wrong_where_C1_right": len(lose)}
        for tag, ids in ((f"{S} right, C1 wrong", win), (f"C1 right, {S} wrong", lose)):
            pick = ids.sample(min(5, len(ids)), random_state=42) if len(ids) else []
            for pid in pick:
                x1 = c1[c1.patient_id == pid].iloc[0]
                xs = dS[dS.patient_id == pid].iloc[0]
                fr.append({"case": tag, "patient_id": pid, "true_left": x1.level_left,
                           "true_right": x1.level_right, "true_ref": x1.true_ref,
                           "C1_left": x1.pred_left, "C1_right": x1.pred_right, "C1_ref": x1.pred_ref,
                           f"{S}_left": xs.pred_left, f"{S}_right": xs.pred_right, f"{S}_ref": xs.pred_ref,
                           "probs_left": pw.loc[pid, "left"], "probs_right": pw.loc[pid, "right"]})
    pd.DataFrame(fr).to_csv(outdir / "failure_cases.csv", index=False)

    # ---- plots ------------------------------------------------------------------------
    pl = outdir / "plots"
    shown = [c for c in ("C1", "C5", "C7") if c in conds]
    fig, axs = plt.subplots(1, len(shown), figsize=(4.5 * len(shown), 4))
    for ax, c in zip(np.atleast_1d(axs), shown):
        confusion_plot(ax, dec[R0][c], f"{c} {NAMES[c]}")
    fig.tight_layout(); fig.savefig(pl / "confusion_matrices.png", dpi=200); plt.close(fig)

    fpr, tpr = roc_points(c1.true_ref.to_numpy(), c1.score.to_numpy())
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, color="0.4", label=f"CNN score (AUC {cnn_auc:.3f})")
    for c, mk in (("C1", "o"), ("C2", "s"), ("C5", "D"), ("C7", "^")):
        if c in conds:
            r = referral_metrics(dec[R0][c].true_ref, dec[R0][c].pred_ref)
            ax.scatter(1 - r["specificity"], r["recall"], marker=mk, s=50, label=c, zorder=3)
    ax.plot([0, 1], [0, 1], ":", color="0.7")
    ax.set_xlabel("1 - specificity"); ax.set_ylabel("recall (referable DR)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(pl / "referral_roc.png", dpi=200); plt.close(fig)

    g = pd.read_csv(RESULTS / f"val_grid_{R0}.csv") if (RESULTS / f"val_grid_{R0}.csv").exists() else None
    if g is not None:
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
        if "kind" in g:
            gm, gp = g[g.kind == "mismatch"], g[g.kind == "pair"]
        else:
            gm, gp = g, g.iloc[0:0]
        for ax, col, lab in ((axs[0], "val_f1", "val referral F1"), (axs[1], "val_eye_qwk", "val eye kappa")):
            if col not in g:
                continue
            b = gm.groupby("lam")[col].max()
            ax.plot(b.index, b.values, marker="o", label="mismatch weight lam")
            if len(gp):
                b = gp.groupby("mu")[col].max()
                ax.plot(b.index, b.values, marker="^", label="pair weight mu")
            ax.set_xlabel("constraint weight"); ax.set_ylabel(lab + " (best beta)"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(pl / "val_weight_curves.png", dpi=200); plt.close(fig)

    # ---- summary ----------------------------------------------------------------------
    tr = load_json(RESULTS / f"train_log_{probs_tag}.json")
    tr.pop("epochs_log", None)
    summ = {"runs": runs, "test_patients": n, "cnn_referral_auc": cnn_auc,
            "training": tr, "prediction": load_json(RESULTS / f"predict_info_{probs_tag}.json"),
            "chosen_params": {k: v for k, v in params.items() if k != "pair_matrix_W"},
            "symbolic_changes_vs_C1": changes, "referral_wins_losses_vs_C1": counts,
            "solver_ms_per_patient_median": {r["condition"]: r["solver_ms_median"] for r in rows}}
    (outdir / "summary.json").write_text(json.dumps(summ, indent=2, default=float))
    print(f"\nsaved to {outdir}/")


if __name__ == "__main__":
    main()
