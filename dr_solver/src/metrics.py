"""Fast numpy metrics (used inside the bootstrap, so no sklearn per call)."""
from __future__ import annotations

import numpy as np

K = 5


def qwk(y_true, y_pred, k: int = K) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    O = np.zeros((k, k))
    np.add.at(O, (y_true, y_pred), 1)
    n = O.sum()
    if n == 0:
        return float("nan")
    i, j = np.indices((k, k))
    W = (i - j) ** 2 / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / n
    denom = (W * E).sum()
    return float(1 - (W * O).sum() / denom) if denom > 0 else float("nan")


def referral_metrics(y_ref, p_ref) -> dict:
    y = np.asarray(y_ref, dtype=bool)
    p = np.asarray(p_ref, dtype=bool)
    tp = int((y & p).sum()); fp = int((~y & p).sum())
    fn = int((y & ~p).sum()); tn = int((~y & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp and (prec + rec) else 0.0
    spec = tn / (tn + fp) if tn + fp else float("nan")
    fnr = fn / (tp + fn) if tp + fn else float("nan")
    return {"precision": prec, "recall": rec, "f1": f1, "specificity": spec,
            "fnr": fnr, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def auc(y, score) -> float:
    """ROC AUC via rank statistic (ties handled by average rank)."""
    from scipy.stats import rankdata

    y = np.asarray(y, dtype=bool)
    s = np.asarray(score, dtype=float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
