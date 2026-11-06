"""Statistics for the paper: paired bootstrap, McNemar, risk-coverage, triage safety."""
from __future__ import annotations

from math import comb
from typing import Dict, Sequence

import numpy as np


def paired_bootstrap(correct_a: Sequence[int], correct_b: Sequence[int], n_boot: int = 10000, seed: int = 0) -> Dict:
    a, b = np.asarray(correct_a, float), np.asarray(correct_b, float)
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    obs = float(a.mean() - b.mean())
    p = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {"diff": obs, "ci95": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))), "p": min(p, 1.0)}


def mcnemar_exact(correct_a: Sequence[int], correct_b: Sequence[int]) -> Dict:
    a, b = np.asarray(correct_a, int), np.asarray(correct_b, int)
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"n01": n01, "n10": n10, "p": float(p)}


def risk_coverage(conf: Sequence[float], correct: Sequence[int]) -> Dict:
    """Sort by confidence desc; risk = error rate among the answered top-c fraction."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    order = np.argsort(-conf, kind="stable")
    err = 1 - correct[order]
    n = len(conf)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(err) / np.arange(1, n + 1)
    return {"coverage": coverage, "risk": risk, "aurc": float(risk.mean())}


_RANK = {"HOME_CARE": 0, "ROUTINE": 1, "URGENT": 2, "EMERGENCY": 3}


def triage_safety(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict:
    t = np.array([_RANK[x] for x in y_true])
    p = np.array([_RANK[x] for x in y_pred])
    emerg = t == 3
    pred_emerg = p == 3
    return {
        "under_triage_rate": float((p < t).mean()),
        "over_triage_rate": float((p > t).mean()),
        "severe_under_triage_rate": float((t - p >= 2).mean()),
        "emergency_sensitivity": float(pred_emerg[emerg].mean()) if emerg.any() else float("nan"),
        "emergency_specificity": float((~pred_emerg[~emerg]).mean()) if (~emerg).any() else float("nan"),
        "n": int(len(t)),
    }
