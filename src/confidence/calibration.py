"""Calibration utilities: metrics, bootstrap CIs, learned fusion, isotonic calibration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def ece(conf, correct, n_bins: int = 10) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def brier(conf, correct) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    return float(np.mean((conf - correct) ** 2))


def auroc(conf, correct) -> float:
    correct = np.asarray(correct)
    if len(np.unique(correct)) < 2:
        return float("nan")
    return float(roc_auc_score(correct, conf))


def reliability_table(conf, correct, n_bins: int = 10) -> List[Dict]:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int(m.sum()),
                     "mean_conf": float(conf[m].mean()) if m.any() else None,
                     "accuracy": float(correct[m].mean()) if m.any() else None})
    return rows


def bootstrap_ci(metric: Callable, conf, correct, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    n = len(conf)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        vals.append(metric(conf[i], correct[i]))
    vals = np.array(vals, float)
    return float(np.nanpercentile(vals, 100 * alpha / 2)), float(np.nanpercentile(vals, 100 * (1 - alpha / 2)))


def split_indices(n: int, fractions=(0.4, 0.2, 0.4), seed: int = 0):
    """fit / calibration / test. Never tune and evaluate on the same split."""
    assert abs(sum(fractions) - 1) < 1e-9
    perm = np.random.default_rng(seed).permutation(n)
    a, b = int(n * fractions[0]), int(n * (fractions[0] + fractions[1]))
    return perm[:a], perm[a:b], perm[b:]


def fuse_fixed(X: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    w = np.asarray(weights, float)
    return np.clip(X @ (w / w.sum()), 0, 1)


@dataclass
class LearnedFusion:
    names: List[str]
    model: LogisticRegression = None
    iso: IsotonicRegression = None

    def fit(self, X_fit, y_fit, X_cal=None, y_cal=None, C: float = 1.0):
        self.model = LogisticRegression(C=C, max_iter=1000).fit(X_fit, y_fit)
        if X_cal is not None:
            raw = self.model.predict_proba(X_cal)[:, 1]
            self.iso = IsotonicRegression(out_of_bounds="clip").fit(raw, y_cal)
        return self

    def predict(self, X) -> np.ndarray:
        p = self.model.predict_proba(X)[:, 1]
        return self.iso.predict(p) if self.iso is not None else p

    def weights(self) -> Dict[str, float]:
        return dict(zip(self.names, map(float, self.model.coef_[0])))
