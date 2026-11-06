#!/usr/bin/env python
"""Fit/compare confidence fusion on per-item records.

Input JSONL, one record per question, produced by your experiment runner at the AEB STOP step:
  {"id": "q1", "signals": {"retriever_top1":0.4,"evidence_sufficiency":0.7,"support_agreement":0.5,
                           "raw_confidence":0.8,"hallucination_inverse":0.9},
   "correct": 1}
Usage: python scripts/run_calibration_study.py --records runs/records.jsonl --out reports/calibration.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.confidence.calibration import (LearnedFusion, auroc, bootstrap_ci, brier, ece, fuse_fixed,
                                        reliability_table, split_indices)

HAND = {"retriever_top1": .15, "evidence_sufficiency": .30, "support_agreement": .25,
        "raw_confidence": .15, "hallucination_inverse": .15}


def metrics(name, conf, y):
    lo, hi = bootstrap_ci(ece, conf, y)
    return {"name": name, "ece": ece(conf, y), "ece_ci95": [lo, hi], "brier": brier(conf, y),
            "auroc": auroc(conf, y), "reliability": reliability_table(conf, y)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", default="reports/calibration.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-model", default="models/fusion.joblib")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.records) if l.strip()]
    names = list(recs[0]["signals"].keys())
    X = np.array([[r["signals"][n] for n in names] for r in recs], float)
    y = np.array([r["correct"] for r in recs], int)
    fit, cal, test = split_indices(len(y), seed=a.seed)

    w = [HAND.get(n, 1.0) for n in names]
    lf = LearnedFusion(names).fit(X[fit], y[fit], X[cal], y[cal])
    lf_noiso = LearnedFusion(names).fit(X[fit], y[fit])

    report = {
        "n": len(y), "split_sizes": [len(fit), len(cal), len(test)], "seed": a.seed,
        "learned_logit_weights": lf.weights(),
        "signal_correlations": np.corrcoef(X.T).round(3).tolist(), "signal_names": names,
        "per_signal_auroc_test": {n: auroc(X[test][:, i], y[test]) for i, n in enumerate(names)},
        "test": [metrics("hand_weights", fuse_fixed(X[test], w), y[test]),
                 metrics("learned_logistic", lf_noiso.predict(X[test]), y[test]),
                 metrics("learned_logistic+isotonic", lf.predict(X[test]), y[test])],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    import joblib
    Path(a.save_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(lf, a.save_model)  # load in confidence/estimation.py: joblib.load(path).predict(X)
    for m in report["test"]:
        print(f"{m['name']:28s} ECE={m['ece']:.3f} CI={m['ece_ci95'][0]:.3f}-{m['ece_ci95'][1]:.3f} "
              f"Brier={m['brier']:.3f} AUROC={m['auroc']:.3f}")


if __name__ == "__main__":
    main()
