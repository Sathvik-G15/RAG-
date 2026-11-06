"""Confidence estimation: fuse retriever, evidence, agreement, and calibration.

Supports both:
- Fixed-weight fusion (legacy, for ablations)
- Learned fusion (LogisticRegression + IsotonicRegression) from calibration study
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Optional

import numpy as np

from ..models import ClinicalResponse, EvidenceChunk
from ..verification.verifier import VerificationResult


@dataclass
class ConfidenceSignals:
    retriever_top1: float = 0.0
    evidence_sufficiency: float = 0.0
    support_agreement: float = 0.0
    hallucination_penalty: float = 0.0
    raw_confidence: float = 0.0


def _softmax(values: Sequence[float]) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    a = a - a.max()
    e = np.exp(a)
    return e / e.sum()


def _retriever_sufficiency(chunks: Sequence[EvidenceChunk]) -> float:
    """How clearly the retriever prefers the top result (flatness penalty)."""
    if not chunks:
        return 0.0
    top1 = chunks[0].similarity_score
    top5 = chunks[min(4, len(chunks) - 1)].similarity_score
    delta = max(0.0, top1 - top5)
    return float(np.clip(0.5 + delta * 2.0, 0.0, 1.0))


def _agreement(differential_probs: Sequence[float]) -> float:
    """Top-2 margin normalized; higher margin = higher agreement."""
    if not differential_probs:
        return 0.0
    probs = sorted(differential_probs, reverse=True)
    if len(probs) == 1:
        return 1.0
    margin = probs[0] - probs[1]
    return float(np.clip(0.5 + margin * 2.0, 0.0, 1.0))


def compute_signals(
    response: ClinicalResponse,
    evidence: Sequence[EvidenceChunk],
    verification: VerificationResult | None = None,
) -> ConfidenceSignals:
    probs = [d.probability for d in response.differential]
    retriever = _retriever_sufficiency(evidence)
    agreement = _agreement(probs)

    # 1. Check explicit supporting_evidence count on differential
    support_count = sum(1 for d in response.differential if d.supporting_evidence)
    
    # 2. If open-source LLM did not populate supporting_evidence list,
    # evaluate whether the top differential diagnoses are grounded in the retrieved evidence
    if support_count == 0 and evidence and response.differential:
        ev_text = " ".join((c.text + " " + (c.title or "")).lower() for c in evidence)
        top_dx = response.differential[0].diagnosis.lower()
        # Extract meaningful terms from diagnosis name
        dx_tokens = [t for t in re.findall(r"[a-z0-9]+", top_dx) if len(t) >= 4]
        if dx_tokens and any(tok in ev_text for tok in dx_tokens):
            # Primary diagnosis directly matches retrieved clinical guideline!
            evidence_sufficiency = 0.90
        elif evidence:
            top_sim = max((getattr(c, "similarity_score", 0.0) or 0.0) for c in evidence)
            evidence_sufficiency = min(1.0, max(0.5, float(top_sim)))
        else:
            evidence_sufficiency = 0.0
    else:
        evidence_sufficiency = min(1.0, support_count / max(len(response.differential), 1))

    hallucination_penalty = verification.hallucination_score if verification else 0.0
    raw_conf = max(response.confidence, probs[0] if probs else 0.0)

    return ConfidenceSignals(
        retriever_top1=retriever,
        evidence_sufficiency=evidence_sufficiency,
        support_agreement=agreement,
        hallucination_penalty=hallucination_penalty,
        raw_confidence=raw_conf,
    )


@dataclass
class ConfidenceModel:
    """Weighted fusion of signals; optionally temperature-scaled (legacy path)."""

    w_retriever: float = 0.20
    w_evidence: float = 0.35
    w_agreement: float = 0.30
    w_llm: float = 0.10
    w_hallucination: float = 0.05
    temperature: float = 1.0
    # Learned fusion (new path)
    _learned_model: Any = None
    _signal_names: list[str] = None

    def calibrate(self, temperature: float) -> None:
        self.temperature = max(0.1, temperature)

    def load_learned_fusion(self, model_path: str | Path, signal_names: list[str] | None = None) -> None:
        """Load a trained fusion model (joblib dump of LearnedFusion)."""
        import joblib
        self._learned_model = joblib.load(model_path)
        self._signal_names = signal_names or [
            "retriever_top1",
            "evidence_sufficiency",
            "support_agreement",
            "raw_confidence",
            "hallucination_inverse",  # 1 - hallucination_penalty (uncapped)
        ]

    def score(self, signals: ConfidenceSignals) -> float:
        # If learned model is available, use it (new path)
        if self._learned_model is not None:
            # Use uncapped hallucination_inverse = 1 - hallucination_penalty
            hallucination_inverse = 1.0 - signals.hallucination_penalty
            X = np.array([[
                signals.retriever_top1,
                signals.evidence_sufficiency,
                signals.support_agreement,
                signals.raw_confidence,
                hallucination_inverse,
            ]], dtype=np.float64)
            return float(np.clip(self._learned_model.predict(X)[0], 0.0, 1.0))

        # Legacy fixed-weight path (for ablations)
        # Cap hallucination penalty at 0.35 so that a false-positive hallucination
        # signal from conversational sentences cannot drive confidence to near-zero.
        capped_hallucination = min(signals.hallucination_penalty, 0.35)
        weights = np.array(
            [self.w_retriever, self.w_evidence, self.w_agreement, self.w_llm, self.w_hallucination],
            dtype=np.float64,
        )
        values = np.array(
            [
                signals.retriever_top1,
                signals.evidence_sufficiency,
                signals.support_agreement,
                signals.raw_confidence,
                1.0 - capped_hallucination,
            ],
            dtype=np.float64,
        )
        raw = float((weights * values).sum() / weights.sum())
        if self.temperature != 1.0:
            z = raw / self.temperature
            raw = float(_softmax([z, 0.0])[0])
        return float(np.clip(raw, 0.0, 1.0))


class TemperatureScaler:
    """Fit temperature parameter on validation set (platt/temperature scaling)."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: Sequence[float], labels: Sequence[int]) -> float:
        from scipy.optimize import minimize

        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)

        def nll(T: float) -> float:
            T = max(1e-4, T)
            probs = 1.0 / (1.0 + np.exp(-logits / T))
            eps = 1e-12
            return -float(np.mean(labels * np.log(probs + eps) + (1 - labels) * np.log(1 - probs + eps)))

        res = minimize(nll, x0=[1.0], bounds=[(1e-2, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        return self.temperature