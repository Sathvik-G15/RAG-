"""Confidence estimation: fuse retriever, evidence, agreement, and calibration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
    support_count = sum(1 for d in response.differential if d.supporting_evidence)
    evidence_sufficiency = min(1.0, support_count / max(len(response.differential), 1))
    hallucination_penalty = verification.hallucination_score if verification else 0.0

    return ConfidenceSignals(
        retriever_top1=_retriever_sufficiency(evidence),
        evidence_sufficiency=evidence_sufficiency,
        support_agreement=agreement,
        hallucination_penalty=hallucination_penalty,
        raw_confidence=response.confidence,
    )


@dataclass
class ConfidenceModel:
    """Weighted fusion of signals; optionally temperature-scaled."""

    w_retriever: float = 0.15
    w_evidence: float = 0.30
    w_agreement: float = 0.25
    w_llm: float = 0.15
    w_hallucination: float = 0.15
    temperature: float = 1.0

    def calibrate(self, temperature: float) -> None:
        self.temperature = max(0.1, temperature)

    def score(self, signals: ConfidenceSignals) -> float:
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
                1.0 - signals.hallucination_penalty,
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
