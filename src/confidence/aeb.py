"""Adaptive Evidence Budgeting (AEB) pipeline — the core research contribution.

Iteratively retrieves evidence in rounds until the calibrated confidence
reaches a sufficiency threshold, or the retrieval budget is exhausted.

The loop is:
    1. Retrieve k chunks (starting at initial_k).
    2. Build patient context and reason over the retrieved evidence.
    3. Verify claims (hallucination detection).
    4. Compute fused confidence from retriever/evidence/agreement/LLM/verification.
    5. If confidence >= threshold -> stop. Else increase budget (k += step_k)
       and repeat up to max_k.

This answers the clinically important question: "When does the AI have
enough evidence to make a safe recommendation?"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..agents.context_builder import build_context_query
from ..agents.reasoner import BaseReasoner
from ..config import AEBConfig
from ..models import (
    ClinicalResponse,
    EvidenceChunk,
    PatientProfile,
    TriageDecision,
    Visit,
)
from ..retrieval.hybrid import HybridRetriever
from ..verification.verifier import LexicalVerifier, VerificationResult
from .estimation import ConfidenceModel, ConfidenceSignals, compute_signals

logger = logging.getLogger(__name__)


@dataclass
class AEBStep:
    round: int
    k: int
    evidence: list[EvidenceChunk] = field(default_factory=list)
    confidence: float = 0.0
    signals: ConfidenceSignals | None = None
    verification: VerificationResult | None = None
    stopped: bool = False
    reason: str = ""


@dataclass
class AEBResult:
    response: ClinicalResponse
    steps: list[AEBStep] = field(default_factory=list)
    total_retrieved: int = 0
    confidence_curve: list[float] = field(default_factory=list)
    stopped_by_threshold: bool = False
    budget_exhausted: bool = False


class AEBPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        reasoner: BaseReasoner,
        confidence_model: ConfidenceModel | None = None,
        verifier: object | None = None,
        config: AEBConfig | None = None,
    ):
        self.retriever = retriever
        self.reasoner = reasoner
        self.confidence_model = confidence_model or ConfidenceModel()
        self.verifier = verifier or LexicalVerifier()
        self.config = config or AEBConfig()

    def _round_reason(self, round_no: int, k: int, confidence: float) -> str:
        if confidence >= self.config.confidence_threshold:
            return f"Sufficient evidence reached (confidence {confidence:.3f} >= {self.config.confidence_threshold})"
        if k >= self.config.max_k:
            return f"Budget exhausted at max_k={k} (confidence {confidence:.3f})"
        return f"Insufficient confidence ({confidence:.3f} < {self.config.confidence_threshold}); requesting more evidence"

    def run(
        self,
        query: str,
        visit: Visit | None = None,
        patient: PatientProfile | None = None,
        max_k_override: int | None = None,
    ) -> AEBResult:
        cfg = self.config
        max_k = max_k_override or cfg.max_k
        threshold = cfg.confidence_threshold

        k = cfg.initial_k
        steps: list[AEBStep] = []
        confidence_curve: list[float] = []
        accumulated: dict[str, EvidenceChunk] = {}
        stopped_by_threshold = False
        budget_exhausted = False
        final_response: ClinicalResponse | None = None

        round_no = 1
        while True:
            round_evidence = self.retriever.retrieve(
                query,
                visit=visit,
                patient=patient,
                patient_context_str=build_context_query(query, patient, visit),
                top_k=k,
            )
            for ch in round_evidence:
                accumulated[str(ch.chunk_id)] = ch

            evidence_list = list(accumulated.values())
            response = self.reasoner.generate(query, evidence_list)
            verification = self.verifier.verify(response.reasoning, evidence_list)
            signals = compute_signals(response, evidence_list, verification)
            confidence = self.confidence_model.score(signals)

            confidence_curve.append(confidence)

            reason = self._round_reason(round_no, k, confidence)
            step = AEBStep(
                round=round_no,
                k=k,
                evidence=list(round_evidence),
                confidence=confidence,
                signals=signals,
                verification=verification,
                stopped=confidence >= threshold or k >= max_k,
                reason=reason,
            )
            steps.append(step)
            final_response = response

            if confidence >= threshold:
                stopped_by_threshold = True
                break
            if k >= max_k:
                budget_exhausted = True
                break

            k = min(k + cfg.step_k, max_k)
            round_no += 1

        if final_response is None:
            final_response = ClinicalResponse()

        # Apply post-loop decision overrides (safety first).
        final_response.retrieval_k_used = len(accumulated)
        final_response.evidence = list(accumulated.values())
        final_response.confidence = confidence_curve[-1] if confidence_curve else 0.0
        final_response.hallucination_score = (
            steps[-1].verification.hallucination_score if steps and steps[-1].verification else 0.0
        )
        if steps and steps[-1].verification:
            final_response.hallucination_score = steps[-1].verification.hallucination_score

        if confidence_curve and confidence_curve[-1] < threshold:
            if not final_response.escalated_reason:
                final_response.escalated_reason = (
                    "Insufficient confidence to make a safe recommendation. Clinician review advised."
                )
            final_response.decision = TriageDecision.ESCALATE

        return AEBResult(
            response=final_response,
            steps=steps,
            total_retrieved=len(accumulated),
            confidence_curve=confidence_curve,
            stopped_by_threshold=stopped_by_threshold,
            budget_exhausted=budget_exhausted,
        )
