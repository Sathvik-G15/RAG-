"""Adaptive Evidence Budgeting (AEB) pipeline — the core research contribution.

Iteratively retrieves evidence in rounds until the calibrated confidence
reaches a sufficiency threshold with patience, or the retrieval budget is exhausted,
or verification gates stop the loop (high contradiction rate / unverifiable).

The loop is:
    1. Retrieve k chunks (starting at initial_k).
    2. Build patient context and reason over the retrieved evidence.
    3. Verify claims (hallucination detection + contradiction tracking).
    4. Compute fused confidence from retriever/evidence/agreement/LLM/verification.
    5. If confidence >= threshold for `patience` consecutive rounds AND verification passes -> stop.
       Else increase budget (k += step_k) and optionally reformulate query.
       Repeat up to max_k.

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
from .stopping import StoppingPolicy, reformulate_query, drop_rate

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
    # New fields from hardening
    stop_reason: str = ""
    conf_drop_rate: float = 0.0


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
        
        # Initialize stopping policy from config (with hardening defaults)
        self.stopping_policy = StoppingPolicy(
            threshold=self.config.confidence_threshold,
            min_steps=getattr(self.config, 'min_steps', 2),
            patience=getattr(self.config, 'patience', 2),
            max_contradiction_rate=getattr(self.config, 'max_contradiction_rate', 0.10),
            max_k=self.config.max_k,
        )
        self.reformulate_query_enabled = getattr(self.config, 'reformulate_query', True)

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
        current_query = query

        round_no = 1
        while True:
            round_evidence = self.retriever.retrieve(
                current_query,
                visit=visit,
                patient=patient,
                patient_context_str=build_context_query(current_query, patient, visit),
                top_k=k,
            )
            for ch in round_evidence:
                # Use hash of text to avoid key collisions from similar prefixes
                import hashlib
                key = hashlib.md5(ch.text.encode()).hexdigest()[:16] if ch.text else str(ch.chunk_id)
                accumulated[key] = ch

            evidence_list = list(accumulated.values())[:max_k]
            response = self.reasoner.generate(current_query, evidence_list)

            # Skip verification for abstention responses (no primary diagnosis)
            # These are appropriate uncertainty expressions, not hallucinations
            if response.primary_diagnosis is None:
                from ..verification.verifier import VerificationResult
                verification = VerificationResult(
                    total_claims=0, supported_claims=0, hallucination_score=0.0, checks=[],
                    contradicted_claims=0, neutral_claims=0, contradiction_rate=0.0, verifiable=True, verifier="skipped"
                )
            else:
                verification = self.verifier.verify(response.reasoning, evidence_list)

            signals = compute_signals(response, evidence_list, verification)
            confidence = self.confidence_model.score(signals)

            confidence_curve.append(confidence)

            # Check stopping policy
            contradiction_rate = getattr(verification, 'contradiction_rate', 0.0)
            verifiable = getattr(verification, 'verifiable', True)
            stop, stop_reason = self.stopping_policy.should_stop(
                confidence_curve, k, contradiction_rate, verifiable
            )

            reason = self._round_reason(round_no, k, confidence, stop_reason)
            step = AEBStep(
                round=round_no,
                k=k,
                evidence=list(round_evidence),
                confidence=confidence,
                signals=signals,
                verification=verification,
                stopped=stop,
                reason=reason,
            )
            steps.append(step)
            final_response = response

            if stop:
                if stop_reason == "threshold":
                    stopped_by_threshold = True
                elif stop_reason == "budget_exhausted":
                    budget_exhausted = True
                break

            k = min(k + cfg.step_k, max_k)
            round_no += 1

            # Query reformulation for next round
            if self.reformulate_query_enabled and final_response and final_response.differential:
                diff = [(d.diagnosis, d.probability) for d in final_response.differential]
                current_query = reformulate_query(query, diff)

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
            stop_reason=stop_reason if 'stop_reason' in locals() else "unknown",
            conf_drop_rate=drop_rate(confidence_curve),
        )

    def _round_reason(self, round_no: int, k: int, confidence: float, stop_reason: str = "") -> str:
        if stop_reason == "threshold":
            return f"Threshold met with patience (confidence {confidence:.3f} >= {self.config.confidence_threshold})"
        if stop_reason == "budget_exhausted":
            return f"Budget exhausted at max_k={k} (confidence {confidence:.3f})"
        if stop_reason == "gated":
            return f"Verification gate blocked stopping (contradiction rate too high or unverifiable)"
        if confidence >= self.config.confidence_threshold:
            return f"Confidence reached threshold ({confidence:.3f} >= {self.config.confidence_threshold}) but waiting for patience"
        return f"Insufficient confidence ({confidence:.3f} < {self.config.confidence_threshold}); requesting more evidence"