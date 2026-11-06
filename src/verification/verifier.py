"""NLI-based claim verification and hallucination detection.

Uses the hardened claims module with:
- Negation-aware lexical NLI (for CI/smoke tests only)
- DeBERTa-v3-large-MNLI (for headline results; raises on load failure)
- Explicit contradiction tracking and verifiability reporting
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..config import detect_hardware, get_app_config
from ..models import ClinicalResponse, EvidenceChunk
from .claims import (
    split_claims,
    verify_claims,
    lexical_nli,
    DeBERTaNLI,
    DeBERTaNLISelector,
    VerificationReport,
)


@dataclass
class ClaimCheck:
    claim: str
    supported: bool
    entailment_score: float
    supporting_chunk: str | None = None
    source: str | None = None


@dataclass
class VerificationResult:
    """Backward-compatible result format."""
    total_claims: int
    supported_claims: int
    hallucination_score: float
    checks: list[ClaimCheck] = field(default_factory=list)
    # New fields from hardening
    contradicted_claims: int = 0
    neutral_claims: int = 0
    contradiction_rate: float = 0.0
    verifiable: bool = True
    verifier: str = "unknown"


def _report_to_result(report: VerificationReport, evidence: Sequence[EvidenceChunk] | None = None) -> VerificationResult:
    """Convert new VerificationReport to backward-compatible VerificationResult."""
    checks = []
    # We don't have per-claim detail in the new report, so create minimal checks
    # In a full implementation, verify_claims would return per-claim details
    return VerificationResult(
        total_claims=report.n_claims,
        supported_claims=report.supported,
        hallucination_score=report.hallucination_score,
        checks=checks,
        contradicted_claims=report.contradicted,
        neutral_claims=report.neutral,
        contradiction_rate=report.contradiction_rate,
        verifiable=report.verifiable,
        verifier=report.verifier,
    )


class LexicalVerifier:
    """Dependency-free claim->evidence scorer (surrogate for NLI). Uses hardened lexical_nli."""

    def __init__(self, entailment_threshold: float = 0.26, contra_threshold: float = 0.26):
        self.entailment_threshold = entailment_threshold
        self.contra_threshold = contra_threshold

    def verify(self, reasoning: str, evidence: Sequence[EvidenceChunk]) -> VerificationResult:
        claims = split_claims(reasoning)
        evidence_texts = [f"{c.title}: {c.text}" if getattr(c, "title", None) else c.text for c in evidence] if evidence else [""]
        report = verify_claims(
            claims,
            evidence_texts,
            lexical_nli,
            self.entailment_threshold,
            self.contra_threshold,
            verifier_name="lexical"
        )
        return _report_to_result(report, evidence)


class DeBERTaVerifier:
    """Real NLI entailment verifier (DeBERTa-v3-large-MNLI). Raises on load failure."""

    def __init__(
        self,
        model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        entailment_threshold: float = 0.5,
        contra_threshold: float = 0.6,
        device: str | None = None,
        use_sentence_selection: bool = True,
    ):
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self.contra_threshold = contra_threshold
        self.device = device or ("cuda" if detect_hardware().get("device") == "cuda" else "cpu")
        self._nli = None
        self.use_sentence_selection = use_sentence_selection

    def _load(self):
        """Load the DeBERTa NLI model. Raises on failure - no silent fallback."""
        if self.use_sentence_selection:
            self._nli = DeBERTaNLISelector(model_name=self.model_name, device=self.device)
        else:
            self._nli = DeBERTaNLI(model_name=self.model_name, device=self.device)

    def verify(self, reasoning: str, evidence: Sequence[EvidenceChunk]) -> VerificationResult:
        if self._nli is None:
            self._load()
        claims = split_claims(reasoning)
        evidence_texts = [f"{c.title}: {c.text}" if getattr(c, "title", None) else c.text for c in evidence] if evidence else [""]
        report = verify_claims(
            claims,
            evidence_texts,
            self._nli,
            self.entailment_threshold,
            self.contra_threshold,
            verifier_name=f"deberta:{self.model_name}"
        )
        return _report_to_result(report, evidence)


class VerifierFactory:
    """Factory that routes to the appropriate verifier based on config (not hardware)."""

    @staticmethod
    def get_verifier() -> object:
        cfg = get_app_config()
        strict_mode = getattr(cfg.security, 'strict_mode', False) if hasattr(cfg, 'security') else False
        backend = "deberta"  # Default to deberta for experiments
        
        if hasattr(cfg, 'verification'):
            if isinstance(cfg.verification, dict):
                backend = cfg.verification.get("backend", "deberta")
            else:
                backend = getattr(cfg.verification, 'backend', "deberta")
        
        if backend == "deberta":
            return DeBERTaVerifier()
        
        if strict_mode:
            raise RuntimeError("Strict mode: lexical verifier not allowed. Set verification.backend=deberta")
        return LexicalVerifier()


def verify_response(
    response: ClinicalResponse,
    evidence: Sequence[EvidenceChunk],
    verifier: object | None = None,
) -> VerificationResult:
    if verifier is None:
        verifier = VerifierFactory.get_verifier()
    return verifier.verify(response.reasoning, evidence)


def make_verifier(prefer_real: bool = False, model_name: str | None = None) -> object:
    """Factory that creates DeBERTaVerifier when prefer_real=True, or LexicalVerifier."""
    cfg = get_app_config()
    strict_mode = getattr(cfg.security, 'strict_mode', False) if hasattr(cfg, 'security') else False
    allow_fallback = not strict_mode
    
    if hasattr(cfg, 'verification'):
        if isinstance(cfg.verification, dict):
            allow_fallback = cfg.verification.get("allow_fallback", allow_fallback)
        else:
            allow_fallback = getattr(cfg.verification, 'allow_fallback', allow_fallback)
    
    if prefer_real:
        if model_name:
            return DeBERTaVerifier(model_name=model_name)
        try:
            return DeBERTaVerifier()
        except Exception:
            if allow_fallback:
                return LexicalVerifier()
            raise
    
    # For mock/non-real mode, always use LexicalVerifier (fast, no model loading)
    return LexicalVerifier()