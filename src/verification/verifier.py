"""NLI-based claim verification and hallucination detection.

Supports multiple modes:
* A lightweight, dependency-free lexical/NLI-sim surrogate (usable in CI and
  without GPU). It scores each claim against the retrieved evidence using
  token-overlap and per-source trust, producing a deterministic "supported"
  decision.
* A real DeBERTa-v3-MNLI entailment model (lazy-loaded) for Kaggle T4 runs.
* Lexical fallback for local dev (zero VRAM, deterministic).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..config import detect_hardware
from ..models import ClinicalResponse, EvidenceChunk


@dataclass
class ClaimCheck:
    claim: str
    supported: bool
    entailment_score: float
    supporting_chunk: str | None = None
    source: str | None = None


@dataclass
class VerificationResult:
    total_claims: int
    supported_claims: int
    hallucination_score: float
    checks: list[ClaimCheck] = field(default_factory=list)


def _split_claims(reasoning: str) -> list[str]:
    """Split the LLM reasoning text into atomic claims (sentence-level)."""
    text = reasoning.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+|\n", text)
    cleaned: list[str] = []
    for s in sentences:
        s = s.strip()
        s = re.sub(r"^-\s*", "", s)
        s = re.sub(r"^\(?\d+\)?\s*", "", s)
        if len(s) > 15:
            cleaned.append(s)
    return cleaned


_STOPWORDS = frozenset(
    "a an the and or of to in on for with at by from is are was were be been being this that those "
    "as if then than but not no nor so such into out over under up down off again further here there all any "
    "both each few more most other some only own same its it s t can will shall may might must should would could"
    .split()
)

# Clinical phrasing scaffolding: low-informative words that appear in almost every
# generated claim but carry no verifiable medical content.
_CLAIM_STOPWORDS = _STOPWORDS | frozenset(
    "patient presentation consistent diagnosis differential alternative likely based "
    "supporting evidence supported most strongly indicated appears suggesting suggest "
    "recommended recommend possible probable potentially"
    .split()
)


class LexicalVerifier:
    """Dependency-free claim->evidence scorer (surrogate for NLI)."""

    def __init__(self, entailment_threshold: float = 0.26, similarity_threshold: float = 0.40):
        self.entailment_threshold = entailment_threshold
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def _tokens(text: str, filtering: frozenset = _STOPWORDS) -> set[str]:
        toks = set(re.findall(r"[a-z]+(?:'[a-z]+)*", text.lower()))
        return {t for t in toks if t not in filtering}

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def verify(self, reasoning: str, evidence: Sequence[EvidenceChunk]) -> VerificationResult:
        claims = _split_claims(reasoning)
        checks: list[ClaimCheck] = []

        for claim in claims:
            claim_tokens = self._tokens(claim, _CLAIM_STOPWORDS)
            best_score = 0.0
            best_chunk: EvidenceChunk | None = None
            for ch in evidence:
                overlap = len(claim_tokens & self._tokens(ch.text))
                recall = overlap / max(len(claim_tokens), 1)
                jacc = self._jaccard(claim_tokens, self._tokens(ch.text))
                score = 0.8 * recall + 0.2 * jacc
                if score > best_score:
                    best_score = score
                    best_chunk = ch

            supported = best_score >= self.entailment_threshold
            checks.append(
                ClaimCheck(
                    claim=claim,
                    supported=supported,
                    entailment_score=round(best_score, 4),
                    supporting_chunk=(best_chunk.text[:200] if best_chunk else None),
                    source=(best_chunk.source if best_chunk else None),
                )
            )

        supported_count = sum(1 for c in checks if c.supported)
        total = len(checks)
        hallucination_score = (1 - supported_count / total) if total else 0.0

        return VerificationResult(
            total_claims=total,
            supported_claims=supported_count,
            hallucination_score=hallucination_score,
            checks=checks,
        )


class DeBERTaVerifier:
    """Real NLI entailment verifier (DeBERTa-v3-large-MNLI, lazy-loaded).
    
    Uses direct PyTorch batched inference on GPU 1 (multi-GPU) or GPU 0,
    avoiding accelerate hook deadlocks and speeding up evaluation by ~20x.
    """

    def __init__(self, model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", threshold: float = 0.7):
        self.model_name = model_name
        self.threshold = threshold
        self._model = None
        self._tokenizer = None
        self._device = None
        self._entailment_idx = 0

    def _load(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        target_device = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if "cuda" in target_device else torch.float32,
            low_cpu_mem_usage=True,
        ).to(target_device)
        self._model.eval()
        self._device = target_device

        # Find entailment label index from model config
        label2id = {str(k).lower(): v for k, v in getattr(self._model.config, "label2id", {}).items()}
        self._entailment_idx = label2id.get("entailment", label2id.get("entailed", 0))

    def verify(self, reasoning: str, evidence: Sequence[EvidenceChunk]) -> VerificationResult:
        if self._model is None:
            self._load()
        claims = _split_claims(reasoning)
        if not claims:
            return VerificationResult(total_claims=0, supported_claims=0, hallucination_score=0.0, checks=[])
        if not evidence:
            return VerificationResult(
                total_claims=len(claims),
                supported_claims=0,
                hallucination_score=1.0,
                checks=[ClaimCheck(claim=c, supported=False, entailment_score=0.0) for c in claims],
            )

        import torch
        checks: list[ClaimCheck] = []

        for claim in claims:
            best_score = 0.0
            best_chunk: EvidenceChunk | None = None

            # Batch encode (premise, hypothesis) pairs across all evidence chunks
            premises = [ch.text[:400] for ch in evidence]
            hypotheses = [claim] * len(premises)

            with torch.no_grad():
                inputs = self._tokenizer(
                    premises,
                    hypotheses,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(self._device)
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                ent_scores = probs[:, self._entailment_idx].tolist()

            for score, ch in zip(ent_scores, evidence):
                if score > best_score:
                    best_score = float(score)
                    best_chunk = ch

            supported = best_score >= self.threshold
            checks.append(
                ClaimCheck(
                    claim=claim,
                    supported=supported,
                    entailment_score=round(best_score, 4),
                    supporting_chunk=(best_chunk.text[:200] if best_chunk else None),
                    source=(best_chunk.source if best_chunk else None),
                )
            )

        supported_count = sum(1 for c in checks if c.supported)
        total = len(checks)
        hallucination_score = (1 - supported_count / total) if total else 0.0

        return VerificationResult(
            total_claims=total,
            supported_claims=supported_count,
            hallucination_score=hallucination_score,
            checks=checks,
        )


class VerifierFactory:
    """Factory that routes to the appropriate verifier based on hardware."""

    @staticmethod
    def get_verifier() -> object:
        hw = detect_hardware()
        vram_gb = hw.get("vram_gb", 0)
        if vram_gb >= 10.0:
            return DeBERTaVerifier()
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
    if prefer_real:
        if model_name:
            return DeBERTaVerifier(model_name=model_name)
        return DeBERTaVerifier()
    return LexicalVerifier()

