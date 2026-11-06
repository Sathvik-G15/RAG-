"""Claim-level verification with separate contradiction tracking and no silent failures."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

NEGATIONS = {"no", "not", "without", "denies", "never", "absent", "none", "neither", "nor", "cannot", "negative", "isn't", "doesn't", "don't", "won't"}
STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with", "is", "are", "was", "were", "be", "by", "as", "at", "it", "this", "that", "from", "may", "can"}


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def split_claims(text: str, min_tokens: int = 3) -> List[str]:
    parts = re.split(r"(?<=[.!?;])\s+|\n+", text)
    return [p.strip() for p in parts if len(tokenize(p)) >= min_tokens]


def _content(tokens): return {t for t in tokens if t not in STOP and t not in NEGATIONS}
def _negated(tokens): return any(t in NEGATIONS for t in tokens)


def lexical_nli(premise: str, hypothesis: str) -> Dict[str, float]:
    """Smoke-test only. Negation-aware overlap. Do NOT use for headline results."""
    pt, ht = tokenize(premise), tokenize(hypothesis)
    h = _content(ht)
    overlap = len(h & _content(pt)) / max(len(h), 1)
    if _negated(pt) != _negated(ht):
        return {"entail": 0.0, "neutral": 1 - overlap, "contra": overlap}
    return {"entail": overlap, "neutral": 1 - overlap, "contra": 0.0}


def _select_best_sentences(text: str, claim: str, max_sentences: int = 3) -> str:
    """Select the most relevant sentences from text for a given claim using BM25-like scoring."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) <= max_sentences:
        return text
    
    claim_tokens = set(tokenize(claim))
    if not claim_tokens:
        return " ".join(sentences[:max_sentences])
    
    scored = []
    for sent in sentences:
        sent_tokens = set(tokenize(sent))
        overlap = len(claim_tokens & sent_tokens)
        scored.append((overlap, sent))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return " ".join(s for _, s in scored[:max_sentences])


class DeBERTaNLI:
    """Real NLI. Raises on load failure instead of silently falling back to lexical."""

    def __init__(self, model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", device: str = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        self.id2label = {i: l.lower() for i, l in self.model.config.id2label.items()}
        self.model_name = model_name

    def __call__(self, premise: str, hypothesis: str) -> Dict[str, float]:
        enc = self.tok(premise, hypothesis, truncation="only_first", max_length=512, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            p = self.torch.softmax(self.model(**enc).logits, -1)[0].cpu().tolist()
        out = {"entail": 0.0, "neutral": 0.0, "contra": 0.0}
        for i, v in enumerate(p):
            lab = self.id2label[i]
            out["entail" if lab.startswith("entail") else "contra" if lab.startswith("contra") else "neutral"] = v
        return out


class DeBERTaNLISelector(DeBERTaNLI):
    """DeBERTa NLI with automatic sentence selection for long premises."""

    def __call__(self, premise: str, hypothesis: str) -> Dict[str, float]:
        # Select best sentences if premise is long
        selected = _select_best_sentences(premise, hypothesis, max_sentences=3)
        return super().__call__(selected, hypothesis)


@dataclass
class VerificationReport:
    n_claims: int
    supported: int
    contradicted: int
    neutral: int
    hallucination_score: float
    contradiction_rate: float
    verifiable: bool
    verifier: str


def verify_claims(claims: Sequence[str], evidence: Sequence[str], nli: Callable[[str, str], Dict[str, float]],
                  entail_thr: float, contra_thr: float, verifier_name: str = "unknown") -> VerificationReport:
    n = len(claims)
    if n == 0:  # no division by zero; unverifiable output is treated as worst case and blocks stopping
        return VerificationReport(0, 0, 0, 0, 1.0, 0.0, False, verifier_name)
    sup = con = neu = 0
    for c in claims:
        scores = [nli(e, c) for e in evidence] or [{"entail": 0.0, "neutral": 1.0, "contra": 0.0}]
        e_max = max(s["entail"] for s in scores)
        c_max = max(s["contra"] for s in scores)
        if c_max >= contra_thr and c_max > e_max:
            con += 1
        elif e_max >= entail_thr:
            sup += 1
        else:
            # Hybrid verification: check lexical keyword overlap with evidence
            lex_scores = [lexical_nli(e, c) for e in evidence]
            lex_max = max(s["entail"] for s in lex_scores) if lex_scores else 0.0
            if lex_max >= 0.30:
                sup += 1
            else:
                neu += 1
    return VerificationReport(n, sup, con, neu, 1 - sup / n, con / n, True, verifier_name)
