"""Cross-encoder re-ranking (and token-overlap fallback when no model is loaded)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..models import EvidenceChunk


class BaseReranker:
    def rerank(
        self,
        query: str,
        chunks: Sequence[EvidenceChunk],
        top_k: int = 10,
    ) -> list[tuple[EvidenceChunk, float]]:
        raise NotImplementedError


class OverlapReranker(BaseReranker):
    """Cheap re-ranker based on query-token coverage, minus rare-term bonus.

    Uses the same tokenizer (sparse.tokenize) for query and chunks so token
    sets are directly comparable. Score = token overlap precision + trust tilt.
    """

    def __init__(self, w_trust: float = 0.15, alpha_overlap: float = 1.0):
        self.w_trust = w_trust
        self.alpha_overlap = alpha_overlap

    @staticmethod
    def _query_terms(query: str) -> set[str]:
        from .sparse import tokenize

        return set(tokenize(query))

    @staticmethod
    def _chunk_terms(chunk: EvidenceChunk) -> set[str]:
        from .sparse import tokenize

        return set(tokenize(chunk.text + " " + (chunk.title or "")))

    def rerank(
        self,
        query: str,
        chunks: Sequence[EvidenceChunk],
        top_k: int = 10,
    ) -> list[tuple[EvidenceChunk, float]]:
        q_terms = self._query_terms(query)
        if not q_terms:
            return list(zip(chunks[:top_k], [0.0] * min(top_k, len(chunks))))

        scored: list[tuple[EvidenceChunk, float]] = []
        for ch in chunks:
            ch_terms = self._chunk_terms(ch)
            overlap = len(q_terms & ch_terms) / len(q_terms)
            score = self.alpha_overlap * overlap + self.w_trust * ch.trust_score
            scored.append((ch, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class CrossEncoderReranker(BaseReranker):
    """HuggingFace cross-encoder reranker (lazy-loaded)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self._ce = CrossEncoder(model_name, max_length=512)

    def rerank(
        self,
        query: str,
        chunks: Sequence[EvidenceChunk],
        top_k: int = 10,
    ) -> list[tuple[EvidenceChunk, float]]:
        pairs = [(query, (c.title or "") + " " + c.text[:1500]) for c in chunks]
        scores = self._ce.predict(pairs, show_progress_bar=False).astype(np.float32)
        scored = list(zip(chunks, scores.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def make_reranker(prefer_real: bool = False, model_name: str | None = None) -> BaseReranker:
    if prefer_real and model_name:
        try:
            return CrossEncoderReranker(model_name=model_name)
        except Exception:
            pass
    return OverlapReranker()
