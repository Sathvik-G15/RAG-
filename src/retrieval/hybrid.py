"""Hybrid retrieval: dense + sparse merged by reciprocal rank fusion.

Features:
* Reciprocal Rank Fusion (RRF) merges dense and sparse rankings robustly without
  requiring their score scales to match.
* Optional guideline-priority boosting: chunks from high-trust sources
  (e.g., WHO/CDC/NICE/NIH) get an additive boost.
* Optional patient-specific boosting: geriatric / pediatric / renal implications.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..kb.embedder import BaseEmbedder
from ..kb.vector_store import BaseVectorStore
from ..models import EvidenceChunk, PatientProfile, Visit
from .dense import dense_retrieve
from .rerank import BaseReranker
from .sparse import BM25Index


@dataclass
class HybridConfig:
    dense_top_k: int = 50
    sparse_top_k: int = 50
    final_top_k: int = 10
    rrf_k: int = 60
    alpha_trust: float = 0.10
    alpha_patient: float = 0.05


class HybridRetriever:
    def __init__(
        self,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        bm25: BM25Index | None = None,
        reranker: BaseReranker | None = None,
        config: HybridConfig | None = None,
    ):
        self.embedder = embedder
        self.store = store
        self.bm25 = bm25 or BM25Index()
        self.reranker = reranker
        self.config = config or HybridConfig()
        self._indexed_in_bm25 = 0

    def ensure_bm25_indexed(self) -> None:
        """Lazily index the vector store's chunks into BM25 (for cold-start)."""
        if hasattr(self.store, "_chunks") and len(getattr(self.store, "_chunks", [])) > self._indexed_in_bm25:
            chunks = self.store._chunks[self._indexed_in_bm25:]
            self.bm25.add_chunks(chunks)
            self._indexed_in_bm25 = len(self.store._chunks)

    def _patient_query_terms(self, patient: PatientProfile | None) -> set[str]:
        if patient is None:
            return set()

        terms: set[str] = set()
        if patient.age is not None:
            if patient.age >= 65:
                terms.update({"elderly", "geriatric", "older", "65", "pneumococcal", "statin"})
            if patient.age < 16:
                terms.update({"pediatric", "paediatric", "child", "infant"})
        if patient.comorbidities:
            for c in patient.comorbidities:
                terms.update(c.lower().split())
        if patient.pregnancy_status:
            terms.update({"pregnancy", "pregnant", "obstetric", "fetal"})
        return terms

    def _apply_boosting(
        self,
        query: str,
        ranked: list[tuple[EvidenceChunk, float]],
        patient: PatientProfile | None,
    ) -> list[tuple[EvidenceChunk, float]]:
        """Multiplicative boosting by source trust + patient-context token overlap.

        Multiplicative (not additive) is important: RRF base scores are small
        (~0.03), so an additive boost would dwarf relevance and reorder results
        by source trust. Multiplying preserves RRF order while tilting toward
        trusted sources and patient-relevant content.
        """
        patient_terms = self._patient_query_terms(patient)

        boosted: list[tuple[EvidenceChunk, float]] = []
        for ch, base in ranked:
            trust = ch.trust_score
            patient_match = 0.0
            if patient_terms:
                ch_text = (ch.text + " " + (ch.title or "")).lower()
                overlap = sum(1 for t in patient_terms if t in ch_text)
                patient_match = overlap / max(len(patient_terms), 1)

            trust_factor = 1.0 + self.config.alpha_trust * trust
            patient_factor = 1.0 + self.config.alpha_patient * patient_match
            new_score = base * trust_factor * patient_factor
            boosted.append((ch, float(new_score)))
        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

    def retrieve(
        self,
        query: str,
        visit: Visit | None = None,
        patient: PatientProfile | None = None,
        patient_context_str: str | None = None,
        top_k: int | None = None,
    ) -> list[EvidenceChunk]:
        """Retrieve evidence chunks using the hybrid pipeline."""
        effective_query = patient_context_str or query
        top_k = top_k or self.config.final_top_k

        self.ensure_bm25_indexed()

        dense_idx = dense_retrieve(
            effective_query,
            self.embedder,
            self.store,
            top_k=self.config.dense_top_k,
        )
        sparse_idx = self.bm25.search(effective_query, top_k=self.config.sparse_top_k)

        # Carry dense similarity signal onto chunks for downstream confidence.
        dense_sim: dict[str, float] = {str(c.chunk_id): s for c, s in dense_idx}

        fused = self._rrf(dense_idx, sparse_idx)
        boosted = self._apply_boosting(query, fused, patient)

        if self.reranker is not None:
            chunks_only = [c for c, _ in boosted[: max(top_k * 3, top_k)]]
            reranked = self.reranker.rerank(query, chunks_only, top_k=top_k)
            out: list[EvidenceChunk] = []
            for i, (c, _) in enumerate(reranked):
                c.similarity_score = dense_sim.get(str(c.chunk_id), 0.0)
                c.rank = i
                out.append(c)
            return out

        out = []
        for i, (c, _) in enumerate(boosted[:top_k]):
            c.similarity_score = dense_sim.get(str(c.chunk_id), 0.0)
            c.rank = i
            out.append(c)
        return out

    def _rrf(
        self,
        dense: list[tuple[EvidenceChunk, float]],
        sparse: list[tuple[EvidenceChunk, float]],
    ) -> list[tuple[EvidenceChunk, float]]:
        """Reciprocal Rank Fusion over dense and sparse results.

        Combines two document rankings into one fused ranking by
        score(i) = sum_r 1 / (rrf_k + rank_r(i))
        """
        k = self.config.rrf_k
        fused: dict[str, float] = {}
        chunks_map: dict[str, EvidenceChunk] = {}

        def _key(ch: EvidenceChunk) -> str:
            return str(ch.chunk_id)

        for rank, (ch, _) in enumerate(dense):
            key = _key(ch)
            chunks_map[key] = ch
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank + 1)

        for rank, (ch, _) in enumerate(sparse):
            key = _key(ch)
            chunks_map[key] = ch
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank + 1)

        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return [(chunks_map[key], score) for key, score in ordered]
