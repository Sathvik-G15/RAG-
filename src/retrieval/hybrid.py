"""Hybrid retrieval: dense + sparse merged by reciprocal rank fusion.

Features:
* Reciprocal Rank Fusion (RRF) merges dense and sparse rankings robustly without
  requiring their score scales to match.
* Bounded, normalized boosting: RRF scores normalized to [0,1], then multiplicative
  trust and patient boosts with bounded factors.
* Optional cross-encoder re-ranking.
* Rank displacement measurement for ablation studies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..kb.embedder import BaseEmbedder
from ..kb.vector_store import BaseVectorStore
from ..models import EvidenceChunk, PatientProfile, Visit
from .dense import dense_retrieve
from .rerank import BaseReranker
from .sparse import BM25Index
from .boosting import boost_and_rerank, rank_displacement


@dataclass
class HybridConfig:
    dense_top_k: int = 50
    sparse_top_k: int = 50
    final_top_k: int = 10
    rrf_k: int = 60
    # Boosting parameters (from hardening.yaml)
    boost_alpha: float = 0.2   # trust boost strength
    boost_beta: float = 0.2    # patient match boost strength


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
        self._last_rank_displacement: float = 0.0

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

    def _compute_patient_match(self, chunk: EvidenceChunk, patient_terms: set[str]) -> float:
        """Compute patient match score (0-1) for a chunk."""
        if not patient_terms:
            return 0.0
        ch_text = (chunk.text + " " + (chunk.title or "")).lower()
        overlap = sum(1 for t in patient_terms if t in ch_text)
        return overlap / max(len(patient_terms), 1)

    def retrieve(
        self,
        query: str,
        visit: Visit | None = None,
        patient: PatientProfile | None = None,
        patient_context_str: str | None = None,
        top_k: int | None = None,
    ) -> list[EvidenceChunk]:
        """Retrieve evidence chunks using the hybrid pipeline with bounded boosting.
        
        Retrieval Order (important for ablation):
        1. Dense retrieval (biomedical embeddings) -> top dense_top_k
        2. Sparse retrieval (BM25) -> top sparse_top_k  
        3. Reciprocal Rank Fusion (RRF) merges dense + sparse
        4. Bounded boosting: RRF scores normalized to [0,1], then multiplicative
           trust_score (alpha) and patient_match (beta) boosts
        5. Cross-encoder re-ranking (if reranker provided) on boosted top candidates
           NOTE: Reranker runs AFTER boosting, so it can override boost ordering.
               For ablation: compare with/without reranker and with/without boosting.
        """
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

        # RRF fusion
        fused = self._rrf(dense_idx, sparse_idx)

        # Prepare chunks for bounded boosting
        patient_terms = self._patient_query_terms(patient)
        boost_input = []
        for ch, rrf_score in fused:
            boost_input.append({
                "chunk": ch,
                "rrf_score": rrf_score,
                "trust_score": ch.trust_score,
                "patient_match": self._compute_patient_match(ch, patient_terms),
            })

        # Apply bounded, normalized boosting
        boosted = boost_and_rerank(
            boost_input,
            alpha=self.config.boost_alpha,
            beta=self.config.boost_beta,
        )

        # Track rank displacement for ablation
        before_ids = [str(c["chunk"].chunk_id) for c in boost_input]
        after_ids = [str(c["chunk"].chunk_id) for c in boosted]
        self._last_rank_displacement = rank_displacement(before_ids, after_ids)

        # Convert back to (chunk, score) format
        boosted_pairs = [(c["chunk"], c["final_score"]) for c in boosted]

        if self.reranker is not None:
            chunks_only = [c for c, _ in boosted_pairs[: max(top_k * 3, top_k)]]
            reranked = self.reranker.rerank(query, chunks_only, top_k=top_k)
            out: list[EvidenceChunk] = []
            for i, (c, _) in enumerate(reranked):
                c.similarity_score = dense_sim.get(str(c.chunk_id), 0.0)
                c.rank = i
                out.append(c)
            return out

        out = []
        for i, (c, _) in enumerate(boosted_pairs[:top_k]):
            c.similarity_score = dense_sim.get(str(c.chunk_id), 0.0)
            c.rank = i
            out.append(c)
        return out

    def get_last_rank_displacement(self) -> float:
        """Return the rank displacement from the last retrieval (for ablation logging)."""
        return self._last_rank_displacement

    def _rrf(
        self,
        dense: list[tuple[EvidenceChunk, float]],
        sparse: list[tuple[EvidenceChunk, float]],
    ) -> list[tuple[EvidenceChunk, float]]:
        """Reciprocal Rank Fusion over dense and sparse results."""
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