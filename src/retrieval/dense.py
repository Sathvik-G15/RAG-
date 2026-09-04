"""Dense passage retrieval against the vector store."""

from __future__ import annotations

from ..kb.embedder import BaseEmbedder
from ..kb.vector_store import BaseVectorStore
from ..models import EvidenceChunk


def dense_retrieve(
    query: str,
    embedder: BaseEmbedder,
    store: BaseVectorStore,
    top_k: int = 10,
) -> list[tuple[EvidenceChunk, float]]:
    """Return (chunk, similarity) pairs for the query."""
    q_emb = embedder.embed([query])[0]
    # Check if store is ChromaRetriever which accepts query_embedding parameter
    if store.__class__.__name__ == 'ChromaRetriever':
        chunks = store.search(query, top_k=top_k, query_embedding=q_emb.tolist())
        return [(c, c.similarity_score) for c in chunks]
    results = store.search(q_emb, top_k=top_k)
    if results and isinstance(results[0], EvidenceChunk):
        return [(c, c.similarity_score) for c in results]
    return results
