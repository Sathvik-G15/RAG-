"""Chroma DB integration for CAAR-CDSS retrieval pipeline."""

from __future__ import annotations

import uuid

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from ..models import EvidenceChunk


class ChromaRetriever:
    """Chroma-based retriever compatible with HybridRetriever interface."""

    def __init__(
        self,
        collection_name: str = "medical_guidelines",
        chroma_path: str = "data/chroma_db",
        embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        device: str = "cuda",
    ):
        self.embedder = SentenceTransformer(embedding_model, device=device)
        self.client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def search(self, query: str, top_k: int = 10, query_embedding: list | None = None) -> list[EvidenceChunk]:
        """Search Chroma and return EvidenceChunk objects."""
        if query_embedding is None:
            query_embedding = self.embedder.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        chunks = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            for i, (doc_id, doc, metadata, distance) in enumerate(zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                similarity = 1.0 - distance  # Convert cosine distance to similarity
                chunk = EvidenceChunk(
                    chunk_id=uuid.uuid4(),  # Generate UUID for model validation
                    text=doc,
                    source=metadata.get("source", "unknown"),
                    title=metadata.get("title"),
                    year=metadata.get("year"),
                    specialty=metadata.get("specialty"),
                    trust_score=metadata.get("trust_score", 0.8),
                    similarity_score=similarity,
                    rank=i,
                    metadata={**metadata, "original_id": doc_id},
                )
                chunks.append(chunk)

        return chunks

    def __len__(self) -> int:
        return self.collection.count()
