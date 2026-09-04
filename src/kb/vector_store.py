"""Vector store with pluggable backends (in-memory, FAISS, Qdrant)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..models import EvidenceChunk


class BaseVectorStore:
    def add(self, chunks: list[EvidenceChunk], embeddings: np.ndarray) -> None:
        raise NotImplementedError

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[EvidenceChunk, float]]:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class InMemoryVectorStore(BaseVectorStore):
    """Cosine-similarity search over normalized embeddings. CPU only, fully reproducible."""

    def __init__(self):
        self._chunks: list[EvidenceChunk] = []
        self._embeddings: np.ndarray | None = None

    def add(self, chunks: list[EvidenceChunk], embeddings: np.ndarray) -> None:
        embs = embeddings.astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embs = embs / norms

        if self._embeddings is None:
            self._embeddings = embs
        else:
            self._embeddings = np.vstack([self._embeddings, embs])
        self._chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[EvidenceChunk, float]]:
        if self._embeddings is None or len(self._chunks) == 0:
            return []
        q = query_embedding.astype(np.float32).reshape(1, -1)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        sims = (self._embeddings @ q.T).flatten()
        top_k = min(top_k, len(sims))
        idx = np.argsort(-sims)[:top_k]
        return [(self._chunks[i], float(sims[i])) for i in idx]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        if self._embeddings is not None:
            np.save(path / "embeddings.npy", self._embeddings)
        meta = {"n": len(self._chunks), "dim": int(self._embeddings.shape[1]) if self._embeddings is not None else 0}
        (path / "meta.json").write_text(json.dumps(meta))

    def load(self, path: Path) -> None:
        path = Path(path)
        with open(path / "chunks.pkl", "rb") as f:
            self._chunks = pickle.load(f)
        self._embeddings = np.load(path / "embeddings.npy")

    def __len__(self) -> int:
        return len(self._chunks)


class FAISSVectorStore(BaseVectorStore):
    """FAISS-backed store with cosine similarity (via inner product on normalized vectors)."""

    def __init__(self):
        import faiss

        self._faiss = faiss
        self._index: object | None = None
        self._chunks: list[EvidenceChunk] = []
        self._dim: int = 0

    def add(self, chunks: list[EvidenceChunk], embeddings: np.ndarray) -> None:
        embs = embeddings.astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embs = embs / norms
        self._dim = embs.shape[1]
        if self._index is None:
            self._index = self._faiss.IndexFlatIP(self._dim)
        self._index.add(embs)
        self._chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[EvidenceChunk, float]]:
        if self._index is None or len(self._chunks) == 0:
            return []
        q = query_embedding.astype(np.float32).reshape(1, -1)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        scores, idx = self._index.search(q, min(top_k, len(self._chunks)))
        out = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            out.append((self._chunks[int(i)], float(s)))
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(path / "faiss.index"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)

    def load(self, path: Path) -> None:
        path = Path(path)
        self._index = self._faiss.read_index(str(path / "faiss.index"))
        with open(path / "chunks.pkl", "rb") as f:
            self._chunks = pickle.load(f)

    def __len__(self) -> int:
        return len(self._chunks)


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-backed persistent vector store with cosine similarity."""

    def __init__(self, persist_dir: str | Path = "data/chroma_db", collection_name: str = "guidelines"):
        import chromadb

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._chunks_map: dict[str, EvidenceChunk] = {}

    def add(self, chunks: list[EvidenceChunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        embs = embeddings.astype(np.float32).tolist()
        ids = [f"{c.source}_{i}_{c.chunk_id}" for i, c in enumerate(chunks)]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "title": c.title or "",
                "source": c.source or "",
                "trust_score": float(c.trust_score),
                "specialty": (c.metadata or {}).get("specialty", "general"),
            }
            for c in chunks
        ]

        self._collection.upsert(
            ids=ids,
            embeddings=embs,
            documents=documents,
            metadatas=metadatas,
        )
        for i, c in zip(ids, chunks):
            self._chunks_map[i] = c

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[EvidenceChunk, float]]:
        if len(self) == 0:
            return []
        q_emb = query_embedding.astype(np.float32).flatten().tolist()
        results = self._collection.query(
            query_embeddings=[q_emb],
            n_results=min(top_k, len(self)),
        )

        out: list[tuple[EvidenceChunk, float]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return out

        ids = results["ids"][0]
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
        documents = results["documents"][0] if results.get("documents") else [""] * len(ids)
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)

        for chunk_id, dist, doc_text, meta in zip(ids, distances, documents, metas):
            # Cosine distance to similarity: sim = 1.0 - distance
            sim = float(max(0.0, 1.0 - dist))
            if chunk_id in self._chunks_map:
                chunk = self._chunks_map[chunk_id]
            else:
                chunk = EvidenceChunk(
                    chunk_id=uuid4(),
                    text=doc_text,
                    source=meta.get("source", "ChromaDB"),
                    title=meta.get("title", ""),
                    trust_score=float(meta.get("trust_score", 0.90)),
                    metadata=meta,
                )
            out.append((chunk, sim))

        return out

    def save(self, path: Path) -> None:
        # PersistentClient auto-persists to persist_dir
        pass

    def load(self, path: Path) -> None:
        # Reconnect to PersistentClient
        pass

    def __len__(self) -> int:
        return self._collection.count()


def make_vector_store(backend: str = "memory", persist_dir: str | Path = "data/chroma_db") -> BaseVectorStore:
    if backend == "chroma":
        try:
            return ChromaVectorStore(persist_dir=persist_dir)
        except ImportError:
            return InMemoryVectorStore()
    elif backend == "faiss":
        try:
            return FAISSVectorStore()
        except ImportError:
            return InMemoryVectorStore()
    return InMemoryVectorStore()

