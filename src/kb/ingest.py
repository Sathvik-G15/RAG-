"""KB ingestion: text -> chunks -> embeddings -> vector store."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from ..config import get_kb_config, set_global_seeds
from ..models import EvidenceChunk
from .chunker import ChunkingConfig, build_chunks, clean_pdf_text, iter_text_files
from .embedder import BaseEmbedder, make_embedder
from .vector_store import BaseVectorStore, InMemoryVectorStore, make_vector_store


class KnowledgeBase:
    def __init__(self, embedder: BaseEmbedder, store: BaseVectorStore):
        self.embedder = embedder
        self.store = store

    @classmethod
    def from_config(cls, prefer_real_embedder: bool = False) -> KnowledgeBase:
        set_global_seeds(42)
        kb_cfg = get_kb_config()
        embedder = make_embedder(
            model_name=kb_cfg.get("embeddings", {}).get("model_name"),
            prefer_real=prefer_real_embedder,
        )
        store = make_vector_store(backend="memory")
        return cls(embedder=embedder, store=store)

    def add_documents(
        self,
        documents: Sequence[dict],
        chunking_config: ChunkingConfig | None = None,
    ) -> int:
        """Add documents to the KB.

        Each document is a dict with: text, source, title?, year?, trust_score?
        """
        all_chunks: list[EvidenceChunk] = []
        for doc in documents:
            text = clean_pdf_text(doc["text"])
            chunks = build_chunks(
                text=text,
                source=doc["source"],
                title=doc.get("title"),
                year=doc.get("year"),
                trust_score=doc.get("trust_score", 0.8),
                config=chunking_config or ChunkingConfig(),
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        texts = [c.text for c in all_chunks]
        embeddings = self.embedder.embed(texts)
        self.store.add(all_chunks, embeddings)
        return len(all_chunks)

    def add_text_file(self, path: Path, source: str, trust_score: float = 0.8) -> int:
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="ignore")
        return self.add_documents(
            [
                {
                    "text": text,
                    "source": source,
                    "title": path.stem,
                    "trust_score": trust_score,
                }
            ]
        )

    def add_directory(self, root: Path, source_tag: str, trust_score: float = 0.8) -> int:
        root = Path(root)
        if not root.exists():
            return 0
        n = 0
        for p in iter_text_files(root):
            n += self.add_text_file(p, source=f"{source_tag}/{p.stem}", trust_score=trust_score)
        return n

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.store.save(path)
        meta = {"embedder_dim": self.embedder.dim, "n_chunks": len(self.store)}
        (path / "kb_meta.json").write_text(json.dumps(meta))

    def load(self, path: Path) -> None:
        path = Path(path)
        if isinstance(self.store, InMemoryVectorStore):
            self.store.load(path)
        else:
            self.store.load(path)
