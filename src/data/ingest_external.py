"""External data ingestion: EPFL-LLM Guidelines, MedQA, PubMedQA, etc."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.config import Settings
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ..config import PROJECT_ROOT
from .seed_corpus import SEED_CORPUS

DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external"
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class IngestionConfig:
    embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    chroma_path: str = str(DATA_DIR / "chroma_db")
    collection_name: str = "medical_guidelines"
    batch_size: int = 32
    max_chunks_per_doc: int = 100


class ExternalDataIngestor:
    def __init__(self, config: IngestionConfig | None = None):
        self.config = config or IngestionConfig()
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(self.config.embedding_model, device=device)
        print(f"Using device: {device}")
        self.client = chromadb.PersistentClient(
            path=self.config.chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.embedder.encode(texts, batch_size=self.config.batch_size, show_progress_bar=True)
        return embeddings.tolist()

    def ingest_epfl_guidelines(self, max_docs: int | None = None) -> int:
        """Ingest EPFL-LLM Guidelines corpus from Hugging Face."""
        print("Loading EPFL-LLM Guidelines...")
        ds = load_dataset("epfl-llm/guidelines", split="train", streaming=True)

        count = 0
        batch_texts = []
        batch_metadatas = []
        batch_ids = []

        for item in tqdm(ds, desc="Processing EPFL guidelines"):
            if max_docs and count >= max_docs:
                break

            # EPFL dataset uses 'clean_text' field
            text = item.get("clean_text", "").strip()
            if not text or len(text) < 100:
                continue

            # Chunk long texts
            chunks = self._chunk_text(text, max_chunk_size=512, overlap=50)

            source = item.get("source", "unknown")
            title = item.get("title", "") or f"{source}_guideline"
            url = item.get("url", "")

            trust_scores = {
                "who": 0.95, "cdc": 0.95, "nice": 0.93, "nih": 0.93,
                "esc": 0.92, "aha": 0.92, "ada": 0.90, "kdigo": 0.90,
                "cco": 0.85,  # Cancer Care Ontario
            }
            trust = trust_scores.get(source.lower(), 0.80)

            for chunk_idx, chunk in enumerate(chunks):
                if max_docs and count >= max_docs:
                    break
                batch_texts.append(chunk)
                batch_metadatas.append({
                    "source": f"EPFL/{source}",
                    "title": title,
                    "url": url,
                    "trust_score": trust,
                    "chunk_index": chunk_idx,
                })
                batch_ids.append(f"epfl_{count}")
                count += 1

                if len(batch_texts) >= self.config.batch_size:
                    self._flush_batch(batch_texts, batch_metadatas, batch_ids)
                    batch_texts, batch_metadatas, batch_ids = [], [], []

        if batch_texts:
            self._flush_batch(batch_texts, batch_metadatas, batch_ids)

        print(f"Ingested {count} EPFL guideline chunks")
        return count

    def _chunk_text(self, text: str, max_chunk_size: int = 512, overlap: int = 50) -> list[str]:
        """Simple word-based chunking with overlap."""
        words = text.split()
        if len(words) <= max_chunk_size:
            return [text]

        chunks = []
        step = max_chunk_size - overlap
        for i in range(0, len(words), step):
            chunk_words = words[i:i + max_chunk_size]
            chunks.append(" ".join(chunk_words))
            if i + max_chunk_size >= len(words):
                break
        return chunks

    def _flush_batch(self, texts: list[str], metadatas: list[dict], ids: list[str]):
        embeddings = self._embed_texts(texts)
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def ingest_seed_corpus(self) -> int:
        """Ingest the existing seed corpus as fallback/baseline."""
        print("Ingesting seed corpus...")
        texts = []
        metadatas = []
        ids = []

        for i, doc in enumerate(SEED_CORPUS):
            text = doc["text"]
            texts.append(text)
            metadatas.append({
                "source": doc["source"],
                "title": doc.get("title", ""),
                "year": doc.get("year", 0),
                "trust_score": doc.get("trust_score", 0.8),
            })
            ids.append(f"seed_{i}")

        self._flush_batch(texts, metadatas, ids)
        print(f"Ingested {len(texts)} seed corpus chunks")
        return len(texts)

    def download_benchmarks(self) -> dict[str, Any]:
        """Download evaluation benchmarks to local cache."""
        benchmarks = {}

        print("Downloading MedQA...")
        medqa = load_dataset("openlifescienceai/medqa", split="test")
        benchmarks["medqa"] = medqa
        self._save_benchmark(medqa, "medqa")

        print("Downloading PubMedQA...")
        pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        benchmarks["pubmedqa"] = pubmedqa
        self._save_benchmark(pubmedqa, "pubmedqa")

        print("Downloading MedMCQA...")
        medmcqa = load_dataset("openlifescienceai/medmcqa", split="test")
        benchmarks["medmcqa"] = medmcqa
        self._save_benchmark(medmcqa, "medmcqa")

        print("Downloading MEDIQA-CORR...")
        try:
            mediqa = load_dataset("mediqa_corr", split="test")
            benchmarks["mediqa_corr"] = mediqa
            self._save_benchmark(mediqa, "mediqa_corr")
        except Exception as e:
            print(f"MEDIQA-CORR not available: {e}")

        print("Downloading MultiCaRe...")
        try:
            multicare = load_dataset("multicare", split="test")
            benchmarks["multicare"] = multicare
            self._save_benchmark(multicare, "multicare")
        except Exception as e:
            print(f"MultiCaRe not available: {e}")

        return benchmarks

    def _save_benchmark(self, dataset, name: str):
        path = EXTERNAL_DIR / f"{name}.jsonl"
        with open(path, "w") as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")
        print(f"Saved {name} to {path}")

    def load_benchmark(self, name: str):
        path = EXTERNAL_DIR / f"{name}.jsonl"
        if not path.exists():
            return None
        data = []
        with open(path) as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def get_collection_stats(self) -> dict:
        count = self.collection.count()
        return {"total_chunks": count, "collection": self.config.collection_name}


def run_ingestion(max_epfl_docs: int = 5000):
    """Run full ingestion pipeline."""
    config = IngestionConfig()
    ingestor = ExternalDataIngestor(config)

    print("=== Starting External Data Ingestion ===")

    ingestor.ingest_seed_corpus()
    ingestor.ingest_epfl_guidelines(max_docs=max_epfl_docs)

    stats = ingestor.get_collection_stats()
    print(f"=== Ingestion Complete: {stats} ===")

    print("=== Downloading Benchmarks ===")
    ingestor.download_benchmarks()

    return stats


if __name__ == "__main__":
    run_ingestion(max_epfl_docs=5000)
