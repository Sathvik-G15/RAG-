"""Ingest all local benchmark datasets into ChromaDB for fully offline use.

This embeds benchmark content (questions + context + answers) as medical
knowledge chunks into the existing ChromaDB collection, enriching the RAG
retrieval pool without any internet access.

Particularly valuable sources:
  - MTSamples   : rich real clinical transcription narratives
  - PubMedQA    : biomedical question + abstract context
  - DDXPlus     : differential diagnosis case knowledge
  - MedQA       : clinical vignettes with USMLE reasoning
  - MMLU-Medical: board-level medical MCQ knowledge
  - MedMCQA     : Indian medical entrance clinical Q&A

Run:
    python scripts/ingest_benchmarks.py
    python scripts/ingest_benchmarks.py --datasets mtsamples pubmedqa --limit 500
    python scripts/ingest_benchmarks.py --dry-run  (count docs without writing)
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

# ─── Environment must be set BEFORE torch imports ────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_OFFLINE"] = "1"   # use local cache only
os.environ["HF_DATASETS_OFFLINE"] = "1"   # use local cache only

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_env = PROJECT_ROOT / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env, override=False)
    except ImportError:
        pass

HF_CACHE = PROJECT_ROOT / "data" / "hf_cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE)

DATA_DIR    = PROJECT_ROOT / "data" / "external"
CHROMA_DIR  = PROJECT_ROOT / "data" / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[ingest] {msg}", flush=True)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return items


# ─── Document builders ────────────────────────────────────────────────────────
# Each builder returns a list of {"text": str, "metadata": dict} dicts.
# We keep text ≤ 800 chars so the embedding model isn't overwhelmed.

def _truncate(s: str, n: int = 800) -> str:
    return s[:n] if len(s) > n else s


def build_medqa_docs(items: list[dict]) -> list[dict]:
    """Clinical vignette + options as a single indexed chunk."""
    docs: list[dict] = []
    for item in items:
        q = item.get("query", "")
        opts = item.get("options") or {}
        answer = item.get("expected_dx", "")
        opt_str = " | ".join(f"{k}: {v}" for k, v in sorted(opts.items()))
        text = f"USMLE Clinical Question: {q}\nOptions: {opt_str}\nCorrect Answer: {answer}"
        docs.append({"text": _truncate(text), "metadata": {
            "source": "medqa", "dataset": "medqa",
            "title": q[:80],
        }})
    return docs


def build_pubmedqa_docs(items: list[dict]) -> list[dict]:
    """Question + abstract context as primary chunk; long_answer as second chunk."""
    docs: list[dict] = []
    for item in items:
        q = item.get("query", "")
        ctx_raw = item.get("context") or ""
        if isinstance(ctx_raw, dict):
            ctx = " ".join(ctx_raw.get("contexts") or [])
        else:
            ctx = str(ctx_raw)
        ctx = ctx.strip()
        decision = item.get("expected_dx", "")
        long_ans = (item.get("long_answer") or "").strip()

        # Chunk 1: abstract context
        if ctx:
            text = f"Biomedical Question: {q}\nAbstract: {ctx}\nConclusion: {decision}"
            docs.append({"text": _truncate(text, 1000), "metadata": {
                "source": "pubmedqa", "dataset": "pubmedqa",
                "title": q[:80],
            }})
        # Chunk 2: long answer (often a concise clinical summary)
        if long_ans and len(long_ans) > 30:
            docs.append({"text": _truncate(f"Q: {q}\nAnswer: {long_ans}"), "metadata": {
                "source": "pubmedqa", "dataset": "pubmedqa",
                "title": f"[long_answer] {q[:60]}",
            }})
    return docs


def build_medmcqa_docs(items: list[dict]) -> list[dict]:
    """MCQ + explanation as clinical knowledge chunk."""
    docs: list[dict] = []
    for item in items:
        q = item.get("query", "")
        opts = item.get("options") or {}
        answer = item.get("expected_dx", "")
        exp = (item.get("explanation") or "").strip()
        subject = item.get("subject", "")
        opt_str = " | ".join(f"{k}: {v}" for k, v in sorted(opts.items()))
        text = f"[{subject}] {q}\nOptions: {opt_str}\nAnswer: {answer}"
        if exp:
            text += f"\nExplanation: {exp}"
        docs.append({"text": _truncate(text, 900), "metadata": {
            "source": "medmcqa", "dataset": "medmcqa",
            "title": q[:80], "subject": subject,
        }})
    return docs


def build_ddxplus_docs(items: list[dict]) -> list[dict]:
    """Clinical case (symptoms) + diagnosis + differential as retrieval chunk."""
    docs: list[dict] = []
    for item in items:
        q = item.get("query", "")          # "Symptoms: E_48, ..."
        dx = item.get("expected_dx", "")
        diff = item.get("differential") or []
        diff_str = ", ".join(
            f"{d['diagnosis']} ({d['probability']:.0%})"
            for d in diff[:5]
            if isinstance(d, dict)
        )
        text = f"Clinical Case: {q}\nDiagnosis: {dx}"
        if diff_str:
            text += f"\nDifferential: {diff_str}"
        docs.append({"text": _truncate(text), "metadata": {
            "source": "ddxplus", "dataset": "ddxplus",
            "title": f"DDXPlus: {dx}",
        }})
    return docs


def build_mtsamples_docs(items: list[dict]) -> list[dict]:
    """Full clinical transcription split into overlapping 800-char chunks."""
    docs: list[dict] = []
    CHUNK = 800
    OVERLAP = 150
    for item in items:
        specialty = item.get("specialty", "General")
        ctx = (item.get("context") or item.get("query") or "").strip()
        if not ctx:
            continue
        # Sliding window chunking
        start = 0
        chunk_idx = 0
        while start < len(ctx):
            chunk = ctx[start: start + CHUNK]
            docs.append({"text": chunk, "metadata": {
                "source": "mtsamples", "dataset": "mtsamples",
                "title": f"[{specialty}] Clinical Note (chunk {chunk_idx})",
                "specialty": specialty,
            }})
            start += CHUNK - OVERLAP
            chunk_idx += 1
    return docs


def build_mmlumed_docs(items: list[dict]) -> list[dict]:
    """Board-level MCQ as clinical knowledge snippets."""
    docs: list[dict] = []
    for item in items:
        q = item.get("query", "")
        opts = item.get("options") or {}
        answer = item.get("expected_dx", "")
        subject = item.get("subject", "Medical")
        opt_str = " | ".join(f"{k}: {v}" for k, v in sorted(opts.items()))
        text = f"[{subject}] {q}\nOptions: {opt_str}\nAnswer: {answer}"
        docs.append({"text": _truncate(text), "metadata": {
            "source": "mmlumed", "dataset": "mmlumed",
            "title": q[:80], "subject": subject,
        }})
    return docs


BUILDERS: dict[str, Any] = {
    "medqa":     build_medqa_docs,
    "pubmedqa":  build_pubmedqa_docs,
    "medmcqa":   build_medmcqa_docs,
    "ddxplus":   build_ddxplus_docs,
    "mtsamples": build_mtsamples_docs,
    "mmlumed":   build_mmlumed_docs,
}


# ─── ChromaDB ingest ─────────────────────────────────────────────────────────

def ingest_to_chroma(
    docs: list[dict],
    embedder,
    vector_store,
    batch_size: int = 64,
    dry_run: bool = False,
) -> int:
    """Embed and upsert docs into ChromaDB. Returns number inserted."""
    if not docs:
        return 0
    if dry_run:
        return len(docs)

    from src.models import EvidenceChunk
    import uuid

    inserted = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i: i + batch_size]
        texts = [d["text"] for d in batch]
        metas = [d["metadata"] for d in batch]

        try:
            embeddings = embedder.embed(texts)
        except Exception as e:
            log(f"  Embedding error (batch {i//batch_size}): {e}")
            continue

        chunks: list[EvidenceChunk] = []
        for j, (text, meta, emb) in enumerate(zip(texts, metas, embeddings)):
            chunk_id = str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{meta.get('dataset','')}/{meta.get('title','')}/{i+j}"
            ))
            chunks.append(EvidenceChunk(
                id=chunk_id,
                text=text,
                source=meta.get("source", "benchmark"),
                title=meta.get("title", ""),
                score=1.0,
                metadata=meta,
            ))

        try:
            vector_store.add(chunks, embeddings)
            inserted += len(chunks)
        except Exception as e:
            log(f"  Upsert error (batch {i//batch_size}): {e}")

        # Light GC between batches
        if (i // batch_size) % 10 == 0:
            gc.collect()

    return inserted


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Ingest benchmark datasets into ChromaDB")
    parser.add_argument(
        "--datasets", nargs="+",
        choices=list(BUILDERS),
        default=list(BUILDERS),
        help="Which datasets to ingest (default: all)",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max items per dataset (default: all)")
    parser.add_argument("--chroma-dir", type=str,
                        default=str(CHROMA_DIR),
                        help="Path to ChromaDB directory")
    parser.add_argument("--embedding-model", type=str,
                        default="BAAI/bge-small-en-v1.5",
                        help="Embedding model name")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Embedding batch size")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count documents without writing to ChromaDB")
    args = parser.parse_args()

    log(f"ChromaDB path : {args.chroma_dir}")
    log(f"Embedding     : {args.embedding_model}")
    log(f"Dry run       : {args.dry_run}")

    if not args.dry_run:
        # Load embedder and vector store
        log("Loading embedding model ...")
        from src.kb.embedder import SentenceTransformerEmbedder
        from src.kb.vector_store import ChromaVectorStore

        embedder = SentenceTransformerEmbedder(model_name=args.embedding_model)
        vector_store = ChromaVectorStore(persist_dir=args.chroma_dir)
        log("Models loaded.")
    else:
        embedder = vector_store = None

    total_inserted = 0
    for name in args.datasets:
        jsonl_path = DATA_DIR / f"{name}.jsonl"
        items = read_jsonl(jsonl_path)
        if not items:
            log(f"{name}: no data found at {jsonl_path.name} - skipping.")
            continue

        if args.limit:
            items = items[: args.limit]

        log(f"{name}: building documents from {len(items):,} items ...")
        docs = BUILDERS[name](items)
        log(f"  -> {len(docs):,} chunks")

        if not args.dry_run:
            n = ingest_to_chroma(
                docs, embedder, vector_store,
                batch_size=args.batch_size,
                dry_run=False,
            )
            log(f"  -> Inserted {n:,} chunks into ChromaDB")
            total_inserted += n
            gc.collect()
        else:
            total_inserted += len(docs)

    action = "Would insert" if args.dry_run else "Total inserted"
    log(f"{action}: {total_inserted:,} chunks across {len(args.datasets)} datasets")
    log("Done.")


if __name__ == "__main__":
    main()
