"""Stream, filter, and chunk the epfl-llm/guidelines clinical practice guidelines corpus."""

from __future__ import annotations

import logging
from collections.abc import Generator

from ..models import EvidenceChunk

logger = logging.getLogger(__name__)


def stream_guidelines(
    split: str = "train",
    domain: str | None = "infectious_disease",
    limit: int | None = None,
) -> Generator[dict, None, None]:
    """Stream guideline documents from Hugging Face epfl-llm/guidelines dataset.

    Args:
        split: Dataset split ('train').
        domain: Domain filter (e.g. 'infectious_disease', or None/'all' for all domains).
        limit: Max number of raw documents to yield.
    """
    from datasets import load_dataset

    logger.info("Loading dataset epfl-llm/guidelines (split=%s)...", split)
    ds = load_dataset("epfl-llm/guidelines", split=split, streaming=True)

    count = 0
    domain_filter = domain.lower().replace("-", "_") if domain and domain.lower() != "all" else None

    for item in ds:
        specialty = (item.get("specialty") or "").lower().replace("-", "_")
        doc_domain = (item.get("domain") or "").lower().replace("-", "_")
        title = item.get("title") or ""
        text = item.get("text") or item.get("clean_text") or ""

        if not text.strip():
            continue

        # If domain filter is specified, check match in specialty, domain, or title
        if domain_filter:
            matches = (
                domain_filter in specialty
                or domain_filter in doc_domain
                or domain_filter in title.lower()
            )
            # Support partial key matches like 'infectious' for 'infectious_disease'
            if not matches and "infectious" in domain_filter:
                matches = "infectious" in specialty or "infectious" in doc_domain or "infection" in title.lower()
            if not matches:
                continue

        yield {
            "title": title,
            "text": text,
            "specialty": item.get("specialty") or "general",
            "source": item.get("source") or "epfl-llm/guidelines",
            "year": str(item.get("year") or item.get("publication_year") or "2023"),
            "url": item.get("url") or item.get("source_url") or "",
        }

        count += 1
        if limit and count >= limit:
            break


def chunk_document(
    doc: dict,
    chunk_size: int = 512,
    chunk_overlap: int = 100,
) -> list[EvidenceChunk]:
    """Structure-aware token/word chunking for a clinical guideline document."""
    text = doc["text"]
    title = doc["title"]
    source = doc["source"]
    specialty = doc["specialty"]
    year = doc["year"]

    words = text.split()
    if not words:
        return []

    step = max(1, chunk_size - chunk_overlap)
    chunks: list[EvidenceChunk] = []

    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        chunk_text = " ".join(chunk_words)

        # Detect clinical section header cues if present
        lower_prefix = chunk_text[:80].lower()
        section = "general"
        if any(w in lower_prefix for w in ["diagnosis", "diagnostic", "presentation"]):
            section = "diagnosis"
        elif any(w in lower_prefix for w in ["treatment", "therapy", "management", "first-line"]):
            section = "treatment"
        elif any(w in lower_prefix for w in ["contraindication", "adverse", "warning", "caution"]):
            section = "contraindications"
        elif any(w in lower_prefix for w in ["dosage", "dosing", "administration"]):
            section = "dosing"

        chunk = EvidenceChunk(
            id=f"{source}_{specialty}_{i // step}",
            text=chunk_text,
            source=f"{source} ({specialty}, {year})",
            title=title,
            trust_score=0.92,
            metadata={
                "specialty": specialty,
                "section": section,
                "year": year,
                "chunk_idx": i // step,
            },
        )
        chunks.append(chunk)

    return chunks


def load_and_chunk_guidelines(
    domain: str | None = "infectious_disease",
    limit: int | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 100,
) -> list[EvidenceChunk]:
    """Load guideline documents and return flattened chunk list."""
    all_chunks: list[EvidenceChunk] = []
    for doc in stream_guidelines(domain=domain, limit=limit):
        chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)
    logger.info("Loaded %d chunks from epfl-llm/guidelines (domain=%s)", len(all_chunks), domain)
    return all_chunks
