"""Document chunking with metadata preservation."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..models import EvidenceChunk


@dataclass
class ChunkingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 100
    splitter: str = "recursive"


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_text_recursive(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks of ~chunk_size tokens with overlap.

    Approximates tokens by whitespace-delimited words (≈1.3 tokens/word).
    """
    words = text.split()
    if not words:
        return []

    approx_tokens_per_word = 1.3
    words_per_chunk = max(1, int(chunk_size / approx_tokens_per_word))
    overlap_words = max(0, int(chunk_overlap / approx_tokens_per_word))

    chunks: list[str] = []
    i = 0
    n = len(words)
    while i < n:
        end = min(i + words_per_chunk, n)
        chunk = " ".join(words[i:end])
        chunks.append(chunk)
        if end == n:
            break
        i = max(end - overlap_words, i + 1)

    return chunks


def split_by_sentences(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Sentence-boundary aware splitting."""
    sentences = _SENTENCE_SPLIT.split(text.strip())
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent.split())
        if current_len + sent_len > chunk_size and current:
            chunks.append(" ".join(current))
            overlap_sents = current[-max(1, chunk_overlap // 50):]
            current = list(overlap_sents)
            current_len = sum(len(s.split()) for s in current)
        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_text(text: str, config: ChunkingConfig) -> list[str]:
    if config.splitter == "sentence":
        return split_by_sentences(text, config.chunk_size, config.chunk_overlap)
    return split_text_recursive(text, config.chunk_size, config.chunk_overlap)


def _detect_specialty(text: str) -> str | None:
    """Lightweight specialty detection from text content."""
    text_l = text.lower()
    specialties = {
        "cardiology": ["heart", "cardiac", "cardiology", "ecg", "chest pain"],
        "pulmonology": ["lung", "pulmonary", "respiratory", "asthma", "copd"],
        "endocrinology": ["diabetes", "thyroid", "endocrine", "insulin"],
        "nephrology": ["kidney", "renal", "dialysis", "creatinine"],
        "infectious_disease": ["infection", "antibiotic", "viral", "bacterial", "covid"],
        "oncology": ["cancer", "tumor", "chemotherapy", "oncology"],
        "pediatrics": ["pediatric", "child", "infant", "neonate"],
        "obstetrics": ["pregnancy", "pregnant", "obstetric", "fetal"],
        "geriatrics": ["elderly", "geriatric", "older adult"],
        "emergency": ["emergency", "trauma", "resuscitation", "triage"],
        "pharmacology": ["drug", "medication", "dosage", "interaction"],
    }
    for specialty, keywords in specialties.items():
        if any(kw in text_l for kw in keywords):
            return specialty
    return None


def build_chunks(
    text: str,
    source: str,
    title: str | None = None,
    year: int | None = None,
    trust_score: float = 0.8,
    config: ChunkingConfig | None = None,
) -> list[EvidenceChunk]:
    """Convert raw text into EvidenceChunk objects."""
    config = config or ChunkingConfig()
    raw_chunks = chunk_text(text, config)
    specialty = _detect_specialty(text[:5000])

    chunks: list[EvidenceChunk] = []
    for idx, raw in enumerate(raw_chunks):
        chunks.append(
            EvidenceChunk(
                text=raw,
                source=source,
                title=title,
                year=year,
                specialty=specialty,
                trust_score=trust_score,
                metadata={"chunk_index": idx, "source": source},
            )
        )
    return chunks


def iter_text_files(root: Path) -> Iterable[Path]:
    for p in Path(root).rglob("*.txt"):
        yield p
    for p in Path(root).rglob("*.md"):
        yield p


def clean_pdf_text(text: str) -> str:
    """Strip common PDF artifacts: headers, footers, references section."""
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"(?i)page\s+\d+\s+of\s+\d+", "", text)
    text = re.sub(r"(?i)^\s*references\s*$", "REFERENCES_SECTION", text, flags=re.MULTILINE)
    text = re.sub(r"REFERENCES_SECTION.*$", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
