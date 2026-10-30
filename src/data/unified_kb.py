"""Loader for the single unified clinical knowledge base dataset."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import PROJECT_ROOT

UNIFIED_KB_PATH = PROJECT_ROOT / "data" / "unified_clinical_kb.jsonl"


def load_unified_kb(path: Path | str | None = None) -> list[dict]:
    """Load the single unified clinical knowledge base dataset.

    Falls back to SEED_CORPUS if the JSONL file does not exist.
    """
    kb_file = Path(path) if path else UNIFIED_KB_PATH
    if not kb_file.exists():
        from .seed_corpus import SEED_CORPUS
        return SEED_CORPUS

    documents = []
    with kb_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    documents.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return documents

