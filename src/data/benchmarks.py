"""Benchmark dataset loaders for MedQA, PubMedQA, MedMCQA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "external"


def load_medqa(path: Path | None = None) -> list[dict]:
    """Load MedQA dataset (USMLE-style multiple choice)."""
    path = path or DATA_DIR / "medqa.jsonl"
    if not path.exists():
        return []
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        data = item.get("data", {})
        question = data.get("Question", "")
        options = data.get("Options", {})
        correct_opt = data.get("Correct Option", "")
        correct_answer = options.get(correct_opt, "") if correct_opt else data.get("Correct Answer", "")
        items.append({
            "id": item.get("id"),
            "query": question,
            "options": options,
            "correct_option": correct_opt,
            "expected_dx": correct_answer,
            "subject": item.get("subject_name", ""),
        })
    return items


def load_pubmedqa(path: Path | None = None) -> list[dict]:
    """Load PubMedQA dataset (yes/no/maybe biomedical QA)."""
    path = path or DATA_DIR / "pubmedqa.jsonl"
    if not path.exists():
        return []
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        question = item.get("question", "")
        final_decision = item.get("final_decision", "")
        long_answer = item.get("long_answer", "")
        context_list = item.get("context", {}).get("contexts", [])
        context = " ".join(context_list) if context_list else ""
        items.append({
            "id": item.get("pubid"),
            "query": question,
            "context": context,
            "expected_dx": final_decision,  # "yes", "no", "maybe"
            "long_answer": long_answer,
        })
    return items


def load_medmcqa(path: Path | None = None) -> list[dict]:
    """Load MedMCQA dataset (large-scale medical MCQ)."""
    path = path or DATA_DIR / "medmcqa.jsonl"
    if not path.exists():
        return []
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        question = item.get("question", "")
        options = {
            "A": item.get("opa", ""),
            "B": item.get("opb", ""),
            "C": item.get("opc", ""),
            "D": item.get("opd", ""),
        }
        cop = item.get("cop", 0)
        correct_opt = ["A", "B", "C", "D"][cop] if 0 <= cop < 4 else ""
        correct_answer = options.get(correct_opt, "")
        items.append({
            "id": item.get("id"),
            "query": question,
            "options": options,
            "correct_option": correct_opt,
            "expected_dx": correct_answer,
            "subject": item.get("subject_name", ""),
            "explanation": item.get("exp", ""),
        })
    return items


def get_benchmark(benchmark_name: str) -> list[dict]:
    """Load a benchmark by name."""
    loaders = {
        "medqa": load_medqa,
        "pubmedqa": load_pubmedqa,
        "medmcqa": load_medmcqa,
    }
    if benchmark_name not in loaders:
        raise ValueError(f"Unknown benchmark: {benchmark_name}. Available: {list(loaders.keys())}")
    return loaders[benchmark_name]()


if __name__ == "__main__":
    # Quick test
    for name in ["medqa", "pubmedqa", "medmcqa"]:
        items = get_benchmark(name)
        print(f"{name}: {len(items)} items")
        if items:
            print(f"  Sample: {items[0].get('query', '')[:80]}...")
            print(f"  Expected: {items[0].get('expected_dx', '')[:80]}...")