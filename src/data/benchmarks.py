"""Benchmark dataset loaders for MedQA, PubMedQA, MedMCQA, DDXPlus, MTSamples, HeadQA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "external"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict]:
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


def build_query_prompt(item: dict) -> str:
    """Build a rich LLM prompt from a benchmark item.

    For MCQ datasets (MedQA, MedMCQA, HeadQA) the answer choices are
    appended so the model can select the correct option rather than free-form
    diagnose, dramatically improving measured accuracy.

    For open-ended datasets (DDXPlus, MTSamples, PubMedQA) the clinical
    context is prepended as-is.
    """
    query: str = item.get("query", "") or ""
    options: dict[str, str] = item.get("options") or {}
    context: str = (item.get("context") or "").strip()

    parts: list[str] = []
    if context:
        parts.append(f"[Clinical Context]\n{context[:1200]}")
    parts.append(f"[Question]\n{query}")
    if options:
        opt_lines = "\n".join(f"  {k}. {v}" for k, v in sorted(options.items()))
        parts.append(f"[Options]\n{opt_lines}")
        parts.append(
            "Return JSON where primary_diagnosis is the letter (A/B/C/D) "
            "of the correct option followed by the option text."
        )
    else:
        parts.append(
            "Return JSON with: primary_diagnosis, confidence, differential, reasoning."
        )
    return "\n\n".join(parts)


def load_medqa(path: Path | None = None) -> list[dict]:
    """MedQA – USMLE-style 4-option MCQ."""
    raw = _read_jsonl(path or DATA_DIR / "medqa.jsonl")
    out: list[dict] = []
    for item in raw:
        data = item.get("data") or item
        question = data.get("Question") or data.get("query") or ""
        options = data.get("Options") or data.get("options") or {}
        correct_opt = data.get("Correct Option") or data.get("correct_option") or ""
        correct_answer = (
            options.get(correct_opt, "") if correct_opt
            else data.get("Correct Answer") or data.get("expected_dx") or ""
        )
        out.append({
            "id": item.get("id", ""),
            "query": question,
            "options": options,
            "correct_option": correct_opt,
            "expected_dx": correct_answer,
            "subject": item.get("subject_name", "") or item.get("subject", ""),
            "dataset": "medqa",
        })
    return out


def load_pubmedqa(path: Path | None = None) -> list[dict]:
    """PubMedQA – biomedical yes/no/maybe QA with abstract context."""
    raw = _read_jsonl(path or DATA_DIR / "pubmedqa.jsonl")
    out: list[dict] = []
    for item in raw:
        question = item.get("question") or item.get("query") or ""
        ctx = item.get("context") or {}
        if isinstance(ctx, dict):
            context_str = " ".join(ctx.get("contexts") or [])
        else:
            context_str = str(ctx)
        out.append({
            "id": item.get("pubid") or item.get("id") or "",
            "query": question,
            "context": context_str,
            "expected_dx": item.get("final_decision") or item.get("expected_dx") or "",
            "long_answer": item.get("long_answer", ""),
            "dataset": "pubmedqa",
        })
    return out


def load_medmcqa(path: Path | None = None) -> list[dict]:
    """MedMCQA – large-scale Indian medical entrance MCQ."""
    raw = _read_jsonl(path or DATA_DIR / "medmcqa.jsonl")
    out: list[dict] = []
    for item in raw:
        question = item.get("question") or item.get("query") or ""
        options = item.get("options") or {
            "A": item.get("opa", ""),
            "B": item.get("opb", ""),
            "C": item.get("opc", ""),
            "D": item.get("opd", ""),
        }
        correct_opt = item.get("correct_option", "")
        if not correct_opt:
            cop = int(item.get("cop") or 0)
            correct_opt = ["A", "B", "C", "D"][cop] if 0 <= cop < 4 else ""
        correct_answer = item.get("expected_dx") or options.get(correct_opt, "")
        out.append({
            "id": item.get("id", ""),
            "query": question,
            "options": options,
            "correct_option": correct_opt,
            "expected_dx": correct_answer,
            "subject": item.get("subject_name", "") or item.get("subject", ""),
            "explanation": item.get("exp", "") or item.get("explanation", ""),
            "dataset": "medmcqa",
        })
    return out


def load_ddxplus(path: Path | None = None) -> list[dict]:
    """DDXPlus – synthetic clinical cases with differential diagnosis ground-truth."""
    raw = _read_jsonl(path or DATA_DIR / "ddxplus.jsonl")
    out: list[dict] = []
    for item in raw:
        query = item.get("query") or ""
        expected = item.get("expected_dx") or item.get("PATHOLOGY") or ""
        differential = item.get("differential") or []
        out.append({
            "id": item.get("id") or item.get("ID", ""),
            "query": query,
            "expected_dx": str(expected),
            "differential": differential,
            "dataset": "ddxplus",
        })
    return out


def load_mtsamples(path: Path | None = None) -> list[dict]:
    """MTSamples – real clinical transcription notes across 40 medical specialties."""
    raw = _read_jsonl(path or DATA_DIR / "mtsamples.jsonl")
    out: list[dict] = []
    for item in raw:
        query = item.get("query") or item.get("description") or ""
        context = item.get("context") or item.get("transcription") or ""
        expected = item.get("expected_dx") or item.get("sample_name") or item.get("specialty") or ""
        if not query.strip():
            continue
        out.append({
            "id": item.get("id", ""),
            "query": query,
            "context": str(context)[:2000],
            "expected_dx": expected,
            "specialty": item.get("specialty") or item.get("medical_specialty", ""),
            "dataset": "mtsamples",
        })
    return out


def load_mmlumed(path: Path | None = None) -> list[dict]:
    """MMLU-Medical – 6 medical subsets (clinical_knowledge, medical_genetics,
    anatomy, professional_medicine, college_medicine, college_biology) from
    cais/mmlu. Standard 4-option MCQ, pure Parquet, ~1400 questions total.
    """
    raw = _read_jsonl(path or DATA_DIR / "mmlumed.jsonl")
    out: list[dict] = []
    for item in raw:
        question = item.get("query") or item.get("question") or ""
        options = item.get("options") or {}
        correct_opt = item.get("correct_option") or ""
        correct_answer = item.get("expected_dx") or options.get(correct_opt, "")
        out.append({
            "id": item.get("id", ""),
            "query": question,
            "options": options,
            "correct_option": correct_opt,
            "expected_dx": correct_answer,
            "subject": item.get("subject", ""),
            "dataset": "mmlumed",
        })
    return out


# ─── Registry ─────────────────────────────────────────────────────────────────

_LOADERS: dict[str, Any] = {
    "medqa":     load_medqa,
    "pubmedqa":  load_pubmedqa,
    "medmcqa":   load_medmcqa,
    "ddxplus":   load_ddxplus,
    "mtsamples": load_mtsamples,
    "mmlumed":   load_mmlumed,
}

AVAILABLE_BENCHMARKS: list[str] = list(_LOADERS)


def get_benchmark(benchmark_name: str) -> list[dict]:
    """Load a benchmark by name. Returns [] if the JSONL file is missing."""
    if benchmark_name not in _LOADERS:
        raise ValueError(
            f"Unknown benchmark: {benchmark_name!r}. "
            f"Available: {AVAILABLE_BENCHMARKS}"
        )
    return _LOADERS[benchmark_name]()


def list_available() -> dict[str, int]:
    """Return {name: count} for every dataset file that exists on disk."""
    return {name: len(get_benchmark(name)) for name in AVAILABLE_BENCHMARKS
            if (DATA_DIR / f"{name}.jsonl").exists()}


# ─── CLI quick-check ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    avail = list_available()
    if not avail:
        print("No datasets found. Run: python scripts/download_datasets.py")
    for name, count in avail.items():
        items = get_benchmark(name)
        print(f"{name}: {count:,} items")
        sample = items[0]
        print(f"  query:       {sample.get('query', '')[:80]}...")
        print(f"  expected_dx: {sample.get('expected_dx', '')[:80]}")
        if sample.get("options"):
            print(f"  options:     {sample['options']}")
        print()
