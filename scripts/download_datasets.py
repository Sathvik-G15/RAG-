"""Download all freely accessible medical benchmark datasets.

Datasets downloaded:
  - DDXPlus      (mila-nlp/DDXPlus)                    - synthetic differential-diagnosis cases
  - MTSamples    (rungalileo/medical_transcription_40)  - clinical transcription notes
  - HeadQA       (dvilares/head_qa)                    - Spanish medical board exam (EN split)
  - MedMCQA      (openlifescienceai/medmcqa)           - Indian medical entrance MCQ
  - PubMedQA     (qiaojin/PubMedQA)                    - biomedical yes/no/maybe QA
  - MedQA-USMLE  (bigbio/med_qa)                       - USMLE-style MCQ

Already existing local files are skipped (idempotent).
Run:  python scripts/download_datasets.py
      python scripts/download_datasets.py --datasets ddxplus headqa --n 500
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ─── Setup paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "external"
DATA_DIR.mkdir(parents=True, exist_ok=True)

_env = PROJECT_ROOT / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env, override=False)
    except ImportError:
        pass


# ─── Helpers ──────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[download] {msg}", flush=True)


def save_jsonl(records: list[dict], path: Path) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
        encoding="utf-8",
    )
    log(f"  -> Saved {len(records):,} records to {path.name}")


def load_hf(dataset_id: str, split: str, name: str | None = None,
            trust_remote_code: bool = False) -> list[dict]:
    from datasets import load_dataset
    kwargs: dict = {"split": split}
    if name:
        kwargs["name"] = name
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    ds = load_dataset(dataset_id, **kwargs)
    return list(ds)


# ─── DDXPlus ──────────────────────────────────────────────────────────────────

# DDXPlus symptom codebook URL (original MILA release)
_DDXPLUS_EVIDENCES_URL = (
    "https://raw.githubusercontent.com/mila-iqia/ddxplus/main/"
    "release_evidences.json"
)


def _load_ddxplus_codebook() -> dict[str, str]:
    """Fetch symptom code -> human-readable name mapping.
    Returns empty dict on failure (codes will be used as-is).
    """
    import urllib.request
    try:
        with urllib.request.urlopen(_DDXPLUS_EVIDENCES_URL, timeout=15) as resp:
            evidences = json.loads(resp.read().decode())
        code_map: dict[str, str] = {}
        for code, info in evidences.items():
            name = info.get("name", code)
            code_map[code] = name
        log(f"  DDXPlus codebook loaded: {len(code_map)} symptom codes")
        return code_map
    except Exception as e:
        log(f"  DDXPlus codebook not available ({e}), using raw codes")
        return {}


def download_ddxplus(n: int = 2000) -> None:
    out = DATA_DIR / "ddxplus.jsonl"
    if out.exists():
        log("ddxplus.jsonl already exists - skipping.")
        return
    log("Downloading DDXPlus (aai530-group6/ddxplus) ...")
    raw = load_hf("aai530-group6/ddxplus", split=f"train[:{n}]")
    if not raw:
        log("  DDXPlus: empty split returned.")
        return
    log(f"  DDXPlus columns: {list(raw[0].keys())}")

    # Fetch symptom codebook to decode E_XX codes
    codebook = _load_ddxplus_codebook()

    import ast

    def decode_evidence(code: str) -> str:
        """E_54_@_V_161 -> 'symptom_name: value_name' or raw code."""
        base = code.split("_@_")[0]
        value = code.split("_@_")[1] if "_@_" in code else None
        name = codebook.get(base, base)
        if value:
            return f"{name} ({value})"
        return name

    def parse_list(s) -> list:
        if isinstance(s, list):
            return s
        if isinstance(s, str):
            try:
                return ast.literal_eval(s)
            except Exception:
                return []
        return []

    records = []
    for row in raw:
        pathology = str(row.get("PATHOLOGY") or "")
        age = row.get("AGE")
        sex = row.get("SEX", "")

        evidences_raw = parse_list(row.get("EVIDENCES", []))
        evidence_names = [decode_evidence(str(e)) for e in evidences_raw]

        diff_raw = parse_list(row.get("DIFFERENTIAL_DIAGNOSIS", []))
        differential = []
        for item in diff_raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                differential.append({
                    "diagnosis": str(item[0]),
                    "probability": float(item[1]),
                })
            elif isinstance(item, dict):
                differential.append({
                    "diagnosis": item.get("disease", item.get("pathology", "")),
                    "probability": float(item.get("probability", 0.5)),
                })

        query_parts: list[str] = []
        if evidence_names:
            query_parts.append("Symptoms: " + ", ".join(evidence_names[:20]))
        if age is not None:
            query_parts.append(f"Age: {age}")
        if sex:
            query_parts.append(f"Sex: {sex}")
        query = ". ".join(query_parts) or "No clinical information provided."

        records.append({
            "id": str(row.get("ID", row.get("id", ""))),
            "query": query,
            "expected_dx": pathology,
            "differential": differential,
            "dataset": "ddxplus",
        })
    save_jsonl(records, out)


# ─── MTSamples ────────────────────────────────────────────────────────────────

# MTSamples specialty label map (label int -> specialty name)
_MTSAMPLES_LABELS: dict[int, str] = {
    0: "Allergy / Immunology", 1: "Bariatrics", 2: "Cardiovascular / Pulmonary",
    3: "Dentistry", 4: "Dermatology", 5: "Diets and Nutritions",
    6: "Discharge Summary", 7: "Emergency Room Reports", 8: "Endocrinology",
    9: "ENT - Otolaryngology", 10: "Gastroenterology", 11: "General Medicine",
    12: "Hematology - Oncology", 13: "Hospice - Palliative Care",
    14: "IME-QME-Work Comp etc.", 15: "Lab Medicine - Pathology",
    16: "Letters", 17: "Nephrology", 18: "Neurology", 19: "Neurosurgery",
    20: "Obstetrics / Gynecology", 21: "Office Notes", 22: "Ophthalmology",
    23: "Orthopedic", 24: "Pain Management", 25: "Pediatrics - Neonatal",
    26: "Physical Medicine - Rehab", 27: "Podiatry", 28: "Psychiatry / Psychology",
    29: "Radiology", 30: "Rheumatology", 31: "Sleep Medicine",
    32: "SOAP / Chart / Progress Notes", 33: "Speech - Language",
    34: "Surgery", 35: "Urology",
}


def download_mtsamples(n: int = 1000) -> None:
    out = DATA_DIR / "mtsamples.jsonl"
    if out.exists():
        log("mtsamples.jsonl already exists - skipping.")
        return
    log("Downloading MTSamples (rungalileo/medical_transcription_40) ...")
    raw = load_hf("rungalileo/medical_transcription_40", split=f"train[:{n}]")
    records = []
    for idx, row in enumerate(raw):
        # Actual schema: id (int), text (str - full transcription), label (int)
        text = (row.get("text") or "").strip()
        label_int = row.get("label", -1)
        specialty = _MTSAMPLES_LABELS.get(label_int, str(label_int))
        if not text:
            continue
        # Split text: the last part after the final specialty mention is keywords
        # Use the full text as the clinical context; first 300 chars as query
        query_end = text.find(",", 200)
        query = text[:query_end].strip() if query_end > 0 else text[:300].strip()
        records.append({
            "id": str(row.get("id", idx)),
            "query": query,
            "context": text[:2000],
            "expected_dx": specialty,
            "specialty": specialty,
            "dataset": "mtsamples",
        })
    save_jsonl(records, out)


# ─── HeadQA (English split) ───────────────────────────────────────────────────

# MMLU medical subsets (6 domains -> ~1,400 clean MCQ questions, pure Parquet)
_MMLU_MEDICAL_CONFIGS = [
    "clinical_knowledge",
    "medical_genetics",
    "anatomy",
    "professional_medicine",
    "college_medicine",
    "college_biology",
]


def download_mmlumed(n: int = 1500) -> None:
    """Download MMLU Medical subsets (cais/mmlu, 6 medical domains)."""
    out = DATA_DIR / "mmlumed.jsonl"
    if out.exists():
        log("mmlumed.jsonl already exists - skipping.")
        return
    log("Downloading MMLU-Medical (cais/mmlu, 6 medical subsets) ...")
    # MMLU schema: question (str), choices (list[str] len 4), answer (int 0-3)
    records: list[dict] = []
    per_subset = max(1, n // len(_MMLU_MEDICAL_CONFIGS))
    for cfg in _MMLU_MEDICAL_CONFIGS:
        try:
            raw = load_hf("cais/mmlu", split=f"test[:{per_subset}]", name=cfg)
            for row in raw:
                question = (row.get("question") or "").strip()
                choices = row.get("choices") or []
                answer_idx = int(row.get("answer") or 0)
                opts: dict[str, str] = {}
                for i, ch in enumerate(choices):
                    opts[chr(ord("A") + i)] = str(ch)
                correct_opt = chr(ord("A") + answer_idx) if 0 <= answer_idx < len(choices) else ""
                correct_text = opts.get(correct_opt, "")
                records.append({
                    "id": f"{cfg}_{row.get('id', len(records))}",
                    "query": question,
                    "options": opts,
                    "correct_option": correct_opt,
                    "expected_dx": correct_text,
                    "subject": cfg.replace("_", " ").title(),
                    "dataset": "mmlumed",
                })
        except Exception as e:
            log(f"  Warning: MMLU subset '{cfg}' failed: {e}")
    if records:
        save_jsonl(records, out)
    else:
        log("  x No MMLU records saved.")


# ─── MedMCQA ─────────────────────────────────────────────────────────────────

def download_medmcqa(n: int = 2000) -> None:
    out = DATA_DIR / "medmcqa.jsonl"
    if out.exists():
        log("medmcqa.jsonl already exists - skipping.")
        return
    log("Downloading MedMCQA ...")
    raw = load_hf("openlifescienceai/medmcqa", split=f"validation[:{n}]")
    records = []
    for row in raw:
        question = (row.get("question") or "").strip()
        opts = {
            "A": row.get("opa") or "",
            "B": row.get("opb") or "",
            "C": row.get("opc") or "",
            "D": row.get("opd") or "",
        }
        cop_idx = int(row.get("cop") or 0)
        correct_opt = ["A", "B", "C", "D"][cop_idx] if 0 <= cop_idx < 4 else ""
        correct_answer = opts.get(correct_opt, "")
        records.append({
            "id": str(row.get("id", "")),
            "query": question,
            "options": opts,
            "correct_option": correct_opt,
            "expected_dx": correct_answer,
            "subject": row.get("subject_name", ""),
            "explanation": row.get("exp", ""),
            "dataset": "medmcqa",
        })
    save_jsonl(records, out)


# ─── PubMedQA ────────────────────────────────────────────────────────────────

def download_pubmedqa(n: int = 1000) -> None:
    out = DATA_DIR / "pubmedqa.jsonl"
    if out.exists():
        log("pubmedqa.jsonl already exists - skipping.")
        return
    log("Downloading PubMedQA ...")
    raw = load_hf("qiaojin/PubMedQA", split=f"train[:{n}]", name="pqa_labeled")
    records = []
    for row in raw:
        question = (row.get("question") or "").strip()
        ctx = row.get("context") or {}
        contexts = ctx.get("contexts", []) if isinstance(ctx, dict) else []
        context_str = " ".join(contexts)
        records.append({
            "id": str(row.get("pubid", "")),
            "query": question,
            "context": context_str,
            "expected_dx": row.get("final_decision", ""),
            "long_answer": row.get("long_answer", ""),
            "dataset": "pubmedqa",
        })
    save_jsonl(records, out)


# ─── MedQA (USMLE via BigBio) ────────────────────────────────────────────────

def download_medqa(n: int = 1000) -> None:
    out = DATA_DIR / "medqa.jsonl"
    if out.exists():
        log("medqa.jsonl already exists - skipping.")
        return
    log("Downloading MedQA (USMLE) ...")
    raw = load_hf("bigbio/med_qa", split=f"test[:{n}]", name="med_qa_en_bigbio_qa")
    records = []
    for i, row in enumerate(raw):
        question = (row.get("question") or "").strip()
        choices = row.get("choices") or []
        opts: dict[str, str] = {}
        for c in choices:
            k = str(c.get("key", ""))
            v = str(c.get("value", ""))
            opts[k] = v
        answer = row.get("answer", [])
        correct_text = answer[0] if isinstance(answer, list) and answer else str(answer or "")
        correct_opt = next((k for k, v in opts.items() if v == correct_text), "")
        records.append({
            "id": str(row.get("id", i)),
            "query": question,
            "options": opts,
            "correct_option": correct_opt,
            "expected_dx": correct_text,
            "dataset": "medqa",
        })
    save_jsonl(records, out)


# ─── Registry & CLI ───────────────────────────────────────────────────────────

DOWNLOADERS: dict[str, object] = {
    "ddxplus":   download_ddxplus,
    "mtsamples": download_mtsamples,
    "mmlumed":   download_mmlumed,
    "medmcqa":   download_medmcqa,
    "pubmedqa":  download_pubmedqa,
    "medqa":     download_medqa,
}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Download free medical benchmark datasets")
    parser.add_argument(
        "--datasets", nargs="+",
        choices=list(DOWNLOADERS),
        default=list(DOWNLOADERS),
        help="Datasets to download (default: all)",
    )
    parser.add_argument("--n", type=int, default=None,
                        help="Max samples per dataset (overrides per-dataset default)")
    args = parser.parse_args()

    log(f"Output directory: {DATA_DIR}")
    for name in args.datasets:
        fn = DOWNLOADERS[name]
        try:
            if args.n is not None:
                import inspect
                if "n" in inspect.signature(fn).parameters:
                    fn(n=args.n)  # type: ignore[call-arg]
                else:
                    fn()  # type: ignore[call-arg]
            else:
                fn()  # type: ignore[call-arg]
        except Exception as exc:
            import traceback
            log(f"  x {name} FAILED: {exc}")
            traceback.print_exc()

    log("Done.")


if __name__ == "__main__":
    main()
