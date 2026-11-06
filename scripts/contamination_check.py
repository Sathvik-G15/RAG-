#!/usr/bin/env python
"""Detect benchmark leakage into the retrieval KB.

Usage:
  python scripts/contamination_check.py --kb data/unified_clinical_kb.jsonl \
      --bench data/medqa_test.jsonl --question-key question --options-key options \
      --kb-text-key text --n 13 --threshold 0.2 --out reports/contamination.json
Exit code 1 if any benchmark item is flagged OR the KB declares a benchmark source.
"""
import argparse, json, re, sys
from pathlib import Path

BENCH_SOURCES = ("medqa", "medmcqa", "pubmedqa", "mmlu", "ddxplus", "usmle")


def toks(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def ngrams(t, n):
    return {hash(tuple(t[i:i + n])) for i in range(max(len(t) - n + 1, 0))}


def read(path):
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    if p.suffix == ".json":
        data = json.loads(txt)
        return data if isinstance(data, list) else list(data.values())
    return [json.loads(l) for l in txt.splitlines() if l.strip()]


def item_text(it, qk, ok):
    opts = it.get(ok, "")
    if isinstance(opts, dict):
        opts = " ".join(map(str, opts.values()))
    elif isinstance(opts, list):
        opts = " ".join(map(str, opts))
    return f"{it.get(qk, '')} {opts}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--bench", required=True, nargs="+")
    ap.add_argument("--question-key", default="question")
    ap.add_argument("--options-key", default="options")
    ap.add_argument("--kb-text-key", default="text")
    ap.add_argument("--kb-source-key", default="source")
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--out", default="contamination_report.json")
    a = ap.parse_args()

    kb = read(a.kb)
    bad_sources = sorted({str(r.get(a.kb_source_key, "")) for r in kb
                          if any(b in str(r.get(a.kb_source_key, "")).lower() for b in BENCH_SOURCES)})
    kb_grams = set()
    for r in kb:
        kb_grams |= ngrams(toks(str(r.get(a.kb_text_key, ""))), a.n)

    report = {"kb_records": len(kb), "n": a.n, "threshold": a.threshold,
              "benchmark_sources_in_kb": bad_sources, "benchmarks": {}}
    flagged_total = 0
    for path in a.bench:
        items = read(path)
        flagged = []
        for i, it in enumerate(items):
            t = toks(item_text(it, a.question_key, a.options_key))
            g = ngrams(t, a.n if len(t) >= a.n else max(len(t), 1))
            if not g:
                continue
            frac = len(g & kb_grams) / len(g) if len(t) >= a.n else 0.0
            if frac >= a.threshold:
                flagged.append({"index": i, "overlap": round(frac, 3)})
        flagged_total += len(flagged)
        report["benchmarks"][path] = {"items": len(items), "flagged": len(flagged),
                                      "flagged_rate": len(flagged) / max(len(items), 1), "examples": flagged[:20]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "benchmarks"}, indent=2))
    for p, r in report["benchmarks"].items():
        print(f"{p}: {r['flagged']}/{r['items']} flagged")
    sys.exit(1 if flagged_total or bad_sources else 0)


if __name__ == "__main__":
    main()
