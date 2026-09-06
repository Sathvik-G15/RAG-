"""Command-line interface for CAAR-CDSS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.confidence.pipeline import Pipeline
from src.data.seed_corpus import get_seed_queries
from src.experiments.runner import run_all, summarize
from src.models import Visit, Vitals


def build_pipeline(prefer_real: bool) -> Pipeline:
    return Pipeline.from_seed(prefer_real=prefer_real)


def cmd_analyze(args: argparse.Namespace) -> None:
    pipeline = build_pipeline(prefer_real=args.real)
    visit = None
    if args.vitals:
        v = json.loads(args.vitals)
        visit = Visit(vitals=Vitals(**v))
    aeb, resp = pipeline.analyze(args.query, visit=visit)
    print("=== AEB Result ===")
    print(f"Primary: {resp.primary_diagnosis}")
    print(f"Confidence: {resp.confidence:.3f}")
    print(f"Hallucination: {resp.hallucination_score:.3f}")
    print(f"Decision: {resp.decision.value}  Risk: {resp.risk_level.value}")
    print(f"k used: {resp.retrieval_k_used}  steps: {len(aeb.steps)}")
    if resp.escalated_reason:
        print(f"Escalation: {resp.escalated_reason}")
    print("\nConfidence curve:", [round(c, 3) for c in aeb.confidence_curve])
    print("\nTop differential:")
    for d in resp.differential[:3]:
        print(f"  {d.diagnosis}: {d.probability:.3f} (sources: {', '.join(d.sources[:2])})")
    print("\n" + resp.safety_disclaimer)


def cmd_experiments(args: argparse.Namespace) -> None:
    pipeline = build_pipeline(prefer_real=args.real)
    queries = get_seed_queries()
    out_dir = Path(args.out)
    reports = run_all(pipeline, queries, out_dir)
    print(summarize(reports))
    print(f"\nReports written to {out_dir.resolve()}")


def cmd_ingest(args: argparse.Namespace) -> None:
    from src.config import get_app_config
    from src.data.guidelines_loader import load_and_chunk_guidelines
    from src.kb.embedder import make_embedder
    from src.kb.vector_store import make_vector_store

    print(f"Ingesting corpus '{args.corpus}' (domain='{args.domain}', limit={args.limit})...")
    chunks = load_and_chunk_guidelines(domain=args.domain, limit=args.limit)
    if not chunks:
        print("No chunks generated.")
        return

    print(f"Generated {len(chunks)} chunks from corpus. Initializing embedder & vector store...")
    cfg = get_app_config()
    embedder = make_embedder(model_name=cfg.embedding.model_name, prefer_real=args.real)
    store = make_vector_store(backend="chroma", persist_dir=args.chroma_dir)

    print(f"Computing embeddings for {len(chunks)} chunks...")
    texts = [c.text for c in chunks]
    embs = embedder.embed(texts)

    print(f"Upserting {len(chunks)} chunks into ChromaDB at {args.chroma_dir}...")
    store.add(chunks, embs)
    print(f"Successfully ingested {len(chunks)} chunks into ChromaDB at {args.chroma_dir}.")


def cmd_build_kb(args: argparse.Namespace) -> None:
    pipeline = build_pipeline(prefer_real=args.real)
    print(f"Knowledge base built with {len(pipeline.kb.store)} chunks.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="caar-cdss", description="CAAR-CDSS CLI")
    parser.add_argument("--real", action="store_true", help="Try to use real LLM/embeddings (GPU/internet required)")
    parser.add_argument("--mode", type=str, default="mock", choices=["mock", "real", "local_4bit", "kaggle_fp16", "hf_api"],
                        help="LLM inference backend mode")
    sub = parser.add_subparsers(dest="command", required=True)

    pa = sub.add_parser("analyze", help="Run AEB pipeline on a query")
    pa.add_argument("query")
    pa.add_argument("--vitals", help="JSON dict for vitals")
    pa.set_defaults(func=cmd_analyze)

    pq = sub.add_parser("query", help="Alias for analyze")
    pq.add_argument("query")
    pq.add_argument("--vitals", help="JSON dict for vitals")
    pq.set_defaults(func=cmd_analyze)

    pi = sub.add_parser("ingest", help="Ingest guideline corpus into ChromaDB")
    pi.add_argument("--corpus", default="epfl-llm/guidelines", help="HuggingFace corpus name")
    pi.add_argument("--domain", default="infectious_disease", help="Domain filter (e.g. infectious_disease, all)")
    pi.add_argument("--limit", type=int, default=None, help="Max raw documents to load")
    pi.add_argument("--chroma-dir", default="data/chroma_db", help="ChromaDB persistence directory")
    pi.set_defaults(func=cmd_ingest)

    pe = sub.add_parser("experiments", help="Run evaluation suite")
    pe.add_argument("--out", default="experiments/results")
    pe.set_defaults(func=cmd_experiments)

    pk = sub.add_parser("build-kb", help="Build knowledge base from seed corpus")
    pk.set_defaults(func=cmd_build_kb)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

