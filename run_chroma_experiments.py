"""Run experiments comparing Vanilla vs Hybrid vs AEB on Chroma pipeline."""

from __future__ import annotations

import time
from pathlib import Path

from src.confidence.chroma_pipeline import ChromaPipeline
from src.data.seed_corpus import SEED_PATIENT_QUERIES
from src.experiments.runner import run_all


def run_chroma_experiments():
    """Run all three experiment modes on Chroma pipeline."""
    print("=== Initializing Chroma Pipeline ===")
    pipeline = ChromaPipeline.from_chroma()
    
    queries = SEED_PATIENT_QUERIES
    print(f"\n=== Running experiments on {len(queries)} clinical queries ===")
    
    # Use the existing experiment runner but with our Chroma pipeline
    # We need to adapt it to work with ChromaPipeline
    from src.experiments.runner import ExperimentReport, QueryResult, _dx_matches
    
    def run_vanilla_chroma(query: str):
        t0 = time.perf_counter()
        evidence = pipeline.retriever.retrieve(query, top_k=5)
        response = pipeline.reasoner.generate(query, evidence)
        latency = (time.perf_counter() - t0) * 1000
        return response, len(evidence), latency
    
    def run_hybrid_chroma(query: str):
        t0 = time.perf_counter()
        evidence = pipeline.retriever.retrieve(query, top_k=10)
        response = pipeline.reasoner.generate(query, evidence)
        latency = (time.perf_counter() - t0) * 1000
        return response, len(evidence), latency
    
    def run_aeb_chroma(query: str):
        t0 = time.perf_counter()
        from src.agents.context_builder import infer_profile
        patient = infer_profile(query)
        result = pipeline.aeb.run(query, patient=patient)
        latency = (time.perf_counter() - t0) * 1000
        return result, latency
    
    results_by_method = {}
    
    for method_name, run_fn in [("vanilla", run_vanilla_chroma), 
                                 ("hybrid", run_hybrid_chroma), 
                                 ("aeb", run_aeb_chroma)]:
        print(f"\n--- {method_name.upper()} ---")
        method_results = []
        
        for q in queries:
            query = q["query"]
            expected = q["expected_dx"]
            
            if method_name == "aeb":
                result, latency = run_fn(query)
                resp = result.response
                method_results.append(QueryResult(
                    query=query,
                    expected_dx=expected,
                    predicted_dx=resp.primary_diagnosis,
                    correct=_dx_matches(resp.primary_diagnosis, expected),
                    confidence=resp.confidence,
                    hallucination_score=resp.hallucination_score,
                    latency_ms=latency,
                    retrieval_k=result.total_retrieved,
                    steps=len(result.steps),
                    budget_exhausted=result.budget_exhausted,
                ))
            else:
                resp, k, latency = run_fn(query)
                method_results.append(QueryResult(
                    query=query,
                    expected_dx=expected,
                    predicted_dx=resp.primary_diagnosis,
                    correct=_dx_matches(resp.primary_diagnosis, expected),
                    confidence=resp.confidence,
                    hallucination_score=0.0,
                    latency_ms=latency,
                    retrieval_k=k,
                    steps=1,
                    budget_exhausted=False,
                ))
        
        n = len(method_results)
        report = ExperimentReport(
            method=method_name,
            n=n,
            accuracy=sum(1 for r in method_results if r.correct) / n,
            avg_confidence=sum(r.confidence for r in method_results) / n,
            avg_hallucination=sum(r.hallucination_score for r in method_results) / n,
            avg_latency_ms=sum(r.latency_ms for r in method_results) / n,
            avg_retrieval_k=sum(r.retrieval_k for r in method_results) / n,
            avg_steps=sum(r.steps for r in method_results) / n,
            budget_exhausted_rate=sum(1 for r in method_results if r.budget_exhausted) / n,
            results=method_results,
        )
        results_by_method[method_name] = report
        
        print(f"  Accuracy: {report.accuracy:.2%}")
        print(f"  Avg Confidence: {report.avg_confidence:.3f}")
        print(f"  Avg Hallucination: {report.avg_hallucination:.3f}")
        print(f"  Avg Latency: {report.avg_latency_ms:.1f}ms")
        print(f"  Avg Retrieval K: {report.avg_retrieval_k:.1f}")
        print(f"  Avg Steps: {report.avg_steps:.1f}")
        print(f"  Budget Exhausted: {report.budget_exhausted_rate:.2%}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Method':<10} | {'Acc':<6} | {'Conf':<6} | {'Halluc':<7} | {'Lat(ms)':<8} | {'AvgK':<6} | {'Steps':<6} | {'Budget%':<8}")
    print("-"*80)
    for name, r in results_by_method.items():
        print(f"{name:<10} | {r.accuracy:.2%}   | {r.avg_confidence:.3f}  | {r.avg_hallucination:.3f}    | {r.avg_latency_ms:>6.1f}   | {r.avg_retrieval_k:>4.1f}  | {r.avg_steps:>4.1f}  | {r.budget_exhausted_rate:.2%}")
    
    # Save reports
    import json
    from dataclasses import asdict
    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    for method, report in results_by_method.items():
        (out_dir / f"report_{method}_chroma.json").write_text(
            json.dumps(asdict(report), indent=2, default=str)
        )
    
    return results_by_method


if __name__ == "__main__":
    run_chroma_experiments()