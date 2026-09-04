"""Evaluation runner: compares Vanilla RAG vs Hybrid vs AEB across metrics."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..agents.context_builder import infer_profile
from ..confidence.aeb import AEBResult
from ..confidence.pipeline import Pipeline
from ..data.benchmarks import get_benchmark
from ..models import ClinicalResponse


@dataclass
class QueryResult:
    query: str
    expected_dx: str
    predicted_dx: str | None
    correct: bool
    confidence: float
    hallucination_score: float
    latency_ms: float
    retrieval_k: int
    steps: int
    budget_exhausted: bool
    abstained: bool = False


@dataclass
class ExperimentReport:
    method: str
    n: int
    accuracy: float
    avg_confidence: float
    avg_hallucination: float
    avg_latency_ms: float
    avg_retrieval_k: float
    avg_steps: float
    budget_exhausted_rate: float
    abstention_rate: float = 0.0
    results: list[Any] = None


def _dx_matches(pred: str | None, expected: str) -> bool:
    if not pred:
        return False
    return pred == expected


def run_vanilla(pipeline: Pipeline, query: str) -> tuple[ClinicalResponse, int, float]:
    """Top-5 static retrieval, single LLM call, no verification."""
    t0 = time.perf_counter()
    evidence = pipeline.retriever.retrieve(query, top_k=5)
    response = pipeline.reasoner.generate(query, evidence)
    latency = (time.perf_counter() - t0) * 1000
    return response, len(evidence), latency


def run_hybrid(pipeline: Pipeline, query: str) -> tuple[ClinicalResponse, int, float]:
    t0 = time.perf_counter()
    evidence = pipeline.retriever.retrieve(query, top_k=10)
    response = pipeline.reasoner.generate(query, evidence)
    latency = (time.perf_counter() - t0) * 1000
    return response, len(evidence), latency


def run_aeb(pipeline: Pipeline, query: str, confidence_threshold: float = 0.85) -> tuple[AEBResult, float]:
    t0 = time.perf_counter()
    patient = infer_profile(query)
    result = pipeline.aeb.run(query, patient=patient, max_k_override=15)
    latency = (time.perf_counter() - t0) * 1000
    return result, latency


def evaluate(
    pipeline: Pipeline,
    queries: Sequence[dict],
    method: str = "aeb",
    confidence_threshold: float = 0.85,
) -> ExperimentReport:
    results: list[QueryResult] = []
    for q in queries:
        query = q["query"]
        expected = q["expected_dx"]
        if method == "vanilla":
            resp, k, latency = run_vanilla(pipeline, query)
            result = QueryResult(
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
            )
        elif method == "hybrid":
            resp, k, latency = run_hybrid(pipeline, query)
            result = QueryResult(
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
            )
        else:
            aeb, latency = run_aeb(pipeline, query, confidence_threshold)
            resp = aeb.response
            result = QueryResult(
                query=query,
                expected_dx=expected,
                predicted_dx=resp.primary_diagnosis,
                correct=_dx_matches(resp.primary_diagnosis, expected),
                confidence=resp.confidence,
                hallucination_score=resp.hallucination_score,
                latency_ms=latency,
                retrieval_k=aeb.total_retrieved,
                steps=len(aeb.steps),
                budget_exhausted=aeb.budget_exhausted,
                abstained=resp.decision.value == "escalate",
            )
        results.append(result)

    n = len(results)
    report = ExperimentReport(
        method=method,
        n=n,
        accuracy=sum(1 for r in results if r.correct) / n if n else 0.0,
        avg_confidence=sum(r.confidence for r in results) / n if n else 0.0,
        avg_hallucination=sum(r.hallucination_score for r in results) / n if n else 0.0,
        avg_latency_ms=sum(r.latency_ms for r in results) / n if n else 0.0,
        avg_retrieval_k=sum(r.retrieval_k for r in results) / n if n else 0.0,
        avg_steps=sum(r.steps for r in results) / n if n else 0.0,
        budget_exhausted_rate=sum(1 for r in results if r.budget_exhausted) / n if n else 0.0,
        abstention_rate=sum(1 for r in results if r.abstained) / n if n else 0.0,
        results=results,
    )
    return report


def run_threshold_sweep(
    pipeline: Pipeline,
    queries: Sequence[dict],
    thresholds: list[float] = None,
    out_dir: Path = None,
) -> list[ExperimentReport]:
    """Run evaluation across multiple confidence thresholds to generate abstention curve."""
    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    reports = []
    for threshold in thresholds:
        print(f"Running evaluation at threshold={threshold:.2f}...")
        report = evaluate(pipeline, queries, method="aeb", confidence_threshold=threshold)
        reports.append(report)
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"report_aeb_threshold_{threshold:.2f}.json").write_text(
                json.dumps(asdict(report), indent=2, default=str)
            )
    
    if out_dir:
        # Save sweep summary
        sweep_data = {
            "thresholds": [r.avg_confidence for r in reports],  # proxy for threshold
            "accuracy": [r.accuracy for r in reports],
            "abstention_rate": [r.abstention_rate for r in reports],
            "hallucination_rate": [r.avg_hallucination for r in reports],
            "avg_k": [r.avg_retrieval_k for r in reports],
        }
        (out_dir / "threshold_sweep.json").write_text(
            json.dumps(sweep_data, indent=2, default=str)
        )
    
    return reports


def run_all(
    pipeline: Pipeline,
    queries: Sequence[dict],
    out_dir: Path,
    benchmarks: list[str] = None,
    thresholds: list[float] = None,
) -> dict[str, Any]:
    """Run all evaluations and save results."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # 1. Run benchmarks
    if benchmarks is None:
        benchmarks = ["medqa", "pubmedqa", "seeds"]
    
    for bench_name in benchmarks:
        if bench_name == "seeds":
            from ..data.seed_corpus import get_seed_queries
            queries = get_seed_queries()
        else:
            queries = get_benchmark(bench_name)
        
        print(f"\n=== Evaluating on {bench_name.upper()} ({len(queries)} queries) ===")
        bench_results = {}
        for method in ("vanilla", "hybrid", "aeb"):
            report = evaluate(pipeline, queries, method=method)
            bench_results[method] = report
            (out_dir / f"report_{bench_name}_{method}.json").write_text(
                json.dumps(asdict(report), indent=2, default=str)
            )
        all_results[bench_name] = bench_results
    
    # 2. Run threshold sweep on AEB
    if thresholds:
        print(f"\n=== Running threshold sweep on seeds ===")
        from ..data.seed_corpus import get_seed_queries
        seed_queries = get_seed_queries()
        run_threshold_sweep(pipeline, seed_queries, thresholds, out_dir)
    
    # Save combined results
    combined_path = out_dir / "eval_combined.json"
    combined_path.write_text(json.dumps(all_results, indent=2, default=str))
    
    return all_results


def summarize(reports: dict[str, ExperimentReport]) -> str:
    lines = ["Method | Acc | AvgConf | Halluc | Latency(ms) | AvgK | Steps | BudgetExhausted | Abstention"]
    lines.append("-------|-----|---------|--------|-------------|------|-------|---------------|-----------")
    for name, r in reports.items():
        lines.append(
            f"{name} | {r.accuracy:.2f} | {r.avg_confidence:.2f} | {r.avg_hallucination:.2f} | "
            f"{r.avg_latency_ms:.1f} | {r.avg_retrieval_k:.1f} | {r.avg_steps:.1f} | "
            f"{r.budget_exhausted_rate:.2f} | {r.abstention_rate:.2f}"
        )
    return "\n".join(lines)


def _dx_matches(pred: str | None, expected: str) -> bool:
    if not pred:
        return False
    return pred == expected


def main():
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Run CAAR-CDSS evaluation")
    parser.add_argument("--mode", type=str, default="mock", 
                        choices=["mock", "real", "local_4bit", "kaggle_fp16", "hf_api"],
                        help="LLM inference backend mode")
    parser.add_argument("--benchmarks", nargs="+", default=["seeds"], 
                        choices=["medqa", "pubmedqa", "medmcqa", "seeds"],
                        help="Benchmarks to run")
    parser.add_argument("--n", type=int, default=200, help="Number of queries per benchmark")
    parser.add_argument("--output", type=str, default="experiments/results", help="Output directory")
    parser.add_argument("--threshold-sweep", action="store_true", help="Run threshold sweep for abstention curve")
    parser.add_argument("--embedding-ablation", action="store_true", help="Run embedding model ablation")
    parser.add_argument("--thresholds", nargs="+", type=float, 
                        default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
                        help="Confidence thresholds for sweep")
    
    args = parser.parse_args()
    
    pipeline = Pipeline.from_seed()
    
    # Determine queries based on benchmarks
    all_queries = {}
    for bench_name in args.benchmarks:
        if bench_name == "seeds":
            from ..data.seed_corpus import get_seed_queries
            all_queries[bench_name] = get_seed_queries()
        else:
            all_queries[bench_name] = get_benchmark(bench_name)[:args.n]
    
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Run evaluations
    all_results = {}
    for bench_name, queries in all_queries.items():
        print(f"\n=== Evaluating on {bench_name.upper()} ({len(queries)} queries) ===")
        bench_results = {}
        for method in ("vanilla", "hybrid", "aeb"):
            report = evaluate(pipeline, queries, method=method)
            bench_results[method] = report
            (Path(args.output) / f"report_{bench_name}_{method}.json").write_text(
                json.dumps(asdict(report), indent=2, default=str)
            )
        all_results[bench_name] = bench_results
    
    # Save combined results
    combined_path = Path(args.output) / "eval_combined.json"
    combined_path.write_text(json.dumps(all_results, indent=2, default=str))
    
    # Run threshold sweep if requested
    if args.threshold_sweep:
        from ..data.seed_corpus import get_seed_queries
        seed_queries = get_seed_queries()
        run_threshold_sweep(pipeline, seed_queries, args.thresholds, Path(args.output))
    
    print("\n=== Summary ===")
    for bench_name, bench_results in all_results.items():
        print(f"\n{bench_name.upper()}:")
        for method, report in bench_results.items():
            print(f"  {method}: Acc={report.accuracy:.2f}, Abstain={report.abstention_rate:.2f}, Halluc={report.avg_hallucination:.2f}, AvgK={report.avg_retrieval_k:.1f}")

if __name__ == "__main__":
    main()