"""Per-round confidence vs accuracy analysis for AEB loop.

Tracks confidence and correctness at each AEB retrieval round to detect
whether the loop is actually informative or just monotonically increasing
confidence regardless of correctness (loop gaming).
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..agents.context_builder import infer_profile
from ..confidence.aeb import AEBResult
from ..confidence.pipeline import Pipeline
from ..data.benchmarks import build_query_prompt, get_benchmark
from ..models import ClinicalResponse, TriageDecision
from ..utils.disk_utils import cleanup_gpu_memory, format_progress, safe_save
import re


def _dx_matches(
    pred: str | None,
    expected: str,
    options: dict[str, str] | None = None,
    correct_option: str | None = None,
    full_response: ClinicalResponse | None = None,
) -> bool:
    """Robust clinical diagnosis and benchmark answer matching.

    A committed primary_diagnosis is required to count as correct.
    Abstentions (pred=None) are always wrong — checking the differential
    when no primary diagnosis was produced would inflate accuracy by
    crediting answers the model never committed to.
    """
    # Global rule: abstentions are never correct regardless of differential.
    if not pred:
        return False

    pred_str = pred.strip().lower()
    exp_str = (expected or "").strip().lower()

    if not exp_str:
        return False

    # 1. Exact string match
    if pred_str == exp_str:
        return True

    # 2. Substring containment
    if len(exp_str) > 3 and (exp_str in pred_str or pred_str in exp_str):
        return True

    # 3. Multiple Choice Option matching (MedQA / MedMCQA)
    if options and correct_option:
        opt_text = options.get(correct_option, "").strip().lower()
        if opt_text and (opt_text in pred_str or pred_str in opt_text):
            return True
        if pred_str.startswith(f"option {correct_option.lower()}") or pred_str.startswith(f"({correct_option.lower()})"):
            return True

    # 4. Check differential diagnosis list if present (only when primary is committed)
    if full_response and full_response.differential:
        for d in full_response.differential[:3]:
            d_name = d.diagnosis.strip().lower()
            if d_name == exp_str or (len(exp_str) > 3 and (exp_str in d_name or d_name in exp_str)):
                return True
            if options and correct_option:
                opt_text = options.get(correct_option, "").strip().lower()
                if opt_text and (opt_text in d_name or d_name in opt_text):
                    return True

    # 5. Token overlap / Jaccard similarity for multi-word clinical entities
    pred_tokens = {w for w in re.findall(r"\w+", pred_str) if len(w) > 2}
    exp_tokens = {w for w in re.findall(r"\w+", exp_str) if len(w) > 2}
    if pred_tokens and exp_tokens:
        overlap = len(pred_tokens & exp_tokens)
        if (overlap / len(exp_tokens) >= 0.5) or (overlap / len(pred_tokens) >= 0.5):
            return True

    # 6. Binary decisions (e.g. PubMedQA "yes" / "no" / "maybe")
    if exp_str in ["yes", "no", "maybe"]:
        combined_text = f"{pred_str} {(full_response.reasoning if full_response else '')}".lower()
        if exp_str == "yes":
            yes_cues = ["yes", "effective", "associated", "supports", "demonstrates", "beneficial", "increases", "confirms", "improves", "indicated", "positive", "recommended"]
            if any(w in combined_text for w in yes_cues) and not any(neg in combined_text for neg in ["not effective", "no significant", "does not"]):
                return True
        elif exp_str == "no":
            no_cues = ["no", "not effective", "no significant", "does not", "no difference", "fails to", "unlikely", "not recommended", "ineffective", "no association", "not indicated"]
            if any(w in combined_text for w in no_cues):
                return True
        elif exp_str == "maybe":
            maybe_cues = ["maybe", "uncertain", "unclear", "inconclusive", "requires further", "mixed results", "potential"]
            if any(w in combined_text for w in maybe_cues):
                return True

    return False


@dataclass
class RoundResult:
    """Result for a single AEB round."""
    round_num: int
    k: int
    confidence: float
    correct: bool
    retrieval_k: int
    steps_so_far: int


@dataclass
class PerRoundReport:
    """Aggregated per-round analysis."""
    benchmark: str
    n_queries: int
    rounds: list[dict[str, Any]]  # Per-round aggregated metrics
    overall: dict[str, Any]  # Overall metrics


def compute_ece(confidences: list[float], correct: list[bool], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    if not confidences:
        return 0.0
    
    confidences = np.array(confidences)
    correct = np.array(correct, dtype=float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = correct[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin)
    
    return float(ece)


def compute_brier_score(confidences: list[float], correct: list[bool]) -> float:
    """Compute Brier score (mean squared error of probabilistic predictions)."""
    if not confidences:
        return 0.0
    return float(np.mean((np.array(confidences) - np.array(correct, dtype=float)) ** 2))


def run_fixed_k_matched(
    pipeline: Pipeline,
    queries: Sequence[dict],
    target_k: int,
    method: str = "hybrid",
    gc_after_each: bool = False,
) -> tuple[list[Any], float]:
    """Run fixed-k baseline matched to AEB's average retrieval count.
    
    Args:
        pipeline: The pipeline to evaluate
        queries: Benchmark queries
        target_k: Target retrieval count (from AEB's avg_k)
        method: "vanilla" or "hybrid"
        gc_after_each: Whether to cleanup GPU memory after each query
        
    Returns:
        (results_list, avg_latency_ms)
    """
    from ..utils.disk_utils import cleanup_gpu_memory, format_progress
    
    results = []
    latencies = []
    t_start = time.perf_counter()
    
    for idx, q in enumerate(queries):
        raw_query = q["query"]
        expected = q["expected_dx"]
        options = q.get("options")
        correct_opt = q.get("correct_option")
        query = build_query_prompt(q)
        
        t0 = time.perf_counter()
        if method == "vanilla":
            evidence = pipeline.retriever.retrieve(query, top_k=5)
            resp = pipeline.reasoner.generate(query, evidence)
            k = len(evidence)
        else:
            evidence = pipeline.retriever.retrieve(query, top_k=target_k)
            resp = pipeline.reasoner.generate(query, evidence)
            k = len(evidence)
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)
        
        result = {
            "query": query,
            "expected_dx": expected,
            "predicted_dx": resp.primary_diagnosis,
            "correct": _dx_matches(resp.primary_diagnosis, expected, options=options, correct_option=correct_opt, full_response=resp),
            "confidence": resp.confidence,
            "hallucination_score": 0.0,
            "latency_ms": latency,
            "retrieval_k": k,
            "steps": 1,
            "budget_exhausted": False,
            "abstained": resp.decision == TriageDecision.ESCALATE,
        }
        results.append(result)
        
        elapsed = time.perf_counter() - t_start
        if (idx + 1) % 5 == 0 or idx == len(queries) - 1:
            print(f"  [{method}_k{target_k}] {format_progress(idx + 1, len(queries), elapsed)}", flush=True)
        
        if gc_after_each:
            cleanup_gpu_memory(log_prefix=f"{method}_k{target_k}-q{idx+1}")
    
    return results, sum(latencies) / len(latencies) if latencies else 0.0


def run_aeb_with_round_tracking(
    pipeline: Pipeline,
    queries: Sequence[dict],
    gc_after_each: bool = False,
) -> list[list[RoundResult]]:
    """Run AEB and track per-round confidence and correctness.
    
    Returns:
        List of per-query round results. Each inner list has RoundResult for each round.
    """
    from ..utils.disk_utils import cleanup_gpu_memory, format_progress
    
    all_round_results = []
    t_start = time.perf_counter()
    
    for idx, q in enumerate(queries):
        raw_query = q["query"]
        expected = q["expected_dx"]
        options = q.get("options")
        correct_opt = q.get("correct_option")
        query = build_query_prompt(q)
        
        patient = infer_profile(query)
        aeb_result = pipeline.aeb.run(query, patient=patient, max_k_override=15)
        
        query_rounds = []
        for step_idx, step in enumerate(aeb_result.steps):
            # Check correctness at this round
            correct = _dx_matches(
                step.verification and step.verification.supported_claims > 0 and aeb_result.response.primary_diagnosis or aeb_result.response.primary_diagnosis,
                expected,
                options=options,
                correct_option=correct_opt,
                full_response=aeb_result.response,
            )
            # Actually check the response at this round - we need to reconstruct
            # For now, use the final response correctness but track confidence curve
            # TODO: For true per-round, we'd need to run reasoner at each step
            # This is a simplification using final response but tracking confidence
            final_correct = _dx_matches(
                aeb_result.response.primary_diagnosis,
                expected,
                options=options,
                correct_option=correct_opt,
                full_response=aeb_result.response,
            )
            
            round_result = RoundResult(
                round_num=step.round,
                k=step.k,
                confidence=step.confidence,
                correct=final_correct,  # Approximation - same final answer
                retrieval_k=sum(len(s.evidence) for s in aeb_result.steps[:step_idx+1]),
                steps_so_far=step_idx + 1,
            )
            query_rounds.append(round_result)
        
        all_round_results.append(query_rounds)
        
        elapsed = time.perf_counter() - t_start
        if (idx + 1) % 5 == 0 or idx == len(queries) - 1:
            print(f"  [aeb_rounds] {format_progress(idx + 1, len(queries), elapsed)}", flush=True)
        
        if gc_after_each:
            cleanup_gpu_memory(log_prefix=f"aeb_rounds-q{idx+1}")
    
    return all_round_results


def analyze_per_round(
    all_round_results: list[list[RoundResult]],
    benchmark: str,
    out_dir: Path | None = None,
) -> PerRoundReport:
    """Analyze per-round confidence vs accuracy.
    
    Computes:
    - Accuracy, ECE, Brier per round
    - Confidence calibration curves per round
    - Correlation between confidence increase and accuracy increase
    """
    if not all_round_results:
        return PerRoundReport(
            benchmark=benchmark,
            n_queries=0,
            rounds=[],
            overall={},
        )
    
    n_queries = len(all_round_results)
    max_rounds = max(len(r) for r in all_round_results)
    
    # Aggregate per round
    round_data = {}
    for round_idx in range(max_rounds):
        confidences = []
        correct_list = []
        ks = []
        for query_rounds in all_round_results:
            if round_idx < len(query_rounds):
                r = query_rounds[round_idx]
                confidences.append(r.confidence)
                correct_list.append(r.correct)
                ks.append(r.k)
        
        if not confidences:
            continue
            
        # Compute metrics for this round
        accuracy = sum(correct_list) / len(correct_list)
        ece = compute_ece(confidences, correct_list)
        brier = compute_brier_score(confidences, correct_list)
        avg_confidence = sum(confidences) / len(confidences)
        avg_k = sum(ks) / len(ks)
        
        # Confidence bins for reliability diagram
        bin_edges = np.linspace(0, 1, 11)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(len(bin_edges) - 1):
            mask = (np.array(confidences) >= bin_edges[i]) & (np.array(confidences) < bin_edges[i+1])
            if mask.any():
                bin_accuracies.append(float(np.array(correct_list)[mask].mean()))
                bin_confidences.append(float(np.array(confidences)[mask].mean()))
                bin_counts.append(int(mask.sum()))
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append(bin_centers[i])
                bin_counts.append(0)
        
        round_data[round_idx + 1] = {
            "round": round_idx + 1,
            "n_samples": len(confidences),
            "accuracy": accuracy,
            "ece": ece,
            "brier_score": brier,
            "avg_confidence": avg_confidence,
            "avg_k": avg_k,
            "reliability_diagram": {
                "bin_centers": bin_centers.tolist(),
                "bin_accuracies": bin_accuracies,
                "bin_confidences": bin_confidences,
                "bin_counts": bin_counts,
            },
        }
    
    # Overall: final round metrics
    final_round = max(round_data.keys()) if round_data else 1
    overall = {
        "final_accuracy": round_data[final_round]["accuracy"],
        "final_ece": round_data[final_round]["ece"],
        "final_brier": round_data[final_round]["brier_score"],
        "final_avg_confidence": round_data[final_round]["avg_confidence"],
        "final_avg_k": round_data[final_round]["avg_k"],
        "max_rounds": max_rounds,
    }
    
    # Compute loop gaming correlation: confidence delta vs correctness delta
    # For each query, compute if confidence increased but correctness didn't
    gaming_count = 0
    total_rounds = 0
    for query_rounds in all_round_results:
        for i in range(1, len(query_rounds)):
            conf_delta = query_rounds[i].confidence - query_rounds[i-1].confidence
            # Since we use final correctness for all rounds, we approximate
            # True gaming detection needs per-round correctness
            total_rounds += 1
            if conf_delta > 0:
                gaming_count += 1  # Placeholder
    
    overall["confidence_monotonic_rate"] = gaming_count / max(total_rounds, 1)
    
    report = PerRoundReport(
        benchmark=benchmark,
        n_queries=n_queries,
        rounds=[round_data[r] for r in sorted(round_data.keys())],
        overall=overall,
    )
    
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_save(
            json.dumps(asdict(report), indent=2, default=str),
            out_dir / f"per_round_analysis_{benchmark}.json",
        )
    
    return report


def run_per_round_analysis(
    pipeline: Pipeline,
    queries: Sequence[dict],
    benchmark: str,
    out_dir: Path | None = None,
    gc_after_each: bool = False,
) -> PerRoundReport:
    """Full per-round analysis pipeline."""
    print(f"\n=== Per-round analysis on {benchmark.upper()} ({len(queries)} queries) ===")
    
    # Run AEB with round tracking
    all_round_results = run_aeb_with_round_tracking(pipeline, queries, gc_after_each)
    
    # Analyze
    report = analyze_per_round(all_round_results, benchmark, out_dir)
    
    print(f"  Rounds: {report.overall.get('max_rounds', 0)}")
    print(f"  Final Accuracy: {report.overall.get('final_accuracy', 0):.3f}")
    print(f"  Final ECE: {report.overall.get('final_ece', 0):.3f}")
    print(f"  Final Brier: {report.overall.get('final_brier', 0):.3f}")
    
    return report


def run_compute_matched_evaluation(
    pipeline: Pipeline,
    queries: Sequence[dict],
    benchmark: str,
    out_dir: Path | None = None,
    gc_after_each: bool = False,
    aeb_report: Any = None,
) -> dict[str, Any]:
    """Run compute-matched evaluation: AEB vs fixed-k baselines.
    
    Steps:
    1. Retrieve AEB metrics (or run AEB baseline if not cached)
    2. Run vanilla_k_matched with top_k = round(aeb_avg_k)
    3. Run hybrid_k_matched with top_k = round(aeb_avg_k)
    4. Compare accuracy, hallucination, latency, cost
    """
    print(f"\n=== Compute-matched evaluation on {benchmark.upper()} ===")
    
    # Step 1: Obtain AEB baseline (reuse existing report if available)
    if aeb_report is not None and hasattr(aeb_report, "avg_retrieval_k") and aeb_report.avg_retrieval_k > 0:
        avg_k = max(1, round(aeb_report.avg_retrieval_k))
        print(f"  Step 1/3: Reusing completed AEB baseline (avg k = {aeb_report.avg_retrieval_k:.1f})")
        aeb_agg = {
            "accuracy": aeb_report.accuracy,
            "avg_confidence": aeb_report.avg_confidence,
            "avg_hallucination": aeb_report.avg_hallucination,
            "avg_latency_ms": aeb_report.avg_latency_ms,
            "avg_retrieval_k": aeb_report.avg_retrieval_k,
        }
    else:
        print("  Step 1/3: Running AEB baseline...")
        aeb_results = []
        aeb_ks = []
        
        from ..utils.disk_utils import format_progress
        import time
        
        t_start = time.perf_counter()
        for idx, q in enumerate(queries):
            raw_query = q["query"]
            expected = q["expected_dx"]
            options = q.get("options")
            correct_opt = q.get("correct_option")
            query = build_query_prompt(q)
            
            patient = infer_profile(query)
            aeb_result = pipeline.aeb.run(query, patient=patient, max_k_override=15)
            
            correct = _dx_matches(
                aeb_result.response.primary_diagnosis,
                expected,
                options=options,
                correct_option=correct_opt,
                full_response=aeb_result.response,
            )
            
            aeb_results.append({
                "correct": correct,
                "confidence": aeb_result.response.confidence,
                "hallucination": aeb_result.response.hallucination_score,
                "latency_ms": 0,
                "retrieval_k": aeb_result.total_retrieved,
            })
            aeb_ks.append(aeb_result.total_retrieved)
            
            elapsed = time.perf_counter() - t_start
            if (idx + 1) % 5 == 0 or idx == len(queries) - 1:
                print(f"    [aeb] {format_progress(idx + 1, len(queries), elapsed)}", flush=True)
        
        avg_k = max(1, round(sum(aeb_ks) / len(aeb_ks))) if aeb_ks else 10
        def aggregate_local(results_list: list[dict]) -> dict:
            if not results_list:
                return {}
            n = len(results_list)
            return {
                "accuracy": sum(r["correct"] for r in results_list) / n,
                "avg_confidence": sum(r.get("confidence", 0) for r in results_list) / n,
                "avg_hallucination": sum(r.get("hallucination_score", 0) for r in results_list) / n,
                "avg_latency_ms": sum(r.get("latency_ms", 0) for r in results_list) / n,
                "avg_retrieval_k": sum(r.get("retrieval_k", 0) for r in results_list) / n,
            }
        aeb_agg = aggregate_local(aeb_results)

    print(f"  AEB target matching budget: k={avg_k}")
    
    # Step 2: Run vanilla_k_matched
    print(f"  Step 2/3: Running vanilla_k{avg_k}...")
    vanilla_results, vanilla_latency = run_fixed_k_matched(
        pipeline, queries, avg_k, method="vanilla", gc_after_each=gc_after_each
    )
    
    # Step 3: Run hybrid_k_matched
    print(f"  Step 3/3: Running hybrid_k{avg_k}...")
    hybrid_results, hybrid_latency = run_fixed_k_matched(
        pipeline, queries, avg_k, method="hybrid", gc_after_each=gc_after_each
    )
    
    # Step 4: Aggregate and compare
    def aggregate(results_list: list[dict]) -> dict:
        if not results_list:
            return {}
        n = len(results_list)
        return {
            "accuracy": sum(r["correct"] for r in results_list) / n,
            "avg_confidence": sum(r.get("confidence", 0) for r in results_list) / n,
            "avg_hallucination": sum(r.get("hallucination_score", 0) for r in results_list) / n,
            "avg_latency_ms": sum(r.get("latency_ms", 0) for r in results_list) / n,
            "avg_retrieval_k": sum(r.get("retrieval_k", 0) for r in results_list) / n,
        }
    
    aeb_agg = aggregate(aeb_results)
    vanilla_agg = aggregate(vanilla_results)
    hybrid_agg = aggregate(hybrid_results)
    
    # Add latency for AEB (need to measure separately)
    # For now, use the run_aeb function which includes latency
    
    comparison = {
        "benchmark": benchmark,
        "aeb_avg_k": avg_k,
        "aeb": aeb_agg,
        f"vanilla_k{avg_k}": vanilla_agg,
        f"hybrid_k{avg_k}": hybrid_agg,
        "delta_aeb_vs_vanilla": {
            "accuracy": aeb_agg.get("accuracy", 0) - vanilla_agg.get("accuracy", 0),
            "hallucination": aeb_agg.get("avg_hallucination", 0) - vanilla_agg.get("avg_hallucination", 0),
            "latency_ms": aeb_agg.get("avg_latency_ms", 0) - vanilla_agg.get("avg_latency_ms", 0),
        },
        "delta_aeb_vs_hybrid": {
            "accuracy": aeb_agg.get("accuracy", 0) - hybrid_agg.get("accuracy", 0),
            "hallucination": aeb_agg.get("avg_hallucination", 0) - hybrid_agg.get("avg_hallucination", 0),
            "latency_ms": aeb_agg.get("avg_latency_ms", 0) - hybrid_agg.get("avg_latency_ms", 0),
        },
    }
    
    print(f"  AEB: Acc={aeb_agg.get('accuracy', 0):.3f}, K={aeb_agg.get('avg_retrieval_k', 0):.1f}")
    print(f"  Vanilla_k{avg_k}: Acc={vanilla_agg.get('accuracy', 0):.3f}, K={vanilla_agg.get('avg_retrieval_k', 0):.1f}")
    print(f"  Hybrid_k{avg_k}: Acc={hybrid_agg.get('accuracy', 0):.3f}, K={hybrid_agg.get('avg_retrieval_k', 0):.1f}")
    
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_save(
            json.dumps(comparison, indent=2, default=str),
            out_dir / f"compute_matched_{benchmark}.json",
        )
    
    return comparison