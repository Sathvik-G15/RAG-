"""Evaluation harness with custom metrics (no RAGAS dependency)."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.confidence.pipeline import Pipeline
from src.data.ingest_external import ExternalDataIngestor, IngestionConfig


@dataclass
class EvalResult:
    query: str
    expected_answer: str | None
    generated_answer: str
    retrieved_contexts: list[str]
    confidence: float
    hallucination_score: float
    abstained: bool

    # Custom metrics
    citation_accuracy: float | None = None
    abstention_correct: bool | None = None
    answer_correct: bool | None = None


class EvaluationHarness:
    def __init__(self, pipeline: Pipeline, ingestor: ExternalDataIngestor):
        self.pipeline = pipeline
        self.ingestor = ingestor

    def run_on_benchmark(self, benchmark_name: str, max_samples: int = 50) -> list[EvalResult]:
        """Run evaluation on a benchmark dataset."""
        data = self.ingestor.load_benchmark(benchmark_name)
        if not data:
            print(f"Benchmark {benchmark_name} not found locally")
            return []

        results = []
        for i, item in enumerate(data[:max_samples]):
            query, expected = self._extract_query_answer(item, benchmark_name)
            if not query:
                continue

            print(f"[{i+1}/{min(max_samples, len(data))}] Evaluating: {query[:80]}...")

            try:
                aeb_result, response = self.pipeline.analyze(query)

                retrieved_contexts = [c.text for c in response.evidence]

                result = EvalResult(
                    query=query,
                    expected_answer=expected,
                    generated_answer=response.reasoning,
                    retrieved_contexts=retrieved_contexts,
                    confidence=response.confidence,
                    hallucination_score=response.hallucination_score,
                    abstained=response.decision.value == "escalate",
                )

                # Compute citation accuracy
                result.citation_accuracy = self._compute_citation_accuracy(
                    response.reasoning, retrieved_contexts
                )

                # Compute answer correctness (simple keyword match)
                if expected:
                    result.answer_correct = self._check_answer_correct(
                        response.reasoning, expected
                    )
                    result.abstention_correct = self._check_abstention_correct(
                        result.abstained, result.answer_correct
                    )

                results.append(result)

            except Exception as e:
                print(f"  Error: {e}")
                continue

        return results

    def _extract_query_answer(self, item: dict, benchmark: str) -> tuple[str, str | None]:
        """Extract query and expected answer from benchmark item."""
        if benchmark == "medqa":
            question = item.get("question", "")
            options = item.get("options", {})
            answer_idx = item.get("answer_idx", 0)
            answer = options.get(str(answer_idx), "") if options else ""
            return question, answer

        elif benchmark == "pubmedqa":
            question = item.get("question", "")
            long_answer = item.get("long_answer", "")
            final_decision = item.get("final_decision", "")
            return question, f"{final_decision}: {long_answer}"

        elif benchmark == "medmcqa":
            question = item.get("question", "")
            options = [item.get(f"opt{i}", "") for i in range(1, 5)]
            answer_idx = item.get("cop", 0)
            answer = options[answer_idx] if answer_idx < len(options) else ""
            return question, answer

        return "", None

    def _compute_citation_accuracy(self, reasoning: str, contexts: list[str]) -> float:
        """Citation accuracy: fraction of claims supported by contexts."""
        sentences = re.split(r"(?<=[.!?])\s+", reasoning.strip())
        if not sentences:
            return 0.0

        supported = 0
        for sent in sentences:
            if len(sent) < 20:
                continue
            sent_lower = sent.lower()
            sent_tokens = set(sent_lower.split())
            for ctx in contexts:
                ctx_lower = ctx.lower()
                ctx_tokens = set(ctx_lower.split())
                overlap = len(sent_tokens & ctx_tokens)
                if overlap >= 3:
                    supported += 1
                    break

        return supported / len(sentences) if sentences else 0.0

    def _check_answer_correct(self, generated: str, expected: str) -> bool:
        """Simple answer correctness check via keyword overlap."""
        if not expected or not generated:
            return False

        gen_lower = generated.lower()
        exp_lower = expected.lower()

        # Extract key medical terms from expected
        exp_tokens = set(re.findall(r"\b[a-z]{4,}\b", exp_lower))
        exp_tokens = {t for t in exp_tokens if t not in {
            "patient", "presents", "history", "diagnosis", "treatment", "management",
            "clinical", "symptoms", "signs", "therapy", "medical", "disease"
        }}

        if not exp_tokens:
            return True

        matches = sum(1 for t in exp_tokens if t in gen_lower)
        return matches / len(exp_tokens) >= 0.3

    def _check_abstention_correct(self, abstained: bool, answer_correct: bool) -> bool:
        """Abstention is correct if we abstained when answer would be wrong, or answered when correct."""
        if abstained:
            return not answer_correct  # Correct to abstain when answer would be wrong
        else:
            return answer_correct  # Correct to answer when answer is right

    def aggregate_results(self, results: list[EvalResult]) -> dict[str, Any]:
        """Aggregate evaluation results."""
        if not results:
            return {}

        n = len(results)

        return {
            "n_samples": n,
            "avg_confidence": sum(r.confidence for r in results) / n,
            "avg_hallucination": sum(r.hallucination_score for r in results) / n,
            "abstention_rate": sum(1 for r in results if r.abstained) / n,
            "avg_citation_accuracy": sum(r.citation_accuracy or 0 for r in results) / n,
            "answer_accuracy": (
                sum(1 for r in results if r.answer_correct) /
                sum(1 for r in results if r.answer_correct is not None)
                if any(r.answer_correct is not None for r in results) else 0
            ),
            "abstention_correctness": (
                sum(1 for r in results if r.abstention_correct) /
                sum(1 for r in results if r.abstention_correct is not None)
                if any(r.abstention_correct is not None for r in results) else 0
            ),
        }

    def save_results(self, results: list[EvalResult], benchmark: str, output_dir: Path):
        """Save detailed results to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_path = output_dir / f"eval_{benchmark}_raw.json"
        with open(raw_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2, default=str)

        agg = self.aggregate_results(results)
        agg_path = output_dir / f"eval_{benchmark}_aggregated.json"
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2, default=str)

        print(f"Results saved to {output_dir}")
        return agg


def run_full_evaluation(max_samples_per_benchmark: int = 30):
    """Run full evaluation pipeline."""
    print("=== Initializing Pipeline ===")
    pipeline = Pipeline.from_seed()

    print("=== Initializing Ingestor ===")
    config = IngestionConfig()
    ingestor = ExternalDataIngestor(config)

    print("=== Creating Evaluation Harness ===")
    harness = EvaluationHarness(pipeline, ingestor)

    benchmarks = ["medqa", "pubmedqa", "medmcqa"]
    all_results = {}

    for bench in benchmarks:
        print(f"\n=== Evaluating on {bench.upper()} ===")
        results = harness.run_on_benchmark(bench, max_samples=max_samples_per_benchmark)
        if results:
            agg = harness.aggregate_results(results)
            harness.save_results(results, bench, Path("experiments/results"))
            all_results[bench] = agg
            print(f"  Aggregated: {agg}")

    combined_path = Path("experiments/results") / "eval_combined.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n=== COMBINED RESULTS ===")
    for bench, agg in all_results.items():
        print(f"\n{bench.upper()}:")
        for k, v in agg.items():
            print(f"  {k}: {v}")

    return all_results


if __name__ == "__main__":
    run_full_evaluation(max_samples_per_benchmark=10)
