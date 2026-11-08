"""Compile and summarize all 7 Kaggle benchmark results into unified reports."""

from __future__ import annotations

import json
from pathlib import Path


def main():
    unified_dir = Path("experiments/results/kaggle_all_benchmarks")
    reports = sorted(unified_dir.glob("report_*.json"))

    benchmarks = ["seeds", "medqa", "mmlumed", "ddxplus", "medmcqa", "pubmedqa", "mtsamples"]
    methods = ["vanilla", "hybrid", "aeb"]

    data_map = {}
    for p in reports:
        d = json.loads(p.read_text())
        bench = p.stem.split("_")[1]
        method = d.get("method", p.stem.split("_")[2])
        n = d.get("n", len(d.get("results", [])))
        acc = d.get("accuracy", 0.0) * 100
        conf = d.get("avg_confidence", 0.0)
        halluc = d.get("avg_hallucination", 0.0)
        k = d.get("avg_retrieval_k", 0.0)
        lat = d.get("avg_latency_ms", 0.0) / 1000.0
        abst = d.get("abstention_rate", 0.0) * 100

        if bench not in data_map:
            data_map[bench] = {}
        data_map[bench][method] = {
            "n": n,
            "accuracy": round(acc, 2),
            "avg_confidence": round(conf, 3),
            "avg_hallucination": round(halluc, 3),
            "avg_retrieval_k": round(k, 1),
            "avg_latency_sec": round(lat, 1),
            "abstention_rate": round(abst, 1),
        }

    # 1. Save master JSON
    summary_json_path = unified_dir / "benchmark_summary.json"
    summary_json_path.write_text(json.dumps(data_map, indent=2))
    print(f"Saved: {summary_json_path}")

    # 2. Build Markdown Table
    lines = [
        "# CAAR-CDSS: Complete Kaggle Benchmark Results",
        "",
        "Evaluated on Kaggle GPU with `microsoft/Phi-3.5-mini-instruct` (FP16) across all 7 medical benchmarks.",
        "",
        "| Benchmark | Method | N | Accuracy | Avg Conf | Avg K | Avg Latency | Abstention |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for b in benchmarks:
        if b in data_map:
            for m in methods:
                if m in data_map[b]:
                    row = data_map[b][m]
                    lines.append(
                        f"| **{b.upper()}** | `{m}` | {row['n']} | **{row['accuracy']:.2f}%** | {row['avg_confidence']:.3f} | {row['avg_retrieval_k']:.1f} | {row['avg_latency_sec']:.1f}s | {row['abstention_rate']:.1f}% |"
                    )

    report_md_path = unified_dir / "BENCHMARK_REPORT.md"
    report_md_path.write_text("\n".join(lines))
    print(f"Saved: {report_md_path}\n")

    # 3. Print Console Table
    print(f"{'Benchmark':<12} | {'Method':<10} | {'N':<4} | {'Accuracy':<9} | {'Conf':<6} | {'Avg K':<6} | {'Latency':<8} | {'Abstention':<10}")
    print("-" * 80)
    for b in benchmarks:
        if b in data_map:
            for m in methods:
                if m in data_map[b]:
                    row = data_map[b][m]
                    print(f"{b:<12} | {m:<10} | {row['n']:4} | {row['accuracy']:6.2f}%   | {row['avg_confidence']:6.3f} | {row['avg_retrieval_k']:6.1f} | {row['avg_latency_sec']:6.1f}s | {row['abstention_rate']:9.1f}%")
            print("-" * 80)


if __name__ == "__main__":
    main()
