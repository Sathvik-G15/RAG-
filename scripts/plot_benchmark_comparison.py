"""Generate comprehensive benchmark comparison plots for CAAR-CDSS.

Compares Vanilla RAG, Hybrid RAG, and AEB across:
1. Accuracy (%) across medical benchmarks
2. Adaptive Retrieval Depth (K)
3. Average Confidence & Calibration
4. Selective Abstention Rate (%)
5. Inference Latency (seconds)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

# Ensure Matplotlib has a writable cache directory on Windows
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import matplotlib.pyplot as plt
import numpy as np


def main():
    json_path = Path("experiments/results/kaggle_all_benchmarks/benchmark_summary.json")
    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        return

    data = json.loads(json_path.read_text())
    out_dir = Path("experiments/results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = ["seeds", "medqa", "mmlumed", "ddxplus", "medmcqa", "pubmedqa", "mtsamples"]
    bench_labels = ["Seeds (8)", "MedQA (200)", "MMLU-Med (200)", "DDXPlus (200)", "MedMCQA (200)", "PubMedQA (200)", "MTSamples (200)"]
    methods = ["vanilla", "hybrid", "aeb"]
    method_labels = ["Vanilla RAG (K=5)", "Hybrid RAG (K=10)", "AEB (Adaptive K)"]
    colors = ["#4A90E2", "#50E3C2", "#FF5A5F"]  # Blue, Teal, Coral

    # Extract metrics matrices (len(benchmarks) x 3)
    acc_matrix = []
    k_matrix = []
    conf_matrix = []
    abst_matrix = []
    lat_matrix = []

    for b in benchmarks:
        b_data = data.get(b, {})
        acc_matrix.append([b_data.get(m, {}).get("accuracy", 0.0) for m in methods])
        k_matrix.append([b_data.get(m, {}).get("avg_retrieval_k", 0.0) for m in methods])
        conf_matrix.append([b_data.get(m, {}).get("avg_confidence", 0.0) for m in methods])
        abst_matrix.append([b_data.get(m, {}).get("abstention_rate", 0.0) for m in methods])
        lat_matrix.append([b_data.get(m, {}).get("avg_latency_sec", 0.0) for m in methods])

    acc_matrix = np.array(acc_matrix)
    k_matrix = np.array(k_matrix)
    conf_matrix = np.array(conf_matrix)
    abst_matrix = np.array(abst_matrix)
    lat_matrix = np.array(lat_matrix)

    # Style configuration
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.linewidth"] = 1.0

    # =========================================================================
    # FIGURE 1: 4-Panel Comprehensive Dashboard
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("CAAR-CDSS Benchmark Evaluation: Vanilla vs. Hybrid vs. AEB\n(Model: microsoft/Phi-3.5-mini-instruct FP16 on Kaggle GPU)", fontsize=14, fontweight="bold", y=0.98)

    x = np.arange(len(benchmarks))
    width = 0.25

    # --- Subplot 1: Diagnostic Accuracy (%) ---
    ax1 = axes[0, 0]
    for i, (m, col) in enumerate(zip(method_labels, colors)):
        rects = ax1.bar(x + (i - 1) * width, acc_matrix[:, i], width, label=m, color=col, edgecolor="black", linewidth=0.6, alpha=0.9)
        # Add labels on positive bars
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2),
                             textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax1.set_title("(A) Diagnostic Accuracy (%) Across Benchmarks", fontweight="bold", pad=8)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(bench_labels, rotation=20, ha="right", fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.set_ylim(0, max(acc_matrix.max() * 1.25, 30))

    # --- Subplot 2: Adaptive Retrieval Depth K ---
    ax2 = axes[0, 1]
    for i, (m, col) in enumerate(zip(method_labels, colors)):
        rects = ax2.bar(x + (i - 1) * width, k_matrix[:, i], width, label=m, color=col, edgecolor="black", linewidth=0.6, alpha=0.9)
        for rect in rects:
            h = rect.get_height()
            ax2.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2),
                         textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax2.set_title("(B) Retrieval Depth (Average Chunks K Retrieved)", fontweight="bold", pad=8)
    ax2.set_ylabel("Average Retrieved Evidence Chunks (K)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(bench_labels, rotation=20, ha="right", fontsize=9)
    ax2.axhline(5, color="#4A90E2", linestyle=":", alpha=0.7, label="Vanilla Fixed K=5")
    ax2.axhline(10, color="#50E3C2", linestyle=":", alpha=0.7, label="Hybrid Fixed K=10")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.set_ylim(0, 20)

    # --- Subplot 3: Selective Abstention Rate (%) ---
    ax3 = axes[1, 0]
    for i, (m, col) in enumerate(zip(method_labels, colors)):
        rects = ax3.bar(x + (i - 1) * width, abst_matrix[:, i], width, label=m, color=col, edgecolor="black", linewidth=0.6, alpha=0.9)
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax3.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2),
                             textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax3.set_title("(C) Selective Prediction (Abstention on Low Confidence / Hallucination)", fontweight="bold", pad=8)
    ax3.set_ylabel("Abstention Rate (%)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(bench_labels, rotation=20, ha="right", fontsize=9)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    ax3.legend(loc="upper left", framealpha=0.9)
    ax3.set_ylim(0, 110)

    # --- Subplot 4: Average Confidence Score ---
    ax4 = axes[1, 1]
    for i, (m, col) in enumerate(zip(method_labels, colors)):
        rects = ax4.bar(x + (i - 1) * width, conf_matrix[:, i], width, label=m, color=col, edgecolor="black", linewidth=0.6, alpha=0.9)
        for rect in rects:
            h = rect.get_height()
            ax4.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2),
                         textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax4.set_title("(D) Average Model Confidence (Uncalibrated vs. Verified)", fontweight="bold", pad=8)
    ax4.set_ylabel("Confidence Score [0.0 - 1.0]")
    ax4.axhline(0.70, color="red", linestyle="--", linewidth=1.2, label="AEB Abstention Threshold (0.70)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(bench_labels, rotation=20, ha="right", fontsize=9)
    ax4.grid(axis="y", linestyle="--", alpha=0.5)
    ax4.legend(loc="upper right", framealpha=0.9)
    ax4.set_ylim(0, 1.0)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    dash_path = out_dir / "kaggle_benchmark_dashboard.png"
    plt.savefig(dash_path, dpi=300)
    plt.close()
    print(f"Saved Dashboard: {dash_path}")

    # =========================================================================
    # FIGURE 2: Individual Focused Plot - Accuracy on Answerable Clinical Benchmarks
    # =========================================================================
    plt.figure(figsize=(9, 5))
    clin_indices = [0, 1, 2, 3]  # seeds, medqa, mmlumed, ddxplus
    x_clin = np.arange(len(clin_indices))
    c_labels = [bench_labels[i] for i in clin_indices]

    for i, (m, col) in enumerate(zip(method_labels, colors)):
        vals = [acc_matrix[idx, i] for idx in clin_indices]
        rects = plt.bar(x_clin + (i - 1) * width, vals, width, label=m, color=col, edgecolor="black", linewidth=0.7, alpha=0.95)
        for rect in rects:
            h = rect.get_height()
            plt.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                         textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.title("Diagnostic Accuracy: Vanilla vs. Hybrid vs. AEB on Clinical Datasets", fontsize=12, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.xticks(x_clin, c_labels, fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(framealpha=0.95, fontsize=10)
    plt.ylim(0, 32)
    plt.tight_layout()
    acc_path = out_dir / "kaggle_accuracy_comparison.png"
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print(f"Saved Accuracy Plot: {acc_path}")

    # =========================================================================
    # FIGURE 3: Individual Focused Plot - Latency vs. Retrieval K Trade-off
    # =========================================================================
    plt.figure(figsize=(9, 5))
    # Mean across all 7 benchmarks
    mean_k = [k_matrix[:, i].mean() for i in range(3)]
    mean_lat = [lat_matrix[:, i].mean() for i in range(3)]
    mean_acc = [acc_matrix[:, i].mean() for i in range(3)]

    for i, (m, col) in enumerate(zip(method_labels, colors)):
        plt.scatter(mean_k[i], mean_lat[i], s=350, color=col, edgecolor="black", linewidth=1.5, label=f"{m} (Mean Acc: {mean_acc[i]:.2f}%)", zorder=4)
        plt.annotate(f"{m}\nAvg K={mean_k[i]:.1f}, {mean_lat[i]:.1f}s", xy=(mean_k[i], mean_lat[i]), xytext=(10, -5),
                     textcoords="offset points", fontsize=9, fontweight="bold")

    plt.title("Efficiency Frontier: Retrieval Depth (K) vs. Latency Trade-off", fontsize=12, fontweight="bold")
    plt.xlabel("Average Retrieved Chunks (K)", fontsize=11)
    plt.ylabel("Average Inference Latency per Query (seconds)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(3, 18)
    plt.ylim(0, 80)
    plt.legend(loc="upper left", framealpha=0.95, fontsize=9)
    plt.tight_layout()
    tradeoff_path = out_dir / "kaggle_latency_vs_k.png"
    plt.savefig(tradeoff_path, dpi=300)
    plt.close()
    print(f"Saved Tradeoff Plot: {tradeoff_path}")

    print("\nAll benchmark plots generated successfully!")


if __name__ == "__main__":
    main()
