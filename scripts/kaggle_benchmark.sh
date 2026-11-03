#!/usr/bin/env bash
# ==============================================================================
# CAAR-CDSS: Kaggle T4 Benchmark Runner Script
# Runs all 7 clinical benchmarks with 8B model in fp16 on Kaggle T4 GPU
# ==============================================================================

set -e

echo "=== 1. Checking GPU Environment ==="
nvidia-smi

echo "=== 2. Installing Required Dependencies ==="
pip install -q --upgrade pip
pip install -q \
    transformers \
    accelerate \
    bitsandbytes \
    sentence-transformers \
    datasets \
    rank-bm25 \
    chromadb \
    fastapi \
    pydantic \
    pydantic-settings \
    scikit-learn \
    scipy \
    FlagEmbedding

echo "=== 3. Downloading Datasets ==="
python scripts/download_datasets.py

echo "=== 4. Running Benchmark Suite on Kaggle GPU ==="
python -m src.experiments.runner \
    --mode kaggle_fp16 \
    --benchmarks seeds medqa pubmedqa mmlumed ddxplus medmcqa mtsamples \
    --n 500 \
    --compute-matched \
    --per-round \
    --verification-ablation \
    --strict-match \
    --output experiments/results/kaggle_eval

echo "=== 5. Packaging Results ==="
tar -czvf kaggle_eval_results.tar.gz -C experiments/results kaggle_eval

echo "=== Done! Download kaggle_eval_results.tar.gz from /kaggle/working ==="
