#!/usr/bin/env bash
# ==============================================================================
# CAAR-CDSS: Kaggle T4 Benchmark Runner Script
# Runs all 7 clinical benchmarks with 8B model in fp16 on Kaggle T4 GPU
# Automatically detects offline uploaded model & datasets if attached
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

echo "=== 3. Detecting Local Datasets & Models ==="
mkdir -p data/external

# Check if datasets were attached via Kaggle Datasets
DATA_INPUT=$(find /kaggle/input -name "pubmedqa.jsonl" 2>/dev/null | head -n 1)
if [ -n "$DATA_INPUT" ]; then
    DATA_DIR=$(dirname "$DATA_INPUT")
    echo "Found offline datasets at: $DATA_DIR"
    cp "$DATA_DIR"/*.jsonl data/external/ 2>/dev/null || true
    echo "Copied datasets into data/external/"
else
    echo "No offline datasets found in /kaggle/input. Downloading datasets..."
    python scripts/download_datasets.py
fi

# Check if model was attached via Kaggle Datasets
MODEL_INDEX=$(find /kaggle/input -name "model.safetensors.index.json" 2>/dev/null | head -n 1)
if [ -n "$MODEL_INDEX" ]; then
    MODEL_DIR=$(dirname "$MODEL_INDEX")
    echo "Found offline model at: $MODEL_DIR"
    
    # Safely update ONLY the LLM model_name (DO NOT overwrite embedding model_name)
    python -c "
import yaml
from pathlib import Path
p = Path('configs/seeds.yaml')
d = yaml.safe_load(p.read_text())
d['llm']['model_name'] = '$MODEL_DIR'
if 'ragas_judge' in d:
    d['ragas_judge']['model_name'] = '$MODEL_DIR'
p.write_text(yaml.dump(d, sort_keys=False))
print('Successfully configured llm.model_name ->', d['llm']['model_name'])
print('Preserved embedding.model_name ->', d['embedding']['model_name'])
"
else
    echo "No offline model found in /kaggle/input. Model will download from Hugging Face if needed."
fi

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
tar -czvf /kaggle/working/kaggle_eval_results.tar.gz -C experiments/results kaggle_eval

echo "=== Done! Download kaggle_eval_results.tar.gz from /kaggle/working ==="
