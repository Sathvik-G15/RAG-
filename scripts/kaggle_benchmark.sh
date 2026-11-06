#!/usr/bin/env bash
# ==============================================================================
# CAAR-CDSS: Kaggle Benchmark Runner Script
#
# Optimized for microsoft/Phi-3.5-mini-instruct (3.8B parameters) in FP16 (kaggle_fp16).
# Runs ~4x faster than 8B models and uses ~7.5GB VRAM on Kaggle T4 / P100 GPUs.
#
# Usage in Kaggle Notebook:
#   !bash scripts/kaggle_benchmark.sh seeds 22       # Fast test: 22 clinical seed cases (~2 min)
#   !bash scripts/kaggle_benchmark.sh part1 100      # Part 1: seeds, medqa, pubmedqa, mmlumed (~45 min)
#   !bash scripts/kaggle_benchmark.sh part2 100      # Part 2: ddxplus, medmcqa, mtsamples (~45 min)
#   !bash scripts/kaggle_benchmark.sh all 50         # All 7 benchmarks, 50 samples each (~1 hr)
# ==============================================================================

set -e

STAGE=${1:-part1}
N_SAMPLES=${2:-50}

if [ "$STAGE" = "part1" ]; then
    BENCHMARKS="seeds medqa pubmedqa mmlumed"
    OUT_DIR="kaggle_eval_part1"
elif [ "$STAGE" = "part2" ]; then
    BENCHMARKS="ddxplus medmcqa mtsamples"
    OUT_DIR="kaggle_eval_part2"
elif [ "$STAGE" = "all" ]; then
    BENCHMARKS="seeds medqa pubmedqa mmlumed ddxplus medmcqa mtsamples"
    OUT_DIR="kaggle_eval_all"
else
    # Custom benchmarks or single benchmark (e.g. 'seeds' or 'medqa pubmedqa')
    BENCHMARKS="$STAGE"
    OUT_DIR="kaggle_eval_${STAGE// /_}"
fi

echo "======================================================================"
echo " CAAR-CDSS: Starting Kaggle Benchmark Run"
echo " Model:      microsoft/Phi-3.5-mini-instruct (FP16)"
echo " Stage:      $STAGE"
echo " Benchmarks: $BENCHMARKS"
echo " Samples:    N=$N_SAMPLES per benchmark"
echo " Output:     /kaggle/working/results/$OUT_DIR"
echo "======================================================================"

echo "=== 1. Checking GPU Environment ==="
nvidia-smi || true

echo "=== 2. Installing Required Dependencies ==="
pip install -q --upgrade pip
pip install -q \
    transformers \
    accelerate \
    sentence-transformers \
    datasets \
    rank-bm25 \
    chromadb \
    fastapi \
    pydantic \
    pydantic-settings \
    scikit-learn \
    scipy \
    FlagEmbedding \
    python-jose[cryptography] \
    aiosqlite \
    rouge-score \
    tqdm

echo "=== 3. Detecting Local Datasets & Pre-computed Models ==="
mkdir -p data/external
mkdir -p /kaggle/working/results

# Copy offline datasets if attached
DATA_INPUT=$(find /kaggle/input -name "pubmedqa.jsonl" 2>/dev/null | head -n 1)
if [ -n "$DATA_INPUT" ]; then
    DATA_DIR=$(dirname "$DATA_INPUT")
    echo "Found offline datasets at: $DATA_DIR"
    cp "$DATA_DIR"/*.jsonl data/external/ 2>/dev/null || true
    echo "Copied datasets into data/external/"
fi

# Check for attached Chroma DB
CHROMA_INPUT=$(find /kaggle/input -name "chroma.sqlite3" 2>/dev/null | head -n 1)
CHROMA_ARG=""
if [ -n "$CHROMA_INPUT" ]; then
    CHROMA_DIR=$(dirname "$CHROMA_INPUT")
    echo "Found pre-built Chroma DB at: $CHROMA_DIR"
    CHROMA_ARG="--chroma-dir $CHROMA_DIR"
elif [ -d "data/chroma_db" ]; then
    echo "Found local Chroma DB at data/chroma_db"
    CHROMA_ARG="--chroma-dir data/chroma_db"
else
    echo "No Chroma DB attached. The pipeline will utilize the in-memory unified KB."
fi

# Check if model was attached via Kaggle Datasets
MODEL_INDEX=$(find /kaggle/input -name "model.safetensors.index.json" 2>/dev/null | head -n 1)
if [ -n "$MODEL_INDEX" ]; then
    MODEL_DIR=$(dirname "$MODEL_INDEX")
    echo "Found offline model at: $MODEL_DIR"
    python -c "
import yaml
from pathlib import Path
p = Path('configs/seeds.yaml')
if p.exists():
    d = yaml.safe_load(p.read_text()) or {}
    if 'llm' in d:
        d['llm']['model_name'] = '$MODEL_DIR'
        d['llm']['serving_backend'] = 'kaggle_fp16'
    if 'ragas_judge' in d:
        d['ragas_judge']['model_name'] = '$MODEL_DIR'
    p.write_text(yaml.dump(d, sort_keys=False))
    print('Updated configs/seeds.yaml with offline model:', d['llm']['model_name'])
"
else
    echo "Using default Hugging Face model: microsoft/Phi-3.5-mini-instruct"
fi

echo "=== 4. Running Benchmark Suite on GPU (kaggle_fp16 mode) ==="
python -m src.experiments.runner \
    --mode kaggle_fp16 \
    --benchmarks $BENCHMARKS \
    --n $N_SAMPLES \
    --compute-matched \
    --per-round \
    --verification-ablation \
    --strict-match \
    --gc-after-each \
    $CHROMA_ARG \
    --output "/kaggle/working/results/$OUT_DIR"

echo "=== 5. Packaging Results ==="
ARCHIVE_NAME="${OUT_DIR}_results.tar.gz"
ZIP_NAME="${OUT_DIR}_results.zip"

tar -czvf "/kaggle/working/$ARCHIVE_NAME" -C /kaggle/working/results "$OUT_DIR" || true
python -c "import shutil; shutil.make_archive('/kaggle/working/${OUT_DIR}_results', 'zip', '/kaggle/working/results/$OUT_DIR')" || true

echo "======================================================================"
echo " Completed successfully!"
echo " Results available in Kaggle Output panel:"
echo "   - /kaggle/working/$ARCHIVE_NAME"
echo "   - /kaggle/working/$ZIP_NAME"
echo "======================================================================"
