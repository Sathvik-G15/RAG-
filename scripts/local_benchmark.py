"""CAAR-CDSS: Local Benchmark Pipeline Script.

Adapted from kaggle_benchmark.py for local hardware:
  - GPU  : NVIDIA RTX 4050 6 GB VRAM
  - RAM  : 24 GB system memory
  - OS   : Windows (PowerShell)

Key differences from the Kaggle version:
  - Paths resolve relative to the project root (no /kaggle/working prefix).
  - HF_TOKEN is read from the .env file instead of Kaggle Secrets.
  - VRAM cap is 5 GiB per GPU (leaves ~1 GB headroom for OS / CUDA overhead).
  - CPU offload budget is 18 GiB (leaves 6 GB headroom for OS + embeddings).
  - Default judge model is Qwen2.5-3B-Instruct (fits comfortably in 4-bit on
    6 GB VRAM; ~2.5 GB quantised).  The 8B model is kept as fallback but will
    rely heavily on CPU offloading.
  - disk_gb() points at the project drive, not /kaggle/working.
  - No cp / mkdir shell commands that are Linux-only; pathlib is used instead.
"""

from __future__ import annotations

import datetime
import gc
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment – must be set BEFORE importing torch / transformers
# ---------------------------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Resolve project root (two levels up from this script: scripts/ -> project/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env so HF_TOKEN and other secrets are available
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv(_env_file, override=False)

# Point HF cache inside the project so it survives across runs
HF_CACHE = PROJECT_ROOT / "data" / "hf_cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE)

# SQLite DB for async operations
_db_path = PROJECT_ROOT / "data" / "caar_cdss.db"
_db_path.parent.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{_db_path}")

# ---------------------------------------------------------------------------
# Pipeline configuration  (edit here to customise a run)
# ---------------------------------------------------------------------------
N_QUERIES    = 500
# Comma-separated list of benchmarks, or "all" to auto-detect downloaded datasets.
# Individual benchmarks: medqa pubmedqa ddxplus mmlumed
BENCHMARKS   = "medqa pubmedqa ddxplus mmlumed"
INGEST_LIMIT = 2000

RUN_RAGAS    = False

# Primary judge: Qwen2.5-1.5B fits comfortably in VRAM (~1.2 GB in 4-bit)
# Lightweight fallback: Qwen2.5-0.5B (~0.6 GB in 4-bit)
JUDGE_MODEL     = "Qwen/Qwen2.5-1.5B-Instruct"
FALLBACK_JUDGE  = "Qwen/Qwen2.5-0.5B-Instruct"



EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHROMA_DIR      = str(PROJECT_ROOT / "data" / "chroma_db")

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def disk_gb() -> float:
    """Free disk space on the project drive."""
    return shutil.disk_usage(PROJECT_ROOT).free / (1024 ** 3)


def cleanup_memory(prefix: str = "") -> None:
    """Terminate child processes, force GC, and clear PyTorch CUDA caches."""
    try:
        import multiprocessing
        for p in multiprocessing.active_children():
            try:
                p.terminate()
                p.join(timeout=0.5)
            except Exception:
                pass
    except Exception:
        pass

    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            before = torch.cuda.memory_allocated() / (1024 ** 3)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            after = torch.cuda.memory_allocated() / (1024 ** 3)
            if prefix:
                log(f"{prefix} VRAM: {before:.2f} GB -> {after:.2f} GB allocated")
    except Exception:
        pass


def run_cmd(cmd: str, check: bool = True) -> int:
    log(f"Running: {cmd}")
    # Use sys.executable so the correct venv Python is used
    res = subprocess.run(cmd, shell=True, cwd=str(PROJECT_ROOT))
    if check and res.returncode != 0:
        log(f"Command exited with non-zero code {res.returncode}")
    return res.returncode


def python(args: str, check: bool = True) -> int:
    """Run a python module/script using the current interpreter."""
    return run_cmd(f'"{sys.executable}" {args}', check=check)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    log("=== CAAR-CDSS Local Benchmark Pipeline Starting ===")
    log(f"Project root : {PROJECT_ROOT}")
    log(f"Free disk    : {disk_gb():.1f} GB")

    # 1. HF_TOKEN
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        log("HF_TOKEN loaded from environment / .env file.")
    else:
        log("WARNING: HF_TOKEN not set. Gated models (e.g. Llama) will fail to download.")

    # 2. Hardware check
    try:
        import torch
        if torch.cuda.is_available():
            log(f"CUDA Available: {torch.cuda.device_count()} GPU(s) detected.")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                log(f"  GPU {i}: {props.name}  ({props.total_memory / (1024**3):.1f} GB VRAM)")
        else:
            log("No CUDA device detected – running in CPU-only mode (expect slow inference).")
    except Exception as e:
        log(f"Hardware check warning: {e}")

    # 3. Ingest guidelines / setup Chroma DB
    chroma_path = Path(CHROMA_DIR)
    if chroma_path.exists() and any(chroma_path.iterdir()):
        log(f"Using existing Chroma DB at {CHROMA_DIR}.")
    else:
        log(f"Ingesting guidelines (limit={INGEST_LIMIT})...")
        python(
            f"-m src.cli ingest "
            f"--corpus epfl-llm/guidelines "
            f"--limit {INGEST_LIMIT} "
            f"--chroma-dir \"{CHROMA_DIR}\"",
        )
        cleanup_memory("[Post-Ingest]")

    # 4. RAGAS Evaluation
    if RUN_RAGAS:
        log(f"=== Running RAGAS Evaluation (Benchmark: {BENCHMARKS}, N={N_QUERIES}) ===")
        cleanup_memory("[Pre-RAGAS]")

        ragas_cmd = (
            f"-m src.experiments.ragas_eval "
            f"--benchmark {BENCHMARKS} "
            f"--n {N_QUERIES} "
            f"--mode local_4bit "
            f"--judge-model {JUDGE_MODEL} "
            f"--embedding-model {EMBEDDING_MODEL} "
            f"--chroma-dir \"{CHROMA_DIR}\" "
            f"--output \"{RESULTS_DIR / f'ragas_{BENCHMARKS}_results.json'}\" "
            f"--max-workers 1 "
            f"--timeout 1800 "
            f"--gc-after-each"
        )
        rc = python(ragas_cmd, check=False)
        if rc != 0:
            log(f"RAGAS evaluation failed (code {rc}). Retrying with fallback judge {FALLBACK_JUDGE}...")
            cleanup_memory("[Pre-RAGAS-Fallback]")
            fallback_cmd = ragas_cmd.replace(JUDGE_MODEL, FALLBACK_JUDGE)
            python(fallback_cmd, check=False)

        cleanup_memory("[Post-RAGAS]")

    # 5. Comparative Evaluation (Vanilla vs Hybrid vs AEB)
    log("=== Running Comparative Evaluation ===")
    cleanup_memory("[Pre-Comparative]")
    comp_cmd = (
        f"-m src.experiments.runner "
        f"--benchmarks {BENCHMARKS} "
        f"--n {N_QUERIES} "
        f"--mode local_4bit "
        f"--chroma-dir \"{CHROMA_DIR}\" "
        f"--gc-after-each "
        f"--output \"{RESULTS_DIR / 'comparative'}\""
    )
    python(comp_cmd, check=False)
    cleanup_memory("[Post-Comparative]")

    # 6. Package results
    zip_base = str(RESULTS_DIR.parent / "results_local")
    log(f"Packaging results to {zip_base}.zip ...")
    shutil.make_archive(zip_base, "zip", RESULTS_DIR)
    log(f"All runs completed. Free disk: {disk_gb():.1f} GB")
    log(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
