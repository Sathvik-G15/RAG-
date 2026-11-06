"""CAAR-CDSS: Complete Kaggle Benchmark Pipeline Script.

Designed for Kaggle T4 x 2 or T4 x 1 environments.
Includes robust memory management, background process termination, and 4-bit judge loading.
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

# Set PyTorch allocator configuration to prevent memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:////kaggle/working/caar_cdss.db"
os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
os.makedirs("/kaggle/working/hf_cache", exist_ok=True)

RESULTS_DIR = Path("/kaggle/working/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Pipeline Configuration Parameters
N_QUERIES = 50
BENCHMARKS = "medqa"
INGEST_LIMIT = 2000
RUN_RAGAS = True
JUDGE_MODEL = "microsoft/Phi-3.5-mini-instruct"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHROMA_DIR = "/kaggle/working/chroma_db"


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def disk_gb() -> float:
    return shutil.disk_usage("/kaggle/working").free / (1024**3)


def cleanup_memory(prefix: str = "") -> None:
    """Terminate lingering subprocesses, force GC, and clear PyTorch CUDA caches."""
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
            before = torch.cuda.memory_allocated() / (1024**3)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            after = torch.cuda.memory_allocated() / (1024**3)
            if prefix:
                log(f"{prefix} VRAM: {before:.2f} GB -> {after:.2f} GB allocated")
    except Exception:
        pass


def run_cmd(cmd: str, check: bool = True) -> int:
    log(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if check and res.returncode != 0:
        log(f"Command exited with non-zero code {res.returncode}")
    return res.returncode


def main() -> None:
    log("=== CAAR-CDSS Benchmark Pipeline Starting ===")
    log(f"Free disk: {disk_gb():.1f} GB")

    # 1. Setup HF_TOKEN from Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        token = secrets.get_secret("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
            log("HF_TOKEN successfully configured from Kaggle Secrets.")
    except Exception:
        log("HF_TOKEN not found in Kaggle Secrets (proceeding with public models if applicable).")

    # 2. Hardware check
    try:
        import torch
        if torch.cuda.is_available():
            log(f"CUDA Available: {torch.cuda.device_count()} GPUs detected.")
            for i in range(torch.cuda.device_count()):
                log(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f} GB)")
    except Exception as e:
        log(f"Hardware check warning: {e}")

    # 3. Setup repository code
    repo_input = Path("/kaggle/input/caar-cdss-repo")
    if repo_input.exists():
        run_cmd("cp -r /kaggle/input/caar-cdss-repo/* /kaggle/working/")
        log("Copied repository files from Kaggle input.")

    # 4. Ingest Guidelines / Setup Chroma DB
    chroma_input = Path("/kaggle/input/caar-cdss-chroma")
    if chroma_input.exists():
        run_cmd(f"mkdir -p {CHROMA_DIR} && cp -r /kaggle/input/caar-cdss-chroma/* {CHROMA_DIR}/")
        log("Using pre-built Chroma DB from dataset input.")
    elif not Path(CHROMA_DIR).exists() or not any(Path(CHROMA_DIR).iterdir()):
        log(f"Ingesting guidelines (limit={INGEST_LIMIT})...")
        run_cmd(f"python -m src.cli ingest --corpus epfl-llm/guidelines --limit {INGEST_LIMIT} --chroma-dir {CHROMA_DIR}")
        cleanup_memory("[Post-Ingest]")

    # 5. RAGAS Evaluation
    if RUN_RAGAS:
        log(f"=== Running RAGAS Evaluation (Benchmark: {BENCHMARKS}, N={N_QUERIES}) ===")
        cleanup_memory("[Pre-RAGAS]")

        ragas_cmd = (
            f"python -m src.experiments.ragas_eval "
            f"--benchmark {BENCHMARKS} "
            f"--n {N_QUERIES} "
            f"--mode local_4bit "
            f"--judge-model {JUDGE_MODEL} "
            f"--embedding-model {EMBEDDING_MODEL} "
            f"--chroma-dir {CHROMA_DIR} "
            f"--output {RESULTS_DIR}/ragas_{BENCHMARKS}_results.json "
            f"--max-workers 1 "
            f"--timeout 1800 "
            f"--gc-after-each"
        )
        rc = run_cmd(ragas_cmd, check=False)
        if rc != 0:
            log(f"RAGAS evaluation exited with code {rc}. Attempting fallback with Qwen/Qwen2.5-7B-Instruct...")
            cleanup_memory("[Pre-RAGAS-Fallback]")
            fallback_cmd = ragas_cmd.replace(JUDGE_MODEL, "Qwen/Qwen2.5-7B-Instruct")
            run_cmd(fallback_cmd, check=False)

        cleanup_memory("[Post-RAGAS]")

    # 6. Comparative Evaluation (Vanilla vs Hybrid vs AEB)
    log("=== Running Comparative Evaluation ===")
    cleanup_memory("[Pre-Comparative]")
    comp_cmd = (
        f"python -m src.experiments.runner "
        f"--benchmarks {BENCHMARKS} "
        f"--n {N_QUERIES} "
        f"--mode local_4bit "
        f"--chroma-dir {CHROMA_DIR} "
        f"--gc-after-each "
        f"--output {RESULTS_DIR}/comparative"
    )
    run_cmd(comp_cmd, check=False)
    cleanup_memory("[Post-Comparative]")

    # 7. Package Results
    zip_path = "/kaggle/working/results.zip"
    log(f"Packaging results to {zip_path}...")
    shutil.make_archive("/kaggle/working/results", "zip", RESULTS_DIR)
    log(f"All runs completed. Free disk: {disk_gb():.1f} GB")


if __name__ == "__main__":
    main()
