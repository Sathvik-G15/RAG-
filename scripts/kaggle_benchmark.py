"""CAAR-CDSS: Kaggle Benchmark Pipeline Script.

Optimized for Kaggle GPU environments (T4 x 2, T4 x 1, or P100).
Uses microsoft/Phi-3.5-mini-instruct (3.8B parameters) in FP16 (kaggle_fp16 mode),
offering ~4x faster execution and lower memory footprint (~7.5GB VRAM) than 8B models.

Usage Examples:
  # Run medqa and seeds (50 samples each, comparative evaluation)
  python scripts/kaggle_benchmark.py --benchmarks seeds medqa --n 50

  # Run all 7 benchmarks (100 samples each) without RAGAS judge (fastest)
  python scripts/kaggle_benchmark.py --benchmarks all --n 100 --skip-ragas

  # Run full benchmark with RAGAS evaluation
  python scripts/kaggle_benchmark.py --benchmarks medqa pubmedqa --n 50 --run-ragas
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# PyTorch allocator settings to minimize fragmentation on Kaggle GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

is_kaggle_env = Path("/kaggle").exists()
if is_kaggle_env:
    os.environ["DATABASE_URL"] = os.environ.get(
        "DATABASE_URL", "sqlite+aiosqlite:////kaggle/working/caar_cdss.db"
    )
    os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/kaggle/working/hf_cache")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def disk_gb() -> float:
    path = "/kaggle/working" if Path("/kaggle/working").exists() else "."
    try:
        return shutil.disk_usage(path).free / (1024**3)
    except Exception:
        return 0.0


def cleanup_memory(prefix: str = "") -> None:
    """Terminate child processes, run GC, and release PyTorch CUDA caches."""
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


def setup_hf_token() -> None:
    """Attempt to load HF_TOKEN from Kaggle Secrets if available."""
    try:
        from kaggle_secrets import UserSecretsClient

        secrets = UserSecretsClient()
        token = secrets.get_secret("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
            log("HF_TOKEN successfully loaded from Kaggle Secrets.")
    except Exception:
        if os.environ.get("HF_TOKEN"):
            log("Using existing HF_TOKEN from environment.")
        else:
            log("No HF_TOKEN found (proceeding with public Hugging Face models).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CAAR-CDSS benchmarks on Kaggle GPU."
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["seeds", "medqa"],
        help="Benchmarks to run ('all', or list e.g. seeds medqa pubmedqa ddxplus medmcqa mtsamples mmlumed)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of evaluation samples per benchmark dataset (default: 50)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="kaggle_fp16",
        choices=["kaggle_fp16", "local_4bit", "real", "mock"],
        help="Inference mode (default: kaggle_fp16 for Phi-3.5-mini FP16 on Kaggle)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/Phi-3.5-mini-instruct",
        help="HuggingFace model ID or offline directory path",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="microsoft/Phi-3.5-mini-instruct",
        help="RAGAS LLM judge model (default: microsoft/Phi-3.5-mini-instruct)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="Embedding model ID (default: BAAI/bge-small-en-v1.5)",
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default=None,
        help="Path to ChromaDB directory (auto-detected if omitted)",
    )
    parser.add_argument(
        "--ingest-limit",
        type=int,
        default=1000,
        help="Number of guideline chunks to ingest if building ChromaDB from scratch",
    )
    parser.add_argument(
        "--run-ragas",
        action="store_true",
        default=False,
        help="Run RAGAS judge evaluation (faithfulness, answer relevance)",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        default=False,
        help="Explicitly skip RAGAS judge evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for reports (default: /kaggle/working/results or experiments/results)",
    )
    parser.add_argument(
        "--strict-match",
        action="store_true",
        default=True,
        help="Use strict letter/entity matching for multiple-choice options",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine working directories
    is_kaggle = Path("/kaggle/working").exists()
    base_dir = Path("/kaggle/working") if is_kaggle else Path(".")
    results_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (base_dir / "results" if is_kaggle else Path("experiments/results"))
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    log("======================================================================")
    log(" CAAR-CDSS: Kaggle Benchmark Pipeline")
    log(f" Model:       {args.model} ({args.mode})")
    log(f" Benchmarks:  {args.benchmarks}")
    log(f" Samples:     N={args.n} per dataset")
    log(f" Results Dir: {results_dir}")
    log(f" Free Disk:   {disk_gb():.1f} GB")
    log("======================================================================")

    # 1. Setup HF_TOKEN
    setup_hf_token()

    # 2. Hardware check
    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            log(f"CUDA Available: {gpu_count} GPU(s) detected.")
            for i in range(gpu_count):
                total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                log(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({total_mem:.1f} GB VRAM)")
        else:
            log("WARNING: CUDA not detected! Running on CPU will be slow.")
    except Exception as e:
        log(f"Hardware check note: {e}")

    # 3. Synchronize repository files if attached as Kaggle input
    repo_input = Path("/kaggle/input/caar-cdss-repo")
    if repo_input.exists():
        log("Copying repository files from /kaggle/input/caar-cdss-repo...")
        run_cmd("cp -r /kaggle/input/caar-cdss-repo/* /kaggle/working/")

    # 4. Check for offline models in /kaggle/input
    model_to_use = args.model
    model_indices = list(Path("/kaggle/input").glob("**/model.safetensors.index.json"))
    if model_indices:
        offline_model_dir = str(model_indices[0].parent)
        log(f"Found offline model weights at: {offline_model_dir}")
        model_to_use = offline_model_dir

    # 5. Check for Chroma DB
    chroma_dir = args.chroma_dir
    if not chroma_dir:
        kaggle_chroma = Path("/kaggle/input/caar-cdss-chroma")
        local_chroma = Path("data/chroma_db")
        working_chroma = Path("/kaggle/working/chroma_db") if is_kaggle else Path("data/chroma_db")

        if kaggle_chroma.exists():
            log(f"Using attached ChromaDB from: {kaggle_chroma}")
            chroma_dir = str(kaggle_chroma)
        elif local_chroma.exists() and any(local_chroma.iterdir()):
            log(f"Using local ChromaDB at: {local_chroma}")
            chroma_dir = str(local_chroma)
        elif working_chroma.exists() and any(working_chroma.iterdir()):
            log(f"Using working ChromaDB at: {working_chroma}")
            chroma_dir = str(working_chroma)
        else:
            # If no ChromaDB exists and unified KB exists, we can run against unified KB
            unified_kb = Path("data/unified_clinical_kb.jsonl")
            if unified_kb.exists():
                log(f"No ChromaDB found, but unified KB exists at {unified_kb}.")
                chroma_dir = None
            else:
                log(f"Ingesting seed guidelines into {working_chroma} (limit={args.ingest_limit})...")
                run_cmd(
                    f"python -m src.cli ingest --corpus epfl-llm/guidelines --limit {args.ingest_limit} --chroma-dir {working_chroma}"
                )
                chroma_dir = str(working_chroma)
                cleanup_memory("[Post-Ingest]")

    # 6. Parse benchmark targets
    benchmarks_list = args.benchmarks
    if "all" in benchmarks_list:
        benchmarks_str = "seeds medqa pubmedqa mmlumed ddxplus medmcqa mtsamples"
    else:
        benchmarks_str = " ".join(benchmarks_list)

    # 7. Update configs/seeds.yaml to reflect selected model and backend
    try:
        import yaml

        cfg_path = Path("configs/seeds.yaml")
        if cfg_path.exists():
            cfg_data = yaml.safe_load(cfg_path.read_text()) or {}
            if "llm" in cfg_data:
                cfg_data["llm"]["model_name"] = model_to_use
                cfg_data["llm"]["serving_backend"] = args.mode
            if "ragas_judge" in cfg_data:
                cfg_data["ragas_judge"]["model_name"] = args.judge_model
            cfg_path.write_text(yaml.dump(cfg_data, sort_keys=False))
            log(f"Updated configs/seeds.yaml: model={model_to_use}, backend={args.mode}")
    except Exception as e:
        log(f"Warning updating config: {e}")

    # 8. Run Comparative Evaluation (Vanilla vs Hybrid vs AEB)
    log("======================================================================")
    log(f" Running Comparative Evaluation: {benchmarks_str} (N={args.n})")
    log("======================================================================")
    cleanup_memory("[Pre-Comparative]")

    comp_cmd_parts = [
        sys.executable,
        "-m src.experiments.runner",
        f"--benchmarks {benchmarks_str}",
        f"--n {args.n}",
        f"--mode {args.mode}",
        f"--output {results_dir}",
        "--compute-matched",
        "--per-round",
        "--verification-ablation",
        "--gc-after-each",
    ]
    if args.strict_match:
        comp_cmd_parts.append("--strict-match")
    if chroma_dir:
        comp_cmd_parts.append(f"--chroma-dir {chroma_dir}")

    comp_cmd = " ".join(comp_cmd_parts)
    run_cmd(comp_cmd, check=False)
    cleanup_memory("[Post-Comparative]")

    # 9. Run RAGAS Evaluation (optional)
    should_run_ragas = args.run_ragas and not args.skip_ragas
    if should_run_ragas:
        log("======================================================================")
        log(f" Running RAGAS Evaluation: {benchmarks_str}")
        log("======================================================================")
        cleanup_memory("[Pre-RAGAS]")

        ragas_cmd = (
            f"{sys.executable} -m src.experiments.ragas_eval "
            f"--benchmark {benchmarks_list[0] if benchmarks_list else 'medqa'} "
            f"--n {min(args.n, 30)} "
            f"--mode {args.mode} "
            f"--judge-model {args.judge_model} "
            f"--embedding-model {args.embedding_model} "
            + (f"--chroma-dir {chroma_dir} " if chroma_dir else "")
            + f"--output {results_dir}/ragas_results.json "
            f"--max-workers 1 "
            f"--timeout 1800 "
            f"--gc-after-each"
        )
        rc = run_cmd(ragas_cmd, check=False)
        if rc != 0:
            log(f"RAGAS evaluation exited with code {rc} (continuing).")
        cleanup_memory("[Post-RAGAS]")

    # 10. Package and compress results
    log("======================================================================")
    log(" Packaging Results")
    log("======================================================================")
    if is_kaggle:
        tar_path = "/kaggle/working/caar_cdss_results.tar.gz"
        zip_path = "/kaggle/working/caar_cdss_results.zip"
        shutil.make_archive("/kaggle/working/caar_cdss_results", "zip", results_dir)
        run_cmd(f"tar -czvf {tar_path} -C {results_dir.parent} {results_dir.name}")
        log(f"Results archived to:\n  - {zip_path}\n  - {tar_path}")
    else:
        log(f"Results saved locally in: {results_dir}")

    log("=== CAAR-CDSS Benchmark Pipeline Completed Successfully! ===")


if __name__ == "__main__":
    main()
