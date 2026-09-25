"""RAGAS automated evaluation harness for CAAR-CDSS with Kaggle & HF API judge support."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any
import sys
import types

# Compatibility shim: some versions of ragas attempt to import ChatVertexAI from langchain_community
try:
    import langchain_community.chat_models  # type: ignore
    if "langchain_community.chat_models.vertexai" not in sys.modules:
        v_mod = types.ModuleType("langchain_community.chat_models.vertexai")
        class ChatVertexAI: pass
        v_mod.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = v_mod
        setattr(langchain_community.chat_models, "vertexai", v_mod)
except Exception:
    pass

logger = logging.getLogger(__name__)


class KagglePipelineJudge:
    """RAGAS judge that wraps a loaded HuggingFace pipeline (for Kaggle T4).
    
    Reuses the already-loaded fp16 model - zero API calls, zero quota.
    """
    def __init__(self, pipe: Any):
        from langchain_community.llms import HuggingFacePipeline
        from ragas.llms import LangchainLLMWrapper

        self._llm = LangchainLLMWrapper(HuggingFacePipeline(pipeline=pipe))

    @property
    def llm(self):
        return self._llm


class HFAPIJudge:
    """RAGAS judge using Hugging Face Inference API (for local dev).
    
    Uses serverless endpoint - zero local VRAM.
    """
    def __init__(self, model_name: str, hf_token: str | None = None, provider: str | None = "together"):
        from langchain_huggingface import HuggingFaceEndpoint
        from ragas.llms import LangchainLLMWrapper

        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN environment variable or parameter is required for HF Inference API judge.")

        logger.info("Using Hugging Face Inference API judge (%s) via provider=%s.", model_name, provider)
        endpoint = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=token,
            temperature=0.01,
            max_new_tokens=256,
            provider=provider,
        )
        self._llm = LangchainLLMWrapper(endpoint)

    @property
    def llm(self):
        return self._llm


def get_ragas_judge(pipe: Any = None, hf_token: str | None = None, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", provider: str | None = "together"):
    """Get RAGAS judge wrapper.
    
    - If pipe is provided (e.g. on Kaggle T4), uses KagglePipelineJudge (0 API calls).
    - Otherwise uses HFAPIJudge via Hugging Face Inference API.
    """
    if pipe is not None:
        logger.info("Using KagglePipelineJudge (reuses loaded fp16 model, 0 API calls).")
        return KagglePipelineJudge(pipe).llm

    logger.info("Using HFAPIJudge via Hugging Face Inference API (%s) with provider=%s.", model_name, provider)
    return HFAPIJudge(model_name, hf_token, provider=provider).llm


def get_ragas_embeddings(model_name: str = "BAAI/bge-small-en-v1.5", force_cpu: bool = True):
    """Get open-source Hugging Face embeddings wrapper for RAGAS evaluation (0 OpenAI calls).

    Args:
        model_name: HuggingFace embedding model to use.
        force_cpu: When True (default), always loads on CPU regardless of CUDA availability.
            This prevents VRAM contention between bge-small and the pipeline/judge LLM,
            which is the primary cause of Step-6 hangs on Kaggle T4.
            Set to False only if you have confirmed spare VRAM (e.g. A100 with headroom).
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    import torch

    if force_cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "Initializing Hugging Face embeddings for RAGAS (%s on %s, force_cpu=%s)...",
        model_name, device, force_cpu,
    )
    hf_emb = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(hf_emb)


def log_resource_snapshot(label: str = "snapshot") -> dict:
    """Capture GPU + RAM + top-process snapshot at the moment of a stall.

    Call this in a notebook cell (or add to your loop) right when the pipeline
    appears to hang.  Mirrors the three diagnostic commands from the debug guide::

        nvidia-smi
        free -h
        ps aux --sort=-%mem | head -n 15

    Returns a dict with keys ``gpu``, ``ram``, ``top_procs`` so callers can
    also log / save the snapshot programmatically.
    """
    import platform
    import subprocess
    import sys

    snapshot: dict = {"label": label}

    # --- GPU via torch (cross-platform, always available if torch is installed) ---
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                alloc = torch.cuda.memory_allocated(i) / 1024 ** 3
                reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
                total = props.total_memory / 1024 ** 3
                gpu_info.append(
                    {"device": i, "name": props.name,
                     "allocated_gb": round(alloc, 2),
                     "reserved_gb": round(reserved, 2),
                     "total_gb": round(total, 2)}
                )
            snapshot["gpu"] = gpu_info
            logger.info("[%s] GPU: %s", label, gpu_info)
        else:
            snapshot["gpu"] = "no_cuda"
    except Exception as exc:
        snapshot["gpu"] = f"error: {exc}"

    # --- nvidia-smi (Linux / Kaggle) ---
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0:
            snapshot["nvidia_smi"] = proc.stdout.strip()
            logger.info("[%s] nvidia-smi:\n%s", label, snapshot["nvidia_smi"])
    except Exception:
        pass  # not available on Windows dev machines

    # --- RAM (Linux: /proc/meminfo; Windows: psutil fallback) ---
    try:
        if platform.system() == "Linux":
            mem_raw = Path("/proc/meminfo").read_text()
            mem_lines = {line.split(":")[0]: line.split(":")[1].strip()
                         for line in mem_raw.splitlines() if ":" in line}
            snapshot["ram"] = {
                "total": mem_lines.get("MemTotal"),
                "free": mem_lines.get("MemFree"),
                "available": mem_lines.get("MemAvailable"),
                "buffers": mem_lines.get("Buffers"),
                "cached": mem_lines.get("Cached"),
            }
        else:
            import psutil  # type: ignore
            vm = psutil.virtual_memory()
            snapshot["ram"] = {
                "total_gb": round(vm.total / 1024 ** 3, 1),
                "available_gb": round(vm.available / 1024 ** 3, 1),
                "percent_used": vm.percent,
            }
        logger.info("[%s] RAM: %s", label, snapshot["ram"])
    except Exception as exc:
        snapshot["ram"] = f"error: {exc}"

    # --- Top processes by memory (Linux: ps aux; Windows: psutil) ---
    try:
        if platform.system() == "Linux":
            proc = subprocess.run(
                ["ps", "aux", "--sort=-%mem"],
                capture_output=True, text=True, timeout=10,
            )
            lines = proc.stdout.splitlines()[:16]  # header + top 15
            snapshot["top_procs"] = "\n".join(lines)
        else:
            import psutil  # type: ignore
            procs = sorted(
                psutil.process_iter(["pid", "name", "memory_percent"]),
                key=lambda p: p.info["memory_percent"] or 0,
                reverse=True,
            )[:15]
            snapshot["top_procs"] = [
                {"pid": p.info["pid"], "name": p.info["name"],
                 "mem_%": round(p.info["memory_percent"] or 0, 2)}
                for p in procs
            ]
        logger.info("[%s] top_procs: %s", label, snapshot["top_procs"])
    except Exception as exc:
        snapshot["top_procs"] = f"error: {exc}"

    return snapshot


def run_ragas_eval(
    queries: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] | None = None,
    pipe: Any = None,
    hf_token: str | None = None,
    judge_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    provider: str | None = "together",
    max_workers: int = 2,
    timeout: int = 600,
) -> dict[str, float]:
    """Run RAGAS evaluation over a set of query/response pairs using 100% open-source models."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    data = {
        "question": queries,
        "answer": answers,
        "contexts": contexts,
    }
    if ground_truths:
        data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(data)
    judge_llm = get_ragas_judge(pipe=pipe, hf_token=hf_token, model_name=judge_model, provider=provider)
    judge_embeddings = get_ragas_embeddings(model_name=embedding_model)

    metrics = [faithfulness, answer_relevancy]
    if ground_truths:
        metrics.extend([context_precision, context_recall])

    run_config = RunConfig(
        timeout=timeout,
        max_workers=max_workers,
        max_retries=5,
        max_wait=30,
    )

    logger.info(
        "Running RAGAS evaluation with %d samples (max_workers=%d, timeout=%ds)...",
        len(queries), max_workers, timeout
    )
    try:
        results = evaluate(
            dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=run_config,
            raise_exceptions=False,
            show_progress=True,
        )
        out_dict = {}
        if hasattr(results, "to_pandas"):
            df = results.to_pandas()
            for col in df.columns:
                if col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "harmfulness", "semantic_similarity"]:
                    series = df[col].dropna()
                    if len(series) > 0:
                        out_dict[col] = round(float(series.mean()), 4)
            if not out_dict:
                means = df.mean(numeric_only=True).to_dict()
                for k, v in means.items():
                    if not (v != v):
                        out_dict[k] = round(float(v), 4)
        elif isinstance(results, dict):
            for k, v in results.items():
                try:
                    val = float(v)
                    if not (val != val):
                        out_dict[k] = round(val, 4)
                except (ValueError, TypeError):
                    pass
        
        if out_dict:
            return out_dict
        raise ValueError(f"No metric scores found in evaluation result: {results}")
    except Exception as exc:
        logger.error("RAGAS evaluate failed: %s", exc, exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on CAAR-CDSS")
    parser.add_argument("--benchmark", type=str, default="medqa", choices=["medqa", "pubmedqa", "seeds"])
    parser.add_argument("--n", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--mode", type=str, default="mock", choices=["mock", "kaggle_fp16", "hf_api", "both", "real", "local_4bit"])
    parser.add_argument("--output", "--output-dir", dest="output", type=str, default="experiments/results/ragas_results.json")
    parser.add_argument("--judge-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model to use for RAGAS judge")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-small-en-v1.5", help="Hugging Face embedding model for RAGAS")
    parser.add_argument("--provider", type=str, default="together", help="HF Inference API provider (together, fireworks, replicate, etc.)")
    parser.add_argument("--max-workers", type=int, default=2, help="Max concurrent evaluation workers (keep <= 2 for GPU)")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds per evaluation job")
    parser.add_argument("--gc-after-each", action="store_true", help="Run GPU memory + GC cleanup after each query (slower but prevents OOM)")
    args = parser.parse_args()

    # Normalise mode alias if both/real was passed
    if args.mode in ("both", "real"):
        args.mode = "kaggle_fp16"

    logging.basicConfig(level=logging.INFO)
    logger.info("Evaluating %d samples on %s benchmark (mode=%s)...", args.n, args.benchmark, args.mode)

    # Load benchmark items or seeds
    queries_data = []
    try:
        if args.benchmark == "seeds":
            from ..data.seed_corpus import get_seed_queries
            queries_data = get_seed_queries()[:args.n]
        else:
            from ..data.benchmarks import get_benchmark
            queries_data = get_benchmark(args.benchmark)[:args.n]
    except Exception as err:
        logger.warning("Could not load benchmark %s (%s). Using sample clinical queries.", args.benchmark, err)

    if queries_data:
        from ..confidence.pipeline import Pipeline
        # hf_api uses zero local models (all inference is remote); only kaggle_fp16
        # and local_4bit actually need a real local LLM loaded into memory.
        prefer_real = args.mode in ("kaggle_fp16", "local_4bit")
        pipeline_obj = Pipeline.from_seed(prefer_real=prefer_real)
        
        sample_queries = []
        sample_answers = []
        sample_contexts = []
        sample_ground_truths = []
        
        from ..utils.disk_utils import cleanup_after_step, format_progress

        logger.info("Generating CDSS responses for %d benchmark queries...", len(queries_data))
        t_start = time.perf_counter()
        for idx, item in enumerate(queries_data):
            q_text = item["query"]
            try:
                _, resp = pipeline_obj.analyze(q_text)
                sample_queries.append(q_text)
                ans = resp.reasoning if resp.reasoning else (resp.primary_diagnosis or "Diagnosis established.")
                sample_answers.append(ans)
                sample_contexts.append([c.text for c in resp.evidence] if resp.evidence else ["Guideline context."])
                sample_ground_truths.append(item.get("expected_dx", ""))
            except Exception:
                continue

            # Progress logging
            elapsed = time.perf_counter() - t_start
            if (idx + 1) % 5 == 0 or idx == len(queries_data) - 1:
                logger.info("Response gen: %s", format_progress(idx + 1, len(queries_data), elapsed))

            # Optional memory cleanup
            if getattr(args, "gc_after_each", False):
                cleanup_after_step(f"query-{idx+1}")
    else:
        sample_queries = ["What is the first-line treatment for community acquired pneumonia?"] * args.n
        sample_answers = ["First line treatment for uncomplicated community-acquired pneumonia in outpatients is amoxicillin or a macrolide."] * args.n
        sample_contexts = [["Amoxicillin 500mg TDS is recommended as first line for low-severity CAP according to NICE guidelines."]] * args.n
        sample_ground_truths = ["amoxicillin"] * args.n

    if args.mode == "mock":
        raise RuntimeError("Mock mode not supported for RAGAS evaluation. Use --mode hf_api, kaggle_fp16, or local_4bit for real evaluation.")
    elif args.mode == "kaggle_fp16":
        # On Kaggle: load fp16 model and reuse as judge (0 API calls, sequential worker=1 to prevent GPU timeout)
        import torch
        from transformers import pipeline, logging as hf_logging
        hf_logging.set_verbosity_error()

        logger.info("Loading %s in fp16 for KagglePipelineJudge...", args.judge_model)
        judge_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "max_new_tokens": 256,
            "return_full_text": False,
        }
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            num_gpus = torch.cuda.device_count()
            judge_kwargs["max_memory"] = {i: "9GiB" for i in range(num_gpus)}
        pipe = pipeline(
            "text-generation",
            model=args.judge_model,
            **judge_kwargs,
        )
        if hasattr(pipe, "tokenizer") and pipe.tokenizer and pipe.tokenizer.pad_token_id is None:
            pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

        res = run_ragas_eval(
            queries=sample_queries,
            answers=sample_answers,
            contexts=sample_contexts,
            ground_truths=sample_ground_truths if any(sample_ground_truths) else None,
            pipe=pipe,
            judge_model=args.judge_model,
            embedding_model=args.embedding_model,
            max_workers=1,  # Sequential GPU inference prevents GPU task queue timeout
            timeout=1800,  # 30 minutes generous timeout
        )
    elif args.mode == "hf_api":
        # Local dev: use HF Inference API (0 local VRAM)
        import os
        from dotenv import load_dotenv
        load_dotenv()
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN required for hf_api mode. Set in .env file.")
        res = run_ragas_eval(
            queries=sample_queries,
            answers=sample_answers,
            contexts=sample_contexts,
            ground_truths=sample_ground_truths if any(sample_ground_truths) else None,
            hf_token=hf_token,
            judge_model=args.judge_model,
            embedding_model=args.embedding_model,
            provider=args.provider,
            max_workers=args.max_workers,
            timeout=args.timeout,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    out_path = Path(args.output)
    if out_path.is_dir() or not out_path.suffix:
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / f"ragas_{args.benchmark}_results.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Safe-save with disk space pre-check
    from ..utils.disk_utils import safe_save
    try:
        safe_save(json.dumps(res, indent=2), out_path)
    except OSError as e:
        logger.error("Failed to save results: %s", e)
        # Print results to stdout as fallback
        print(json.dumps(res, indent=2))

    print("\n--- RAGAS Evaluation Results ---")
    for k, v in res.items():
        print(f"{k}: {v:.4f}")
    print(f"Saved results to: {out_path}\n")


if __name__ == "__main__":
    main()
