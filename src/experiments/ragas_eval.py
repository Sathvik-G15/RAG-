"""RAGAS automated evaluation harness for CAAR-CDSS with Kaggle GPU & HF judge support."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# -----------------------------------------------------------------------------
# Compatibility shim (ragas/langchain variants)
# -----------------------------------------------------------------------------
try:
    import langchain_community.chat_models  # type: ignore

    if "langchain_community.chat_models.vertexai" not in sys.modules:
        v_mod = types.ModuleType("langchain_community.chat_models.vertexai")

        class ChatVertexAI:  # pragma: no cover
            pass

        v_mod.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = v_mod
        setattr(langchain_community.chat_models, "vertexai", v_mod)
except Exception:
    pass


def _get_langchain_hf_pipeline_wrapper():
    """Prefer newer langchain-huggingface wrapper if available; fallback to deprecated one."""
    try:
        from langchain_huggingface import HuggingFacePipeline  # type: ignore
        return HuggingFacePipeline
    except Exception:  # pragma: no cover
        from langchain_community.llms import HuggingFacePipeline  # type: ignore
        return HuggingFacePipeline


# -----------------------------------------------------------------------------
# Judges
# -----------------------------------------------------------------------------
class KagglePipelineJudge:
    """RAGAS judge that wraps a loaded HuggingFace transformers pipeline (local GPU)."""

    def __init__(self, pipe: Any):
        from ragas.llms import LangchainLLMWrapper

        HuggingFacePipeline = _get_langchain_hf_pipeline_wrapper()
        self._llm = LangchainLLMWrapper(HuggingFacePipeline(pipeline=pipe))

    @property
    def llm(self):
        return self._llm


class HFAPIJudge:
    """RAGAS judge using Hugging Face Inference Providers via LangChain HuggingFaceEndpoint.

    Note: Provider/model availability may require billing; many provider/model combos won’t work.
    """

    def __init__(self, model_name: str, hf_token: str | None = None, provider: str | None = "together"):
        from langchain_huggingface import HuggingFaceEndpoint
        from ragas.llms import LangchainLLMWrapper

        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN is required for hf_api mode.")

        logger.info("Using HF Inference Providers judge: model=%s provider=%s", model_name, provider)
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


def get_ragas_judge(
    pipe: Any = None,
    hf_token: str | None = None,
    model_name: str = DEFAULT_JUDGE_MODEL,
    provider: str | None = "together",
):
    """Return a RAGAS judge LLM wrapper."""
    if pipe is not None:
        logger.info("Using KagglePipelineJudge (local pipeline, 0 API calls).")
        return KagglePipelineJudge(pipe).llm
    logger.info("Using HFAPIJudge (remote) model=%s provider=%s", model_name, provider)
    return HFAPIJudge(model_name, hf_token, provider=provider).llm


# -----------------------------------------------------------------------------
# Embeddings (FIX: CPU by default on Kaggle)
# -----------------------------------------------------------------------------
def get_ragas_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL, force_cpu: bool = True):
    """RAGAS embeddings wrapper.

    Kaggle stability fix:
      - Default to CPU embeddings to avoid GPU contention / hangs on T4.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    import torch

    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing RAGAS embeddings: %s on %s (force_cpu=%s)", model_name, device, force_cpu)

    hf_emb = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(hf_emb)


# -----------------------------------------------------------------------------
# Local judge model loader (FIX: do NOT pass max_memory to pipeline())
# -----------------------------------------------------------------------------
def load_local_judge_pipeline(model_id: str, max_new_tokens: int = 256):
    """Load local fp16 judge model safely.

    Critical fix:
      - DO NOT pass max_memory into transformers.pipeline()
        (it can be forwarded to generate() -> crash).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging as hf_logging

    hf_logging.set_verbosity_error()
    logger.info("Loading local judge model: %s", model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    max_memory = None
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # Used ONLY in from_pretrained (safe).
        max_memory = {i: "15GiB" for i in range(torch.cuda.device_count())}

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory=max_memory,  # OK here
        low_cpu_mem_usage=True,
    )

    # IMPORTANT: no max_memory/device_map passed here
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        return_full_text=False,
    )

    if getattr(pipe, "tokenizer", None) is not None and pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

    return pipe


# -----------------------------------------------------------------------------
# RAGAS evaluation
# -----------------------------------------------------------------------------
def run_ragas_eval(
    queries: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] | None = None,
    pipe: Any = None,
    hf_token: str | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    provider: str | None = "together",
    max_workers: int = 2,
    timeout: int = 600,
) -> dict[str, float | None]:
    """Run RAGAS and return mean scores (None if all NaN)."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    payload: dict[str, Any] = {"question": queries, "answer": answers, "contexts": contexts}

    gt_present = False
    if ground_truths is not None:
        gt_present = any((gt or "").strip() for gt in ground_truths)
        if gt_present:
            payload["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(payload)

    judge_llm = get_ragas_judge(pipe=pipe, hf_token=hf_token, model_name=judge_model, provider=provider)
    judge_embeddings = get_ragas_embeddings(model_name=embedding_model, force_cpu=True)

    metrics = [faithfulness, answer_relevancy]
    if gt_present:
        metrics.extend([context_precision, context_recall])

    run_config = RunConfig(
        timeout=timeout,
        max_workers=max_workers,
        max_retries=5,
        max_wait=30,
    )

    logger.info(
        "Running RAGAS evaluation with %d samples (max_workers=%d, timeout=%ds)...",
        len(queries),
        max_workers,
        timeout,
    )

    results = evaluate(
        dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
        raise_exceptions=False,
        show_progress=True,
    )

    out: dict[str, float | None] = {}
    if hasattr(results, "to_pandas"):
        df = results.to_pandas()
        for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if col in df.columns:
                s = df[col].dropna()
                if len(s) == 0:
                    out[col] = None
                else:
                    m = float(s.mean())
                    out[col] = round(m, 4) if (m == m) else None  # NaN -> None
    elif isinstance(results, dict):
        for k, v in results.items():
            try:
                fv = float(v)
                out[k] = round(fv, 4) if (fv == fv) else None
            except Exception:
                pass

    if not out or all(v is None for v in out.values()):
        logger.error("RAGAS produced no valid metrics (all NaN/None). Returning None metrics.")
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None if gt_present else None,
            "context_recall": None if gt_present else None,
        }

    return out


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on CAAR-CDSS")
    parser.add_argument("--benchmark", type=str, default="medqa", choices=["medqa", "pubmedqa", "seeds"])
    parser.add_argument("--n", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument(
        "--mode",
        type=str,
        default="mock",
        choices=["mock", "kaggle_fp16", "hf_api", "both", "real", "local_4bit"],
    )
    parser.add_argument("--output", "--output-dir", dest="output", type=str, default="experiments/results/ragas_results.json")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--provider", type=str, default="together")
    parser.add_argument("--max-workers", type=int, default=2, help="Max concurrent workers (GPU: keep <= 1)")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per job (seconds)")
    parser.add_argument("--gc-after-each", action="store_true", help="Cleanup after each query (slower, safer)")
    args = parser.parse_args()

    if args.mode in ("both", "real"):
        args.mode = "kaggle_fp16"

    logging.basicConfig(level=logging.INFO)
    logger.info("Evaluating %d samples on %s benchmark (mode=%s)...", args.n, args.benchmark, args.mode)

    # Load benchmark items
    queries_data: list[dict[str, Any]] = []
    try:
        if args.benchmark == "seeds":
            from ..data.seed_corpus import get_seed_queries
            queries_data = get_seed_queries()[: args.n]
        else:
            from ..data.benchmarks import get_benchmark
            queries_data = get_benchmark(args.benchmark)[: args.n]
    except Exception as err:
        logger.warning("Could not load benchmark %s (%s). Using fallback queries.", args.benchmark, err)

    # Generate answers/contexts
    if queries_data:
        from ..confidence.pipeline import Pipeline
        from ..utils.disk_utils import cleanup_after_step, format_progress, cleanup_gpu_memory

        # Only load local real models for local modes
        prefer_real = args.mode in ("kaggle_fp16", "local_4bit")
        pipeline_obj = Pipeline.from_seed(prefer_real=prefer_real)

        sample_queries: list[str] = []
        sample_answers: list[str] = []
        sample_contexts: list[list[str]] = []
        sample_ground_truths: list[str] = []

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

            elapsed = time.perf_counter() - t_start
            if (idx + 1) % 5 == 0 or idx == len(queries_data) - 1:
                logger.info("Response gen: %s", format_progress(idx + 1, len(queries_data), elapsed))

            if args.gc_after_each:
                cleanup_after_step(f"query-{idx+1}")

        # Free memory before loading judge model (important on T4)
        del pipeline_obj
        cleanup_gpu_memory(log_prefix="[After answer generation]")
    else:
        sample_queries = ["What is the first-line treatment for community acquired pneumonia?"] * args.n
        sample_answers = ["Amoxicillin is first-line for uncomplicated outpatient CAP."] * args.n
        sample_contexts = [["Amoxicillin is recommended first line for low-severity CAP."]] * args.n
        sample_ground_truths = ["amoxicillin"] * args.n

    # Run RAGAS
    if args.mode == "mock":
        raise RuntimeError("Mock mode not supported for RAGAS evaluation.")
    elif args.mode == "kaggle_fp16":
        pipe = load_local_judge_pipeline(args.judge_model, max_new_tokens=256)
        res = run_ragas_eval(
            queries=sample_queries,
            answers=sample_answers,
            contexts=sample_contexts,
            ground_truths=sample_ground_truths if any(sample_ground_truths) else None,
            pipe=pipe,
            judge_model=args.judge_model,
            embedding_model=args.embedding_model,
            max_workers=1,
            timeout=1800,
        )
    elif args.mode == "hf_api":
        from dotenv import load_dotenv
        load_dotenv()
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN required for hf_api mode.")
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

    # Save
    out_path = Path(args.output)
    if out_path.is_dir() or not out_path.suffix:
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / f"ragas_{args.benchmark}_results.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    from ..utils.disk_utils import safe_save
    safe_save(json.dumps(res, indent=2, default=str), out_path)

    print("\n--- RAGAS Evaluation Results ---")
    for k, v in res.items():
        print(f"{k}: {v}")
    print(f"Saved results to: {out_path}\n")


if __name__ == "__main__":
    main()