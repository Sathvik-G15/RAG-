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
# Compatibility shim: some ragas/langchain combos import ChatVertexAI
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
    """Prefer new langchain-huggingface wrapper if available; fallback to deprecated one."""
    try:
        from langchain_huggingface import HuggingFacePipeline  # type: ignore
        return HuggingFacePipeline
    except Exception:  # pragma: no cover
        from langchain_community.llms import HuggingFacePipeline  # type: ignore
        return HuggingFacePipeline


class KagglePipelineJudge:
    """RAGAS judge that wraps a loaded transformers pipeline (local GPU)."""

    def __init__(self, pipe: Any):
        from ragas.llms import LangchainLLMWrapper
        HuggingFacePipeline = _get_langchain_hf_pipeline_wrapper()
        self._llm = LangchainLLMWrapper(HuggingFacePipeline(pipeline=pipe))

    @property
    def llm(self):
        return self._llm


class HFAPIJudge:
    """RAGAS judge via HF Inference Providers (may require billing/availability)."""

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
    if pipe is not None:
        logger.info("Using KagglePipelineJudge (local pipeline, 0 API calls).")
        return KagglePipelineJudge(pipe).llm
    logger.info("Using HFAPIJudge (remote) model=%s provider=%s", model_name, provider)
    return HFAPIJudge(model_name, hf_token, provider=provider).llm


def get_ragas_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL, force_cpu: bool = True):
    """CPU-by-default on Kaggle to avoid VRAM contention/hangs."""
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


def load_local_judge_pipeline(model_id: str, load_in_4bit: bool = True, max_new_tokens: int = 512):
    """Local 4-bit / fp16 judge model. Does NOT pass max_memory into pipeline()."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging as hf_logging

    hf_logging.set_verbosity_error()
    logger.info("Loading local judge model: %s (load_in_4bit=%s)", model_id, load_in_4bit)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if load_in_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        max_memory = None
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            max_memory = {i: "15GiB" for i in range(torch.cuda.device_count())}

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        return_full_text=False,
    )
    if getattr(pipe, "tokenizer", None) is not None and pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

    for target in (pipe, model):
        gen_cfg = getattr(target, "generation_config", None)
        if gen_cfg is not None:
            if hasattr(gen_cfg, "max_memory"):
                delattr(gen_cfg, "max_memory")
            if hasattr(gen_cfg, "_extra_kwargs") and isinstance(gen_cfg._extra_kwargs, dict):
                gen_cfg._extra_kwargs.pop("max_memory", None)
        for attr in ("model_kwargs", "_forward_params", "_preprocess_params", "_postprocess_params", "_extra_kwargs"):
            d = getattr(target, attr, None)
            if isinstance(d, dict):
                d.pop("max_memory", None)

    return pipe


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

    run_config = RunConfig(timeout=timeout, max_workers=max_workers, max_retries=5, max_wait=30)

    logger.info(
        "Running RAGAS evaluation with %d samples (max_workers=%d, timeout=%ds)...",
        len(queries), max_workers, timeout
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
                    out[col] = round(m, 4) if (m == m) else None
    elif isinstance(results, dict):
        for k, v in results.items():
            try:
                fv = float(v)
                out[k] = round(fv, 4) if (fv == fv) else None
            except Exception:
                pass

    if not out or all(v is None for v in out.values()):
        logger.error("RAGAS produced no valid metrics (all NaN/None). Returning None metrics.")
        return {"faithfulness": None, "answer_relevancy": None, "context_precision": None, "context_recall": None}

    return out


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on CAAR-CDSS")
    parser.add_argument("--benchmark", type=str, default="medqa", choices=["medqa", "pubmedqa", "seeds"])
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--mode", type=str, default="mock",
                        choices=["mock", "kaggle_fp16", "hf_api", "both", "real", "local_4bit"])
    parser.add_argument("--output", "--output-dir", dest="output", type=str, default="experiments/results/ragas_results.json")
    parser.add_argument("--chroma-dir", type=str, default=None, help="Path to pre-ingested ChromaDB directory")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--provider", type=str, default="together")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--gc-after-each", action="store_true")
    args = parser.parse_args()

    if args.mode in ("both", "real"):
        args.mode = "local_4bit"

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

    # Generate answers/contexts with CDSS
    if queries_data:
        from ..confidence.pipeline import Pipeline
        from ..utils.disk_utils import cleanup_after_step, format_progress, cleanup_gpu_memory

        prefer_real = args.mode in ("kaggle_fp16", "local_4bit")
        if args.chroma_dir and Path(args.chroma_dir).exists():
            logger.info("Loading Pipeline from ChromaDB: %s", args.chroma_dir)
            pipeline_obj = Pipeline.from_corpus(chroma_dir=args.chroma_dir, prefer_real=prefer_real)
        else:
            logger.info("Loading Pipeline from Seed Corpus.")
            pipeline_obj = Pipeline.from_seed(prefer_real=prefer_real)

        sample_queries, sample_answers, sample_contexts, sample_ground_truths = [], [], [], []

        logger.info("Generating CDSS responses for %d benchmark queries...", len(queries_data))
        t0 = time.perf_counter()
        for idx, item in enumerate(queries_data):
            q = item["query"]
            try:
                _, resp = pipeline_obj.analyze(q)
                sample_queries.append(q)
                sample_answers.append(resp.reasoning or resp.primary_diagnosis or "Diagnosis established.")
                sample_contexts.append([c.text for c in resp.evidence] if resp.evidence else ["Guideline context."])
                sample_ground_truths.append(item.get("expected_dx", ""))
            except Exception:
                continue

            if (idx + 1) % 5 == 0 or idx == len(queries_data) - 1:
                logger.info("Response gen: %s", format_progress(idx + 1, len(queries_data), time.perf_counter() - t0))

            if args.gc_after_each:
                cleanup_after_step(f"query-{idx+1}")

        # Free memory before judge load
        del pipeline_obj
        cleanup_gpu_memory(log_prefix="[After answer generation]")
    else:
        sample_queries = ["What is the first-line treatment for community acquired pneumonia?"] * args.n
        sample_answers = ["Amoxicillin is first-line for uncomplicated outpatient CAP."] * args.n
        sample_contexts = [["Amoxicillin is recommended first line for low-severity CAP."]] * args.n
        sample_ground_truths = ["amoxicillin"] * args.n

    # Evaluate
    if args.mode in ("kaggle_fp16", "local_4bit"):
        load_in_4bit = (args.mode == "local_4bit")
        pipe = load_local_judge_pipeline(args.judge_model, load_in_4bit=load_in_4bit, max_new_tokens=512)
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
        raise RuntimeError(f"Unsupported mode: {args.mode}. Use --mode local_4bit, kaggle_fp16, or hf_api.")

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