"""Configuration loading and management."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"


class ConfigError(Exception):
    """Raised when configuration loading fails."""


def load_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_seeds_config() -> dict[str, Any]:
    return load_yaml(CONFIGS_DIR / "seeds.yaml")


def get_kb_config() -> dict[str, Any]:
    return load_yaml(CONFIGS_DIR / "kb.yaml")


# Kaggle-safe defaults: paper-quality settings for full experiments
# For smoke tests, override via CLI: --n 50 --benchmarks medqa
KAGGLE_SAFE_DEFAULTS = {
    "n_queries": 1273,             # Full MedQA test set (1,273 items)
    "max_new_tokens": 512,         # Full reasoning + JSON output (prevents truncation)
    "ragas_max_workers": 1,        # Sequential GPU inference prevents OOM
    "ragas_timeout": 3600,         # 60 min timeout per RAGAS job
    "ingest_limit": 5000,          # Full guideline corpus
    "benchmarks": ["medqa", "pubmedqa", "medmcqa"],  # All major benchmarks
    "skip_threshold_sweep": False, # Run threshold sweep for abstention curve
    "skip_embedding_ablation": False,
    "hf_cache_keep_models": [      # Models to keep when cleaning HF cache
        "Qwen/Qwen2.5-1.5B-Instruct",
        "BAAI/bge-small-en-v1.5",
        "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    ],
    "log_truncation_rate": True,   # Log finish_reason == "length" as wrong

}


class AEBConfig(BaseModel):
    initial_k: int = 3
    step_k: int = 3
    max_k: int = 15
    # No default - MUST be set in configs/seeds.yaml to ensure reproducibility.
    # If missing, get_app_config() will raise ConfigError.
    confidence_threshold: float
    stop_on_budget_exhausted: bool = True
    max_rounds: int = 5  # Hard cap on AEB loop iterations (initial_k + max_rounds*step_k = max_k)
    # Hardening parameters
    min_steps: int = 2              # Never stop on round 1
    patience: int = 2               # Threshold must hold for this many consecutive rounds
    max_contradiction_rate: float = 0.10  # Verification gate
    reformulate_query: bool = True  # Add top diagnoses to query for next round


class RetrievalConfig(BaseModel):
    dense_top_k: int = 50
    sparse_top_k: int = 50
    rerank_top_k: int = 15
    final_top_k: int = 10
    # Hardening parameters (bounded normalized boosting)
    boost_alpha: float = 0.2
    boost_beta: float = 0.2


class VerificationConfig(BaseModel):
    nli_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    entailment_threshold: float = 0.7
    similarity_threshold: float = 0.75
    # Hardening parameters
    backend: str = "deberta"        # Headline results must use deberta; lexical = CI smoke test only
    entail_thr: float = 0.70        # Tune on labelled data
    contra_thr: float = 0.70
    allow_fallback: bool = False    # Fail loudly instead of silently using lexical


class TriageConfig(BaseModel):
    # Legacy confidence-only bands (kept for backward compat)
    urgent_confidence: float = 0.70
    routine_confidence: float = 0.85
    home_care_confidence: float = 0.85
    # Hardening parameters (new severity-aware triage)
    high_confidence: float = 0.85
    low_confidence: float = 0.70
    cantmiss_emergency_prob: float = 0.20  # p >= this for emergent dx -> EMERGENCY
    cantmiss_urgent_prob: float = 0.05     # p >= this for any can't-miss dx -> >= URGENT
    low_confidence_risk: str = "ROUTINE"   # Acuity when only confidence is low


class LLMConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str = "microsoft/Llama3-Med-8B-Instruct"
    serving_backend: str = "local_4bit"
    load_in_4bit: bool = True
    bnb_double_quant: bool = True
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.95


class RagasJudgeConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str = "microsoft/Llama3-Med-8B-Instruct"


class EmbeddingConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 512
    chunk_overlap: int = 100


class SecurityConfig(BaseModel):
    upload_requires_role: str = "admin"
    rate_limit_analyze: list[int] = [30, 60]      # calls, seconds
    rate_limit_eval: list[int] = [2, 3600]        # calls, seconds
    external_inference_allowed: bool = False      # Blocks HFInferenceReasoner unless data is synthetic
    strict_mode: bool = False                     # CAAR_STRICT=1: fail loudly on any fallback


class AppConfig(BaseModel):
    aeb: AEBConfig = Field(default_factory=AEBConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    triage: TriageConfig = Field(default_factory=TriageConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ragas_judge: RagasJudgeConfig = Field(default_factory=RagasJudgeConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)


def get_app_config() -> AppConfig:
    seeds = get_seeds_config()

    # Overlay .env overrides onto seeds (model names only)
    import os
    env_overrides = {
        "llm": {"model_name": os.environ.get("CAAR_LLM_MODEL")},
        "embedding": {"model_name": os.environ.get("CAAR_EMBEDDING_MODEL")},
        "verification": {"nli_model": os.environ.get("CAAR_NLI_MODEL")},
        "ragas_judge": {"model_name": os.environ.get("CAAR_RAGAS_JUDGE_MODEL")},
    }

    # Apply non-None overrides
    for section, overrides in env_overrides.items():
        if section not in seeds:
            seeds[section] = {}
        for key, value in overrides.items():
            if value is not None:
                seeds[section][key] = value

    # Validate required fields that have no defaults
    if "aeb" not in seeds or "confidence_threshold" not in seeds["aeb"]:
        raise ConfigError("Missing required config: aeb.confidence_threshold must be set in configs/seeds.yaml")

    return AppConfig(
        aeb=AEBConfig(**seeds.get("aeb", {})),
        retrieval=RetrievalConfig(**seeds.get("retrieval", {})),
        verification=VerificationConfig(**seeds.get("verification", {})),
        triage=TriageConfig(**seeds.get("triage", {})),
        llm=LLMConfig(**seeds.get("llm", {})),
        ragas_judge=RagasJudgeConfig(**seeds.get("ragas_judge", {})),
        embedding=EmbeddingConfig(**seeds.get("embedding", {})),
        security=SecurityConfig(**seeds.get("security", {})),
    )


def detect_hardware() -> dict[str, Any]:
    """Auto-detect available hardware, VRAM, and free disk space to select optimal backends.

    Hardware table (from hardening guide):
    - >=20 GB VRAM -> kaggle_fp16
    - 10-19.9 GB -> kaggle_fp16 on Kaggle else local_4bit
    - <10 GB -> local_4bit
    - No GPU -> hf_inference_api only if external_inference_allowed, otherwise mock (tests) or raise
    """
    import shutil

    # Disk space check
    free_gb = shutil.disk_usage(".").free / (1024**3)
    disk_warning = None
    if free_gb < 2.0:
        disk_warning = f"🛑 CRITICAL: {free_gb:.1f} GB free disk space — high risk of ENOSPC errors! Clean HF cache or reduce workload."
    elif free_gb < 5.0:
        disk_warning = f"⚠️  Low disk: {free_gb:.1f} GB free. Expect issues with large model downloads. Consider cleanup_hf_cache()."
    elif free_gb < 25.0:
        disk_warning = f"ℹ️  {free_gb:.1f} GB free (recommend >= 25 GB for full corpus + models)."

    try:
        import torch
        if not torch.cuda.is_available():
            # No GPU - check if external inference allowed
            cfg = get_app_config()
            if cfg.security.external_inference_allowed:
                return {
                    "device": "cpu",
                    "vram_gb": 0.0,
                    "free_disk_gb": round(free_gb, 1),
                    "disk_warning": disk_warning,
                    "serving_backend": "hf_inference_api",
                    "verification_backend": "hf_inference_api",
                    "ragas_judge": "hf_inference_api",
                    "max_new_tokens": 256,
                }
            else:
                # For tests, use mock; for production, raise
                return {
                    "device": "cpu",
                    "vram_gb": 0.0,
                    "free_disk_gb": round(free_gb, 1),
                    "disk_warning": disk_warning,
                    "serving_backend": "mock",
                    "verification_backend": "local",
                    "ragas_judge": "local",
                    "max_new_tokens": 256,
                }

        num_gpus = torch.cuda.device_count()
        vram_gb = sum(
            torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)
        ) / (1024**3)
        if vram_gb >= 20.0:
            # Multi-GPU (e.g. Kaggle 2xT4 = ~30GB) / A100
            return {
                "device": "cuda",
                "vram_gb": round(vram_gb, 1),
                "free_disk_gb": round(free_gb, 1),
                "disk_warning": disk_warning,
                "serving_backend": "kaggle_fp16",
                "verification_backend": "hf_inference_api",
                "ragas_judge": "kaggle_pipeline",
                "max_new_tokens": 512,
            }
        elif vram_gb >= 10.0:
            # 10-19.9 GB - Kaggle T4 (16 GB) or similar
            # Check if we're on Kaggle (heuristic: multiple GPUs or specific env)
            import os
            on_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None or num_gpus > 1
            if on_kaggle:
                return {
                    "device": "cuda",
                    "vram_gb": round(vram_gb, 1),
                    "free_disk_gb": round(free_gb, 1),
                    "disk_warning": disk_warning,
                    "serving_backend": "kaggle_fp16",
                    "verification_backend": "hf_inference_api",
                    "ragas_judge": "kaggle_pipeline",
                    "max_new_tokens": 512,
                }
            else:
                # Local GPU with 10-19 GB - use 4-bit to be safe
                return {
                    "device": "cuda",
                    "vram_gb": round(vram_gb, 1),
                    "free_disk_gb": round(free_gb, 1),
                    "disk_warning": disk_warning,
                    "serving_backend": "local_4bit",
                    "verification_backend": "local",
                    "ragas_judge": "kaggle_pipeline",
                    "max_new_tokens": 256,
                }
        else:
            # Local GPU < 10 GB VRAM (e.g. RTX 4050 6 GB).
            # Run everything locally: LLM in 4-bit, DeBERTa verifier on CUDA:0,
            # RAGAS judge via local pipeline.  No cloud API calls needed.
            return {
                "device": "cuda",
                "vram_gb": round(vram_gb, 1),
                "free_disk_gb": round(free_gb, 1),
                "disk_warning": disk_warning,
                "serving_backend": "local_4bit",
                "verification_backend": "local",
                "ragas_judge": "kaggle_pipeline",
                "max_new_tokens": 256,
            }
    except ImportError:
        return {
            "device": "cpu",
            "vram_gb": 0.0,
            "free_disk_gb": round(free_gb, 1),
            "disk_warning": disk_warning,
            "serving_backend": "local_4bit",
            "verification_backend": "local",
            "ragas_judge": "local",
            "max_new_tokens": 256,
        }


def set_global_seeds(seed: int = 42) -> None:
    import random

    import numpy as np

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    random.seed(seed)
    np.random.seed(seed)