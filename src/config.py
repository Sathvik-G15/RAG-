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


class AEBConfig(BaseModel):
    initial_k: int = 3
    step_k: int = 3
    max_k: int = 15
    confidence_threshold: float = 0.85
    stop_on_budget_exhausted: bool = True


class RetrievalConfig(BaseModel):
    dense_top_k: int = 50
    sparse_top_k: int = 50
    rerank_top_k: int = 15
    final_top_k: int = 10


class VerificationConfig(BaseModel):
    nli_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    entailment_threshold: float = 0.7
    similarity_threshold: float = 0.75


class LLMConfig(BaseModel):
    model_name: str = "microsoft/Llama3-Med-8B-Instruct"
    serving_backend: str = "local_4bit"
    load_in_4bit: bool = True
    bnb_double_quant: bool = True
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.95


class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-m3"
    chunk_size: int = 512
    chunk_overlap: int = 100


class AppConfig(BaseModel):
    aeb: AEBConfig = Field(default_factory=AEBConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)


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

    return AppConfig(
        aeb=AEBConfig(**seeds.get("aeb", {})),
        retrieval=RetrievalConfig(**seeds.get("retrieval", {})),
        verification=VerificationConfig(**seeds.get("verification", {})),
        llm=LLMConfig(**seeds.get("llm", {})),
        embedding=EmbeddingConfig(**seeds.get("embedding", {})),
    )


def detect_hardware() -> dict[str, Any]:
    """Auto-detect available hardware, VRAM, and free disk space to select optimal backends."""
    import shutil

    # Disk space check
    free_gb = shutil.disk_usage(".").free / (1024**3)
    disk_warning = None
    if free_gb < 25.0:
        disk_warning = f"Warning: {free_gb:.1f} GB free disk space (recommend >= 25 GB for full corpus + models)."

    try:
        import torch
        if not torch.cuda.is_available():
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

        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb >= 20.0:
            # Multi-GPU / A100 / Kaggle 2xT4
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
        elif vram_gb >= 12.0:
            # Single T4 (16 GB) on Kaggle
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
            # Local RTX 4050 6 GB (or similar <10 GB VRAM GPU)
            return {
                "device": "cuda",
                "vram_gb": round(vram_gb, 1),
                "free_disk_gb": round(free_gb, 1),
                "disk_warning": disk_warning,
                "serving_backend": "local_4bit",
                "verification_backend": "hf_inference_api",
                "ragas_judge": "hf_inference_api",
                "max_new_tokens": 256,
            }
    except ImportError:
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

