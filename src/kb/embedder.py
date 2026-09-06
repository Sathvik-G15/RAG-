"""Embedding generation with sentence-transformers, with deterministic fallback."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from ..config import set_global_seeds


class BaseEmbedder:
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError


class HashEmbedder(BaseEmbedder):
    """Deterministic, dependency-free fallback embedder.

    Implements the hashing trick (feature hashing with signed terms) to produce
    L2-normalized bag-of-tokens vectors. Unlike a random-projection embedder,
    token overlap is preserved, so cosine similarity behaves sensibly for
    keyword-heavy clinical queries. Use for tests, CI, and when no GPU/internet
    is available.
    """

    def __init__(self, dim: int = 512, seed: int = 42):
        self._dim = dim
        self._seed = seed

    @property
    def dim(self) -> int:
        return self._dim

    @staticmethod
    def _token_hash(token: str) -> int:
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)

    def _features(self, text: str) -> dict[int, float]:
        counts: dict[int, float] = {}
        for tok in text.lower().split():
            h = self._token_hash(tok)
            idx = h % self._dim
            sign = 1.0 if (h >> 31) & 1 else -1.0
            counts[idx] = counts.get(idx, 0.0) + sign
        return counts

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for idx, val in self._features(t).items():
                out[i, idx] += val
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return out / norms


class SentenceTransformerEmbedder(BaseEmbedder):
    """Real sentence-transformers embedder (biomedical model recommended)."""

    def __init__(self, model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        set_global_seeds(42)
        self._model = SentenceTransformer(model_name, device=device)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(list(texts), batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
        return vecs.astype(np.float32)


class BGE_M3Embedder(BaseEmbedder):
    """BGE-M3 (BAAI/bge-m3) hybrid-capable embedder via FlagEmbedding or sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None, use_fp16: bool = True):
        self._dim = 1024
        self._use_flag = False
        try:
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)
            self._use_flag = True
        except ImportError:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name, device=device)
            self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._use_flag:
            res = self._model.encode(list(texts), batch_size=128, return_dense=True)
            vecs = res["dense_vecs"]
            return np.array(vecs, dtype=np.float32)
        vecs = self._model.encode(list(texts), batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
        return vecs.astype(np.float32)


def make_embedder(model_name: str | None = None, prefer_real: bool = True) -> BaseEmbedder:
    """Factory that picks the real embedder if available, else hash fallback."""
    if prefer_real and model_name:
        if "bge-m3" in model_name.lower():
            try:
                return BGE_M3Embedder(model_name=model_name)
            except Exception:
                pass
        try:
            return SentenceTransformerEmbedder(model_name=model_name)
        except Exception:
            pass
    return HashEmbedder()

