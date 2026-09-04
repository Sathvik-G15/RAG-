"""Sparse retrieval via BM25 (tokenized English index)."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable

from ..models import EvidenceChunk

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]+|[A-Za-z]+")
_STOPWORDS = frozenset(
    "a an the of and or to in on for with at by from is are was were be been being this that these those "
    "as if then than but not no nor so such into out over under up down off again further here there all any "
    "both each few more most other some such only own same s t can will shall may might must should would could"
    .split()
)


def tokenize(text: str) -> list[str]:
    tokens = []
    for m in _TOKEN_RE.findall(text.lower()):
        if m not in _STOPWORDS and len(m) > 1:
            tokens.append(m)
    return tokens


class BM25Index:
    """Standard Okapi BM25 over in-memory documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, idf_floor: float = 0.0):
        self.k1 = k1
        self.b = b
        self.idf_floor = idf_floor
        self._chunks: list[EvidenceChunk] = []
        self._doc_tokens: list[list[str]] = []
        self._df: dict[str, int] = defaultdict(int)
        self._tf: list[Counter] = []
        self._doc_len: list[int] = []
        self._avg_dl: float = 0.0
        self._n: int = 0

    def add_chunks(self, chunks: Iterable[EvidenceChunk]) -> None:
        for ch in chunks:
            tokens = tokenize(ch.text + " " + (ch.title or ""))
            self._chunks.append(ch)
            self._doc_tokens.append(tokens)
            tf = Counter(tokens)
            self._tf.append(tf)
            self._doc_len.append(len(tokens))
            for term in tf.keys():
                self._df[term] += 1
        self._n = len(self._chunks)
        self._avg_dl = (sum(self._doc_len) / self._n) if self._n else 0.0

    @property
    def size(self) -> int:
        return self._n

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        if df == 0:
            return self.idf_floor
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> list[tuple[EvidenceChunk, float]]:
        if self._n == 0:
            return []
        q_terms = tokenize(query)
        results: list[tuple[EvidenceChunk, float]] = []
        for i, ch in enumerate(self._chunks):
            tf = self._tf[i]
            dl = self._doc_len[i]
            denom_k = self.k1 * (1 - self.b + self.b * (dl / max(self._avg_dl, 1.0)))
            s = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = self._idf(term)
                s += idf * (f * (self.k1 + 1)) / (f + denom_k)
            results.append((ch, float(s)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def search(self, query: str, top_k: int = 10) -> list[tuple[EvidenceChunk, float]]:
        return self.score(query)[:top_k]
