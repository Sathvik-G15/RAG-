"""Bounded boosting on normalized RRF scores, plus rank-displacement measurement."""
from __future__ import annotations

from typing import Dict, List, Sequence


def boost_and_rerank(chunks: List[Dict], alpha: float = 0.2, beta: float = 0.2) -> List[Dict]:
    """chunks: dicts with rrf_score, trust_score (0-1), patient_match (0-1).
    RRF scores are normalized to [0,1] first so boosts are comparable across queries.
    Boost factor is bounded to [1-alpha/2, 1+alpha/2] * [1, 1+beta]."""
    if not chunks:
        return []
    top = max(c["rrf_score"] for c in chunks) or 1.0
    out = []
    for c in chunks:
        norm = c["rrf_score"] / top
        trust = 1 + alpha * (c.get("trust_score", 0.5) - 0.5)
        patient = 1 + beta * c.get("patient_match", 0.0)
        out.append({**c, "rrf_norm": norm, "final_score": norm * trust * patient})
    return sorted(out, key=lambda c: -c["final_score"])


def rank_displacement(before_ids: Sequence[str], after_ids: Sequence[str]) -> float:
    """Mean absolute rank change, normalized by list length. 0 = identical order."""
    pos = {d: i for i, d in enumerate(after_ids)}
    shared = [d for d in before_ids if d in pos]
    if not shared:
        return 0.0
    return sum(abs(i - pos[d]) for i, d in enumerate(before_ids) if d in pos) / (len(shared) * max(len(shared) - 1, 1))
