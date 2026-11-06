"""AEB stopping policy: patience, minimum steps, verification gate, query reformulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple


@dataclass
class StoppingPolicy:
    threshold: float
    min_steps: int = 2            # never stop on the first round
    patience: int = 2             # confidence must stay >= threshold this many consecutive rounds
    max_contradiction_rate: float = 0.10
    max_k: int = 15

    def should_stop(self, conf_curve: Sequence[float], current_k: int,
                    contradiction_rate: float, verifiable: bool) -> Tuple[bool, str]:
        """Returns (stop, reason). reason in {threshold, budget_exhausted, continue, gated}."""
        n = len(conf_curve)
        if n >= self.min_steps and n >= self.patience and all(c >= self.threshold for c in conf_curve[-self.patience:]):
            if not verifiable or contradiction_rate > self.max_contradiction_rate:
                if current_k >= self.max_k:
                    return True, "budget_exhausted"
                return False, "gated"
            return True, "threshold"
        if current_k >= self.max_k:
            return True, "budget_exhausted"
        return False, "continue"


def reformulate_query(query: str, differential: Sequence[Tuple[str, float]], top_n: int = 2) -> str:
    """Add current top diagnoses to the query so later rounds retrieve targeted evidence."""
    names = [n for n, _ in sorted(differential, key=lambda x: -x[1])[:top_n]]
    return query if not names else f"{query} | consider: " + "; ".join(names)


def drop_rate(conf_curve: Sequence[float]) -> float:
    """Fraction of rounds where confidence fell. Report this in the paper."""
    if len(conf_curve) < 2:
        return 0.0
    return sum(b < a for a, b in zip(conf_curve, conf_curve[1:])) / (len(conf_curve) - 1)
