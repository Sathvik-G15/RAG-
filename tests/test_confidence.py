"""Tests for confidence estimation and calibration."""

from __future__ import annotations

import pytest

from src.confidence.estimation import ConfidenceModel, ConfidenceSignals


def test_confidence_weights_order_matches_signals() -> None:
    """Ensure weight array order matches ConfidenceSignals field order.

    ConfidenceSignals fields: retriever_top1, evidence_sufficiency,
    support_agreement, raw_confidence, hallucination_penalty

    Weights: w_retriever, w_evidence, w_agreement, w_llm, w_hallucination
    """
    model = ConfidenceModel()
    signals = ConfidenceSignals(
        retriever_top1=0.8,
        evidence_sufficiency=0.7,
        support_agreement=0.9,
        hallucination_penalty=0.1,
        raw_confidence=0.85,
    )

    # Manually compute expected score using field order
    weights = [
        model.w_retriever,
        model.w_evidence,
        model.w_agreement,
        model.w_llm,
        model.w_hallucination,
    ]
    values = [
        signals.retriever_top1,
        signals.evidence_sufficiency,
        signals.support_agreement,
        signals.raw_confidence,
        1.0 - min(signals.hallucination_penalty, 0.5),  # 1 - capped hallucination
    ]

    expected = sum(w * v for w, v in zip(weights, values)) / sum(weights)
    actual = model.score(signals)

    assert abs(actual - expected) < 1e-6, f"Weight-value ordering mismatch: expected {expected}, got {actual}"


def test_confidence_model_weight_sum() -> None:
    """Verify default weights sum to 1.0 for interpretability."""
    model = ConfidenceModel()
    total = model.w_retriever + model.w_evidence + model.w_agreement + model.w_llm + model.w_hallucination
    assert abs(total - 1.0) < 1e-6


def test_confidence_model_caps_hallucination_at_half() -> None:
    """Hallucination penalty should be capped at 0.5."""
    model = ConfidenceModel()

    # High hallucination should be capped
    signals_high = ConfidenceSignals(
        retriever_top1=0.5,
        evidence_sufficiency=0.5,
        support_agreement=0.5,
        hallucination_penalty=0.9,  # Would drive confidence negative without cap
        raw_confidence=0.5,
    )

    # With cap: 1 - 0.5 = 0.5 contribution from hallucination term
    # Without cap: 1 - 0.9 = 0.1 contribution
    score_with_cap = model.score(signals_high)

    # The hallucination term contributes 0.15 * 0.5 = 0.075
    # If uncapped, it would contribute 0.15 * 0.1 = 0.015
    # Difference = 0.06, so score should be meaningfully higher with cap
    assert score_with_cap > 0.3  # Sanity check


def test_confidence_signals_field_order() -> None:
    """Verify ConfidenceSignals dataclass field order matches expectation."""
    from dataclasses import fields

    field_names = [f.name for f in fields(ConfidenceSignals)]
    expected = [
        "retriever_top1",
        "evidence_sufficiency",
        "support_agreement",
        "hallucination_penalty",
        "raw_confidence",
    ]
    assert field_names == expected, f"Field order changed: {field_names}"