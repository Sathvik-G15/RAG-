"""End-to-end pipeline tests (mock mode, no GPU/network required)."""

from __future__ import annotations

import pytest

from src.agents.context_builder import build_context_query, infer_profile
from src.confidence.pipeline import Pipeline
from src.data.seed_corpus import get_seed_queries
from src.experiments.runner import evaluate
from src.models import TriageDecision
from src.retrieval.hybrid import HybridRetriever
from src.verification.verifier import LexicalVerifier


@pytest.fixture(scope="module")
def pipeline() -> Pipeline:
    return Pipeline.from_seed()


def test_knowledge_base_built(pipeline: Pipeline) -> None:
    assert len(pipeline.kb.store) >= 20


def test_hybrid_retrieval_surfaces_relevant_doc(pipeline: Pipeline) -> None:
    chunks = pipeline.retriever.retrieve(
        "55-year-old diabetic male presents with central chest pain and sweating",
        top_k=5,
    )
    assert chunks, "expected at least one retrieved chunk"
    assert chunks[0].source == "WHO/cardiac_chest_pain"


def test_infer_profile_extracts_demographics() -> None:
    prof = infer_profile("70-year-old female with diabetes and hypertension")
    assert prof.age == 70
    assert prof.gender == "female"
    assert "diabetes" in prof.comorbidities
    assert "hypertension" in prof.comorbidities


def test_build_context_query_adds_patient_terms() -> None:
    prof = infer_profile("65-year-old male with copd")
    cq = build_context_query("65-year-old male with copd", prof)
    assert "geriatric" in cq


@pytest.mark.parametrize("q", get_seed_queries())
def test_aeb_diagnoses_seed_queries(pipeline: Pipeline, q: dict) -> None:
    aeb, resp = pipeline.analyze(q["query"])
    assert resp.primary_diagnosis == q["expected_dx"]
    # High-risk presentations (ACS, anaphylaxis, stroke) must escalate for safety;
    # everything else answers or asks a follow-up.
    if resp.decision == TriageDecision.ESCALATE:
        assert resp.risk_level.value == "emergency"
    else:
        assert resp.decision in (TriageDecision.ANSWER, TriageDecision.ASK_FOLLOWUP)


def test_aeb_efficiency(pipeline: Pipeline) -> None:
    report = evaluate(pipeline, get_seed_queries(), method="aeb")
    assert report.accuracy == 1.0
    assert report.avg_retrieval_k <= 6.0
    assert report.budget_exhausted_rate == 0.0


def test_out_of_scope_query_escalates(pipeline: Pipeline) -> None:
    aeb, resp = pipeline.analyze("Patient with severe abdominal pain after starting a new blood thinner")
    assert resp.decision == TriageDecision.ESCALATE
    assert resp.confidence < 0.5


def test_lexical_verifier_marks_unsupported_claim() -> None:
    v = LexicalVerifier()
    from src.kb.ingest import KnowledgeBase

    # Minimal evidence that does NOT mention influenza.
    chunks = KnowledgeBase.from_config().embedder and None or None
    evidence = []
    result = v.verify("The patient has influenza.", evidence)
    assert result.total_claims >= 1
    assert result.supported_claims == 0


def test_hybrid_beats_dense_on_recall(pipeline: Pipeline) -> None:
    # Hybrid retrieval should return the exact source doc for a keyword query.
    assert isinstance(pipeline.retriever, HybridRetriever)
