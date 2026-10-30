"""Comprehensive unit and integration tests for the Risk-Aware Triage System."""

from __future__ import annotations

import pytest

from src.confidence.pipeline import Pipeline
from src.confidence.triage import _check_emergency, assign_risk
from src.models import (
    ClinicalResponse,
    PatientProfile,
    RiskLevel,
    TriageDecision,
    Visit,
    Vitals,
)


# =====================================================================
# 1. Unit Tests: Vital Signs Emergency Triggers
# =====================================================================


def test_emergency_vitals_hypotension() -> None:
    vitals = Vitals(systolic_bp=85)
    reason = _check_emergency(vitals, None, "Headache")
    assert reason is not None
    assert "Systolic blood pressure < 90" in reason


def test_normal_vitals_systolic_bp_no_emergency() -> None:
    vitals = Vitals(systolic_bp=120)
    reason = _check_emergency(vitals, None, "Mild headache")
    assert reason is None


def test_emergency_vitals_hypoxemia() -> None:
    vitals = Vitals(oxygen_saturation=88)
    reason = _check_emergency(vitals, None, "Cough")
    assert reason is not None
    assert "Oxygen saturation < 90%" in reason


def test_emergency_vitals_tachycardia() -> None:
    vitals = Vitals(heart_rate=162)
    reason = _check_emergency(vitals, None, "Palpitations")
    assert reason is not None
    assert "Heart rate > 150 bpm" in reason


def test_emergency_vitals_tachypnea() -> None:
    vitals = Vitals(respiratory_rate=34)
    reason = _check_emergency(vitals, None, "Cough and cold")
    assert reason is not None
    assert "Respiratory rate > 30" in reason


# =====================================================================
# 2. Unit Tests: Clinical Presentation Triggers
# =====================================================================


def test_emergency_chest_pain_with_sweating_geriatric() -> None:
    profile = PatientProfile(age=58)
    query = "58-year-old male with central crushing chest pain and sweating"
    reason = _check_emergency(None, profile, query)
    assert reason is not None
    assert "ACS concern" in reason


def test_emergency_chest_pain_with_sob_unknown_age() -> None:
    query = "Sudden onset severe chest pain and shortness of breath"
    reason = _check_emergency(None, None, query)
    assert reason is not None
    assert "ACS concern" in reason


def test_non_emergency_chest_pain_young_without_red_flags() -> None:
    profile = PatientProfile(age=21)
    query = "21-year-old with sharp chest pain after lifting weights, no shortness of breath, no diaphoresis"
    reason = _check_emergency(None, profile, query)
    assert reason is None


def test_emergency_anaphylaxis_with_hypotension() -> None:
    query = "Patient has lip swelling, urticaria, and severe hypotension after taking amoxicillin"
    reason = _check_emergency(None, None, query)
    assert reason is not None
    assert "anaphylaxis" in reason.lower()


def test_emergency_acute_stroke_syndrome() -> None:
    query = "65-year-old woman with sudden weakness and slurred speech"
    reason = _check_emergency(None, None, query)
    assert reason is not None
    assert "stroke" in reason.lower()


def test_emergency_facial_droop() -> None:
    query = "Patient arrived with acute facial droop on the left side"
    reason = _check_emergency(None, None, query)
    assert reason is not None
    assert "stroke" in reason.lower()


def test_emergency_suspected_sepsis() -> None:
    query = "Elderly bedridden patient with high fever, confusion, suspected sepsis"
    reason = _check_emergency(None, None, query)
    assert reason is not None
    assert "sepsis" in reason.lower()


# =====================================================================
# 3. Unit Tests: Confidence-Based Tiering & Decisions
# =====================================================================


def test_assign_risk_emergency_overrides_confidence() -> None:
    resp = ClinicalResponse(primary_diagnosis="acute_coronary_syndrome", confidence=0.95)
    query = "60-year-old with chest pain, diaphoresis, and shortness of breath"
    assigned = assign_risk(resp, query=query)
    assert assigned.risk_level == RiskLevel.EMERGENCY
    assert assigned.decision == TriageDecision.ESCALATE
    assert "ACS concern" in assigned.escalated_reason


def test_assign_risk_low_confidence_escalates() -> None:
    resp = ClinicalResponse(primary_diagnosis="unknown_syndrome", confidence=0.55)
    assigned = assign_risk(resp, query="Vague recurring dizziness")
    assert assigned.risk_level == RiskLevel.URGENT
    assert assigned.decision == TriageDecision.ESCALATE
    assert "Low confidence" in assigned.escalated_reason


def test_assign_risk_moderate_confidence_asks_followup() -> None:
    resp = ClinicalResponse(primary_diagnosis="tension_headache", confidence=0.76)
    assigned = assign_risk(resp, query="Mild headache for 2 days")
    assert assigned.risk_level == RiskLevel.ROUTINE
    assert assigned.decision == TriageDecision.ASK_FOLLOWUP


def test_assign_risk_high_confidence_answers() -> None:
    resp = ClinicalResponse(primary_diagnosis="uncomplicated_uti", confidence=0.91)
    assigned = assign_risk(resp, query="45-year-old woman with dysuria and frequency, no fever")
    assert assigned.risk_level == RiskLevel.HOME_CARE
    assert assigned.decision == TriageDecision.ANSWER


# =====================================================================
# 4. End-to-End Pipeline Triage Integration Tests
# =====================================================================


@pytest.fixture(scope="module")
def pipeline() -> Pipeline:
    return Pipeline.from_seed()


def test_pipeline_triage_emergency_acs(pipeline: Pipeline) -> None:
    query = "55-year-old diabetic male presents with central chest pain, sweating, and shortness of breath."
    _, resp = pipeline.analyze(query)
    assert resp.risk_level == RiskLevel.EMERGENCY
    assert resp.decision == TriageDecision.ESCALATE
    assert resp.escalated_reason is not None


def test_pipeline_triage_emergency_stroke(pipeline: Pipeline) -> None:
    query = "65-year-old man with sudden right-sided weakness and slurred speech."
    _, resp = pipeline.analyze(query)
    assert resp.risk_level == RiskLevel.EMERGENCY
    assert resp.decision == TriageDecision.ESCALATE


def test_pipeline_triage_emergency_anaphylaxis(pipeline: Pipeline) -> None:
    query = "30-year-old woman with urticaria, lip swelling, and hypotension after receiving penicillin."
    _, resp = pipeline.analyze(query)
    assert resp.risk_level == RiskLevel.EMERGENCY
    assert resp.decision == TriageDecision.ESCALATE


def test_pipeline_triage_routine_presentation(pipeline: Pipeline) -> None:
    query = "45-year-old woman with dysuria and urinary frequency without flank pain."
    _, resp = pipeline.analyze(query)
    # Clinical guideline matches uncomplicated UTI with high confidence
    assert resp.risk_level in (RiskLevel.ROUTINE, RiskLevel.HOME_CARE)
    assert resp.decision in (TriageDecision.ANSWER, TriageDecision.ASK_FOLLOWUP)


def test_pipeline_triage_vitals_overrides_mild_symptom(pipeline: Pipeline) -> None:
    # A patient presenting with a simple sore throat, but SpO2 is critically low (86%)
    visit = Visit(vitals=Vitals(oxygen_saturation=86))
    _, resp = pipeline.analyze("25-year-old with mild sore throat", visit=visit)
    assert resp.risk_level == RiskLevel.EMERGENCY
    assert resp.decision == TriageDecision.ESCALATE
    assert "Oxygen saturation < 90%" in (resp.escalated_reason or "")
