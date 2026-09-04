"""Risk-aware triage and escalation."""

from __future__ import annotations

from ..models import ClinicalResponse, PatientProfile, RiskLevel, TriageDecision, Vitals


def _check_emergency(vitals: Vitals | None, profile: PatientProfile | None, query: str = "") -> str | None:
    """Rule-based emergency triggers. Returns reason string if emergency."""
    q = query.lower()

    if vitals is not None:
        if vitals.systolic_bp is not None and vitals.systolic_bp < 90:
            return "Systolic blood pressure < 90 mmHg"
        if vitals.oxygen_saturation is not None and vitals.oxygen_saturation < 90:
            return "Oxygen saturation < 90%"
        if vitals.heart_rate is not None and vitals.heart_rate > 150:
            return "Heart rate > 150 bpm"
        if vitals.respiratory_rate is not None and vitals.respiratory_rate > 30:
            return "Respiratory rate > 30"

    age = profile.age if profile else None
    chest_pain = "chest pain" in q
    sweating = any(s in q for s in ("sweat", "diaphores"))
    sob = any(s in q for s in ("shortness of breath", "dyspnea", "breathless"))
    if chest_pain and (sweating or sob) and (age is None or age > 50):
        return "High-risk chest pain presentation (ACS concern)"

    if any(s in q for s in ("anaphyla", "lip swelling", "angioedema", "urticaria")) and "hypotension" in q:
        return "Possible anaphylaxis with hypotension"

    if any(s in q for s in ("stroke", "facial droop", "slurred speech", "sudden weakness")):
        return "Acute stroke syndrome"

    if "sepsis" in q or "septic" in q:
        return "Suspected sepsis"

    return None


def assign_risk(
    response: ClinicalResponse,
    profile: PatientProfile | None = None,
    vitals: Vitals | None = None,
    query: str = "",
) -> ClinicalResponse:
    """Overlay risk level and decision on a response based on safety rules."""
    emergency_reason = _check_emergency(vitals, profile, query)
    if emergency_reason:
        response.risk_level = RiskLevel.EMERGENCY
        response.decision = TriageDecision.ESCALATE
        response.escalated_reason = emergency_reason + " — immediate clinician review required."
        return response

    if response.confidence < 0.7:
        response.risk_level = RiskLevel.URGENT
        response.decision = TriageDecision.ESCALATE
        if not response.escalated_reason:
            response.escalated_reason = "Low confidence requires clinician review."
        return response

    if response.confidence < 0.85:
        response.risk_level = RiskLevel.ROUTINE
        response.decision = TriageDecision.ASK_FOLLOWUP
        return response

    response.risk_level = RiskLevel.HOME_CARE
    response.decision = TriageDecision.ANSWER
    return response
