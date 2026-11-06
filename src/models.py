"""Domain models for CAAR-CDSS."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    EMERGENCY = "emergency"
    URGENT = "urgent"
    ROUTINE = "routine"
    HOME_CARE = "home_care"


class TriageDecision(str, Enum):
    ANSWER = "answer"
    ASK_FOLLOWUP = "ask_followup"
    ESCALATE = "escalate"


class PatientProfile(BaseModel):
    age: int | None = None
    gender: str | None = None
    weight: float | None = None
    height: float | None = None
    pregnancy_status: bool | None = None
    comorbidities: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)


class Vitals(BaseModel):
    systolic_bp: float | None = None
    diastolic_bp: float | None = None
    heart_rate: float | None = None
    respiratory_rate: float | None = None
    temperature: float | None = None
    oxygen_saturation: float | None = None
    blood_glucose: float | None = None


class Visit(BaseModel):
    visit_id: UUID = Field(default_factory=uuid4)
    patient_id: UUID | None = None
    symptoms: list[str] = Field(default_factory=list)
    duration: dict[str, str] = Field(default_factory=dict)
    vitals: Vitals = Field(default_factory=Vitals)
    labs: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EvidenceChunk(BaseModel):
    chunk_id: UUID = Field(default_factory=uuid4)
    text: str
    source: str
    title: str | None = None
    year: int | None = None
    specialty: str | None = None
    trust_score: float = 0.8
    similarity_score: float = 0.0
    boosted_score: float = 0.0
    rank: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class DifferentialDiagnosis(BaseModel):
    diagnosis: str
    probability: float
    supporting_evidence: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    query_signal: bool = False

    @field_validator("supporting_evidence", "sources", mode="before")
    @classmethod
    def _coerce_str_list(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            v_str = v.strip()
            return [v_str] if v_str else []
        if isinstance(v, (list, tuple, set)):
            return [str(item).strip() for item in v if item is not None and str(item).strip()]
        return [str(v)]

    @field_validator("probability", mode="before")
    @classmethod
    def _coerce_probability(cls, v: Any) -> float:
        if v is None:
            return 0.5
        try:
            val = float(v)
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.5


class ClinicalResponse(BaseModel):
    visit_id: UUID | None = None
    primary_diagnosis: str | None = None
    differential: list[DifferentialDiagnosis] = Field(default_factory=list)
    confidence: float = 0.0
    uncertainty: float = 0.0
    hallucination_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.ROUTINE
    decision: TriageDecision = TriageDecision.ANSWER
    followup_questions: list[str] = Field(default_factory=list)
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    reasoning: str = ""
    retrieval_k_used: int = 0
    escalated_reason: str | None = None
    triage_reasons: list[str] = Field(default_factory=list)
    safety_disclaimer: str = (
        "This is an AI-assisted analysis for research purposes only. "
        "It does not replace professional medical advice. Always consult a qualified clinician."
    )

    @field_validator("confidence", "uncertainty", "hallucination_score", mode="before")
    @classmethod
    def _coerce_floats(cls, v: Any) -> float:
        if v is None:
            return 0.0
        try:
            val = float(v)
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.0

