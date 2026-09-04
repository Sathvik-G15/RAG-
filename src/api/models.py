"""Pydantic models for CAAR-CDSS API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str
    password: str


class FeedbackRequest(BaseModel):
    query: str
    helpful: bool
    comment: str = ""


class AnalyzeRequest(BaseModel):
    query: str
    age: int | None = None
    gender: str | None = None
    comorbidities: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    vitals: Any | None = None


class GuidelineUploadRequest(BaseModel):
    source: str = Field(..., description="Source identifier (e.g., WHO, NICE)")
    title: str | None = None
    year: int | None = None
    trust_score: float = Field(default=0.8, ge=0.0, le=1.0)


class GuidelineUploadResponse(BaseModel):
    status: str
    chunks_added: int
    document_id: str | None = None


class QueryHistoryResponse(BaseModel):
    id: UUID
    query_text: str
    query_hash: str
    primary_diagnosis: str | None
    confidence: float | None
    uncertainty: float | None
    hallucination_score: float | None
    risk_level: str | None
    decision: str | None
    escalated_reason: str | None
    retrieval_k_used: int | None
    retrieval_steps: int | None
    confidence_curve: list[float] | None
    differential: list[dict] | None
    evidence: list[dict] | None
    reasoning: str | None
    patient_age: int | None
    patient_gender: str | None
    patient_comorbidities: list[str] | None
    patient_medications: list[str] | None
    patient_allergies: list[str] | None
    vitals: dict | None
    created_at: datetime


class EvaluationRunRequest(BaseModel):
    method: str = Field(..., pattern="^(vanilla|hybrid|aeb)$")
    benchmark: str = Field(..., pattern="^(medqa|pubmedqa|seeds)$")
    n_samples: int = Field(default=10, ge=1, le=500)
    config: dict | None = None


class EvaluationRunResponse(BaseModel):
    id: str
    method: str
    benchmark: str
    status: str
    n_samples: int
    config: dict | None
    started_at: datetime
    completed_at: datetime | None


class EvaluationResultsResponse(BaseModel):
    id: str
    method: str
    benchmark: str
    status: str
    n_samples: int
    accuracy: float | None = None
    avg_confidence: float | None = None
    avg_hallucination: float | None = None
    avg_latency_ms: float | None = None
    avg_retrieval_k: float | None = None
    avg_steps: float | None = None
    budget_exhausted_rate: float | None = None
    config: dict | None
    results: dict | None = None
    error: str | None = None
    started_at: datetime
    completed_at: datetime | None


class ReviewRequest(BaseModel):
    hallucination_flag: bool | None = None
    notes: str | None = None
    reviewer: str | None = None


class ReviewResponse(BaseModel):
    id: str
    query_log_id: str | None = None
    evaluation_id: str | None = None
    hallucination_flag: bool | None
    notes: str | None
    reviewer: str | None
    created_at: datetime


class CorpusStatsResponse(BaseModel):
    total_chunks: int
    total_documents: int
    specialty_distribution: dict[str, int]
    source_distribution: dict[str, int]
    last_updated: datetime | None = None


class GuidelineUploadRequest(BaseModel):
    source: str = Field(..., description="Source identifier (e.g., WHO, NICE)")
    title: str | None = None
    year: int | None = None
    trust_score: float = Field(default=0.8, ge=0.0, le=1.0)


class GuidelineUploadResponse(BaseModel):
    status: str
    chunks_added: int
    document_id: str | None = None


class ReviewRequest(BaseModel):
    hallucination_flag: bool | None = None
    notes: str | None = None
    reviewer: str | None = None


class ReviewResponse(BaseModel):
    id: str
    query_log_id: str | None = None
    evaluation_id: str | None = None
    hallucination_flag: bool | None
    notes: str | None
    reviewer: str | None
    created_at: datetime


class CorpusStatsResponse(BaseModel):
    total_chunks: int
    total_documents: int
    specialty_distribution: dict[str, int]
    source_distribution: dict[str, int]
    last_updated: datetime | None = None


class EvaluationRunCreate(BaseModel):
    method: str = Field(..., pattern="^(vanilla|hybrid|aeb)$")
    benchmark: str = Field(..., pattern="^(medqa|pubmedqa|seeds)$")
    n_samples: int = Field(default=10, ge=1, le=500)
    config: dict | None = None