"""FastAPI application for CAAR-CDSS.

Endpoints:
  GET  /health                    - liveness probe
  POST /login                     - issue a (mock) JWT
  POST /analyze                   - run the full AEB pipeline
  POST /retrieve                  - return raw retrieved evidence
  GET  /metrics                   - aggregate evaluation metrics
  POST /feedback                  - record user feedback
  POST /guidelines/upload         - ingest PDF/text document into corpus
  GET  /query/{id}                - retrieve past query's full response
  POST /evaluation/run            - trigger benchmark evaluation run
  GET  /evaluation/{id}/results   - fetch evaluation results
  POST /review/{response_id}      - human annotation endpoint
  GET  /corpus/stats              - corpus size, chunk count, specialty coverage

Runs entirely in-memory (seed corpus + mock LLM) so it works with zero setup.
Swap in the real embedder/LLM via --real mode / env flags.
"""

from __future__ import annotations

import hashlib
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from jose import jwt
from datetime import datetime

from .models import (
    AnalyzeRequest,
    CorpusStatsResponse,
    EvaluationRunRequest,
    EvaluationRunResponse,
    EvaluationResultsResponse,
    FeedbackRequest,
    GuidelineUploadRequest,
    GuidelineUploadResponse,
    LoginRequest,
    QueryHistoryResponse,
    ReviewRequest,
    ReviewResponse,
)

from ..confidence.pipeline import Pipeline
from ..data.seed_corpus import get_seed_queries
from ..experiments.runner import evaluate
from ..models import PatientProfile, Visit

# Database setup
_database_engine = None
_session_maker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _database_engine, _session_maker
    _database_engine, _session_maker = await create_engine_and_session()
    await init_db(_database_engine)
    yield
    await _database_engine.dispose()


app = FastAPI(title="CAAR-CDSS API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: Pipeline | None = None
_feedback: list[dict[str, Any]] = []

JWT_SECRET = "dev-secret-change-me-to-32-random-bytes"  # >= 32 bytes
JWT_ALG = "HS256"


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline.from_seed()
    return _pipeline


async def get_db_session():
    """FastAPI dependency for database session."""
    if _session_maker is None:
        # Database not initialized (e.g., in tests)
        yield None
        return
    async with _session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


def require_auth(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization[len("Bearer "):]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload.get("sub", "anonymous")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


def _hash_query(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()[:64]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "caar-cdss"}


@app.post("/login")
async def login(req: LoginRequest) -> dict[str, Any]:
    # Dev-only: accept any non-empty credentials.
    token = jwt.encode({"sub": req.username, "exp": int(time.time()) + 3600}, JWT_SECRET, algorithm=JWT_ALG)
    return {"access_token": token, "token_type": "bearer", "expires_in": 3600}


@app.post("/analyze")
async def analyze(
    req: AnalyzeRequest,
    auth: str = Depends(require_auth),
    db=Depends(get_db_session),
) -> dict[str, Any]:
    pipeline = get_pipeline()
    patient = PatientProfile(
        age=req.age,
        gender=req.gender,
        comorbidities=req.comorbidities,
        medications=req.medications,
        allergies=req.allergies,
    )
    visit = Visit(vitals=req.vitals) if req.vitals else None

    aeb, resp = pipeline.analyze(req.query, visit=visit, patient=patient)

    # Prepare response dict
    response = {
        "primary_diagnosis": resp.primary_diagnosis,
        "confidence": round(resp.confidence, 4),
        "uncertainty": round(resp.uncertainty, 4),
        "hallucination_score": round(resp.hallucination_score, 4),
        "risk_level": resp.risk_level.value,
        "decision": resp.decision.value,
        "escalated_reason": resp.escalated_reason,
        "retrieval_k_used": resp.retrieval_k_used,
        "retrieval_steps": len(aeb.steps),
        "confidence_curve": [round(c, 4) for c in aeb.confidence_curve],
        "differential": [
            {
                "diagnosis": d.diagnosis,
                "probability": round(d.probability, 4),
                "sources": d.sources,
            }
            for d in resp.differential
        ],
        "evidence": [
            {"source": c.source, "title": c.title, "rank": c.rank, "text": c.text[:400]}
            for c in resp.evidence
        ],
        "reasoning": resp.reasoning,
        "disclaimer": resp.safety_disclaimer,
    }

    # Log query to database (optional - skip if db not available)
    if db is not None:
        query_hash = _hash_query(req.query)
        patient_dict = {
            "age": req.age,
            "gender": req.gender,
            "comorbidities": req.comorbidities,
            "medications": req.medications,
            "allergies": req.allergies,
            "vitals": req.vitals.model_dump() if req.vitals else None,
        }
        await log_query(db, req.query, _hash_query(req.query), response, patient_dict)

    return response


@app.post("/retrieve")
async def retrieve(
    req: AnalyzeRequest,
    auth: str = Depends(require_auth),
    db=Depends(get_db_session),
) -> dict[str, Any]:
    pipeline = get_pipeline()
    chunks = pipeline.retriever.retrieve(req.query, top_k=10)
    return {
        "query": req.query,
        "results": [
            {
                "source": c.source,
                "title": c.title,
                "rank": c.rank,
                "similarity": round(c.similarity_score, 4),
                "text": c.text[:400],
            }
            for c in chunks
        ],
    }


@app.get("/metrics")
async def metrics(auth: str = Depends(require_auth)) -> dict[str, Any]:
    pipeline = get_pipeline()
    report = evaluate(pipeline, get_seed_queries(), method="aeb")
    return {
        "method": report.method,
        "n": report.n,
        "accuracy": round(report.accuracy, 4),
        "avg_confidence": round(report.avg_confidence, 4),
        "avg_hallucination": round(report.avg_hallucination, 4),
        "avg_retrieval_k": round(report.avg_retrieval_k, 4),
        "budget_exhausted_rate": round(report.budget_exhausted_rate, 4),
        "avg_latency_ms": round(report.avg_latency_ms, 2),
    }


@app.post("/feedback")
async def feedback(
    req: FeedbackRequest,
    auth: str = Depends(require_auth),
) -> dict[str, str]:
    _feedback.append({"query": req.query, "helpful": req.helpful, "comment": req.comment, "ts": time.time()})
    return {"status": "recorded"}


# === New Sprint 2 Endpoints ===


@app.post("/guidelines/upload", response_model=GuidelineUploadResponse)
async def upload_guideline(
    request: GuidelineUploadRequest,
    file: UploadFile = File(...),
    auth: str = Depends(require_auth),
    db=Depends(get_db_session),
) -> GuidelineUploadResponse:
    """Ingest a PDF or text document into the corpus at runtime."""
    pipeline = get_pipeline()
    kb = pipeline.kb

    content = await file.read()
    if file.content_type == "application/pdf":
        # For PDF, we'd need pdfplumber - for now treat as text
        text = content.decode("utf-8", errors="ignore")
    else:
        text = content.decode("utf-8", errors="ignore")

    chunks = kb.add_documents(
        [
            {
                "text": text,
                "source": request.source,
                "title": request.title or file.filename,
                "year": request.year,
                "trust_score": request.trust_score,
            }
        ]
    )

    return GuidelineUploadResponse(
        status="success",
        chunks_added=chunks,
        document_id=request.title or file.filename,
    )


@app.get("/query/{query_id}", response_model=QueryHistoryResponse)
async def get_query_history(
    query_id: UUID,
    auth: str = Depends(require_auth),
    db=Depends(get_db_session),
) -> QueryHistoryResponse:
    """Retrieve a past query's full response and retrieved evidence."""
    query_log = await get_query_log(db, query_id)
    if not query_log:
        raise HTTPException(status_code=404, detail="Query not found")

    return QueryHistoryResponse(
        id=query_log.id,
        query_text=query_log.query_text,
        query_hash=query_log.query_hash,
        primary_diagnosis=query_log.primary_diagnosis,
        confidence=query_log.confidence,
        uncertainty=query_log.uncertainty,
        hallucination_score=query_log.hallucination_score,
        risk_level=query_log.risk_level,
        decision=query_log.decision,
        escalated_reason=query_log.escalated_reason,
        retrieval_k_used=query_log.retrieval_k_used,
        retrieval_steps=query_log.retrieval_steps,
        confidence_curve=query_log.confidence_curve,
        differential=query_log.differential,
        evidence=query_log.evidence,
        reasoning=query_log.reasoning,
        patient_age=query_log.patient_age,
        patient_gender=query_log.patient_gender,
        patient_comorbidities=query_log.patient_comorbidities,
        patient_medications=query_log.patient_medications,
        patient_allergies=query_log.patient_allergies,
        vitals=query_log.vitals,
        created_at=query_log.created_at,
    )


async def _run_evaluation_background(
    eval_id: UUID,
    method: str,
    benchmark: str,
    n_samples: int,
):
    """Background task to run evaluation."""

    async with _session_maker() as session:
        try:
            pipeline = get_pipeline()
            queries = get_seed_queries()[:n_samples]

            report = evaluate(pipeline, queries, method=method)

            await update_evaluation_run(
                session,
                eval_id,
                "completed",
                results={
                    "accuracy": report.accuracy,
                    "avg_confidence": report.avg_confidence,
                    "avg_hallucination": report.avg_hallucination,
                    "avg_latency_ms": report.avg_latency_ms,
                    "avg_retrieval_k": report.avg_retrieval_k,
                    "avg_steps": report.avg_steps,
                    "budget_exhausted_rate": report.budget_exhausted_rate,
                },
            )
        except Exception as e:
            await update_evaluation_run(session, eval_id, "failed", error=str(e))


@app.post("/evaluation/run", response_model=EvaluationRunResponse)
async def run_evaluation(
    request: EvaluationRunRequest,
    background_tasks: BackgroundTasks,
    auth: str = Depends(require_auth),
    db=Depends(get_db_session),
) -> EvaluationRunResponse:
    """Trigger a benchmark evaluation run in the background."""
    eval_id = await create_evaluation_run(
        db, request.method, request.benchmark, request.n_samples, request.config
    )

    # Schedule background evaluation
    background_tasks.add_task(
        _run_evaluation_background,
        eval_id,
        request.method,
        request.benchmark,
        request.n_samples,
    )

    return EvaluationRunResponse(
        id=str(eval_id),
        method=request.method,
        benchmark=request.benchmark,
        status="pending",
        n_samples=request.n_samples,
        config=request.config,
        started_at=datetime.utcnow(),
        completed_at=None,
    )


@app.get("/evaluation/{eval_id}/results", response_model=EvaluationResultsResponse)
async def get_evaluation_results(
    eval_id: UUID,
    auth: str = Depends(require_auth),
    db=Depends(get_db_session),
) -> EvaluationResultsResponse:
    """Fetch evaluation run results."""
    from sqlalchemy import select

    result = await db.execute(select(EvaluationRun).where(EvaluationRun.id == eval_id))
    eval_run = result.scalar_one_or_none()
    if not eval_run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")

    results = eval_run.results
    return EvaluationResultsResponse(
        id=str(eval_run.id),
        method=eval_run.method,
        benchmark=eval_run.benchmark,
        status=eval_run.status,
        n_samples=eval_run.n_samples,
        accuracy=results.get("accuracy") if results else None,
        avg_confidence=results.get("avg_confidence") if results else None,
        avg_hallucination=results.get("avg_hallucination") if results else None,
        avg_latency_ms=results.get("avg_latency_ms") if results else None,
        avg_retrieval_k=results.get("avg_retrieval_k") if results else None,
        avg_steps=results.get("avg_steps") if results else None,
        budget_exhausted_rate=results.get("budget_exhausted_rate") if results else None,
        config=eval_run.config,
        results=eval_run.results,
        error=eval_run.error,
        started_at=eval_run.started_at,
        completed_at=eval_run.completed_at,
    )


@app.post("/review/{response_id}", response_model=ReviewResponse)
async def submit_review(
    response_id: UUID,
    request: ReviewRequest,
    auth: str = Depends(require_auth),
    db=Depends(get_db_session),
) -> ReviewResponse:
    """Submit human annotation for a query response or evaluation."""
    from sqlalchemy import select

    # Verify the query_log or evaluation exists
    if request.hallucination_flag is not None:
        # Check if it's a query log or evaluation
        query_exists = False
        eval_exists = False

        if response_id:
            query_result = await db.execute(select(QueryLog).where(QueryLog.id == response_id))
            query_exists = query_result.scalar_one_or_none() is not None

            eval_result = await db.execute(select(EvaluationRun).where(EvaluationRun.id == response_id))
            eval_exists = eval_result.scalar_one_or_none() is not None

        if not query_exists and not eval_exists:
            raise HTTPException(status_code=404, detail="Response not found")

    review_id = await add_review(
        db,
        query_log_id=response_id if query_exists else None,
        evaluation_id=response_id if eval_exists else None,
        hallucination_flag=request.hallucination_flag,
        notes=request.notes,
        reviewer=request.reviewer,
    )

    return ReviewResponse(
        id=str(review_id),
        query_log_id=str(response_id) if query_exists else None,
        evaluation_id=str(response_id) if eval_exists else None,
        hallucination_flag=request.hallucination_flag,
        notes=request.notes,
        reviewer=request.reviewer,
        created_at=datetime.utcnow(),
    )


@app.get("/corpus/stats", response_model=CorpusStatsResponse)
async def get_corpus_stats(
    auth: str = Depends(require_auth),
    db=Depends(get_db_session),
) -> CorpusStatsResponse:
    """Get corpus statistics: size, chunk count, specialty coverage."""
    from sqlalchemy import select

    # Try to get from database cache first
    from .database import CorpusStats
    result = await db.execute(select(CorpusStats).limit(1))
    stats = result.scalar_one_or_none()

    if stats:
        return CorpusStatsResponse(
            total_chunks=stats.total_chunks,
            total_documents=stats.total_documents,
            specialty_distribution=stats.specialty_distribution,
            source_distribution=stats.source_distribution,
            last_updated=stats.last_updated,
        )

    # Fallback: compute from ChromaDB
    pipeline = get_pipeline()
    store = pipeline.kb.store

    if hasattr(store, "_collection"):
        total_chunks = store._collection.count()
        # Get specialty distribution
        results = store._collection.get(include=["metadatas"])
        specialty_dist = {}
        source_dist = {}
        for meta in results.get("metadatas", []):
            spec = meta.get("specialty", "unknown")
            src = meta.get("source", "unknown")
            specialty_dist[spec] = specialty_dist.get(spec, 0) + 1
            source_dist[src] = source_dist.get(src, 0) + 1

        # Update cache
        await update_corpus_stats(db, total_chunks, len(set(r.get("source") for r in results.get("metadatas", []))), specialty_dist, source_dist)

        return CorpusStatsResponse(
            total_chunks=total_chunks,
            total_documents=len(set(r.get("source") for r in results.get("metadatas", []))),
            specialty_distribution=specialty_dist,
            source_distribution=source_dist,
            last_updated=datetime.utcnow(),
        )

    return CorpusStatsResponse(
        total_chunks=0,
        total_documents=0,
        specialty_distribution={},
        source_distribution={},
        last_updated=None,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
