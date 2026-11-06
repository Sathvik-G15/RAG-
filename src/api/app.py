"""FastAPI application for CAAR-CDSS.

Endpoints:
  GET  /health                    - liveness probe
  POST /login                     - issue a JWT
  POST /analyze                   - run the full AEB pipeline
  POST /retrieve                  - return raw retrieved evidence
  GET  /metrics                   - aggregate evaluation metrics
  POST /feedback                  - record user feedback
  POST /guidelines/upload         - ingest PDF/text document into corpus (admin only)
  GET  /query/{id}                - retrieve past query's full response
  POST /evaluation/run            - trigger benchmark evaluation run (admin only)
  GET  /evaluation/{id}/results   - fetch evaluation results
  POST /review/{response_id}      - human annotation endpoint
  GET  /corpus/stats              - corpus size, chunk count, specialty coverage

Security:
  - Rate limiting on /analyze and /evaluation/run
  - Admin-only upload and evaluation endpoints
  - PHI redaction in logs
  - Prompt-injection fencing for retrieved evidence
  - External inference blocked by default (requires security.external_inference_allowed=true)
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from jose import jwt

from ..confidence.pipeline import Pipeline
from ..data.seed_corpus import get_seed_queries
from ..experiments.runner import evaluate
from ..models import PatientProfile, Visit, Vitals
from .database import (
    EvaluationRun,
    QueryLog,
    add_review,
    create_engine_and_session,
    create_evaluation_run,
    get_query_log,
    init_db,
    log_query,
    update_corpus_stats,
    update_evaluation_run,
)
from .models import (
    AnalyzeRequest,
    CorpusStatsResponse,
    EvaluationResultsResponse,
    EvaluationRunRequest,
    EvaluationRunResponse,
    FeedbackRequest,
    GuidelineUploadResponse,
    LoginRequest,
    QueryHistoryResponse,
    ReviewRequest,
    ReviewResponse,
)
from .security import (
    RateLimiter,
    MAX_UPLOAD_BYTES,
    UNTRUSTED_TRUST,
    redact_phi,
    require_role,
    trust_for_source,
    validate_upload,
    wrap_evidence,
)

# Database setup
_database_engine = None
_session_maker = None

# Rate limiters
analyze_limiter = RateLimiter(30, 60)  # 30 calls per 60 seconds
eval_limiter = RateLimiter(2, 3600)    # 2 calls per hour

# JWT config (read from env in production)
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

JWT_SECRET = os.environ.get("JWT_SECRET", "dev-secret-change-me-to-32-random-bytes")
JWT_ALG = "HS256"

# Admin credentials from environment (for production)
ADMIN_USERNAME = os.environ.get("CAAR_ADMIN_USER")
ADMIN_PASSWORD_HASH = os.environ.get("CAAR_ADMIN_PASS_HASH")  # bcrypt hash
DEV_AUTH_ENABLED = os.environ.get("CAAR_DEV_AUTH", "1") == "1"

# PHI redaction filter for logging
class PHIFilter(logging.Filter):
    def filter(self, record):
        record.msg = redact_phi(str(record.msg))
        return True

logging.getLogger().addFilter(PHIFilter())


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

# CAAR-CDSS pipeline initialization (reloaded for unified Chroma KB)
_pipeline: Pipeline | None = None
_feedback: list[dict[str, Any]] = []


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        prefer_real = os.environ.get("CAAR_USE_REAL", "0") == "1" or os.environ.get("CAAR_USE_REAL_LLM", "0") == "1"
        chroma_dir = Path("data/chroma_db")
        if prefer_real and chroma_dir.exists():
            _pipeline = Pipeline.from_corpus(chroma_dir=chroma_dir, prefer_real=True)
        else:
            _pipeline = Pipeline.from_unified_kb(prefer_real=prefer_real)
    return _pipeline


async def get_db_session():
    """FastAPI dependency for database session."""
    if _session_maker is None:
        yield None
        return
    async with _session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


def get_current_claims(authorization: str | None = Header(default=None)) -> dict:
    """Extract JWT claims. Returns dict with 'role' or 'roles' keys."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization[len("Bearer "):]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


def create_admin_token() -> str:
    """Create an admin token from environment variables. Only works if ADMIN_USERNAME and ADMIN_PASSWORD_HASH are set."""
    if not ADMIN_USERNAME or not ADMIN_PASSWORD_HASH:
        raise RuntimeError("Admin credentials not configured. Set CAAR_ADMIN_USER and CAAR_ADMIN_PASS_HASH.")
    token = jwt.encode(
        {"sub": ADMIN_USERNAME, "role": "admin", "exp": int(time.time()) + 3600},
        JWT_SECRET,
        algorithm=JWT_ALG
    )
    return token


def _hash_query(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()[:64]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "caar-cdss"}


@app.post("/login")
async def login(req: LoginRequest) -> dict[str, Any]:
    # Dev-only login: only enabled if CAAR_DEV_AUTH=1
    # In production: verify against user DB with proper password hashing
    if not DEV_AUTH_ENABLED:
        raise HTTPException(status_code=401, detail="Login disabled. Set CAAR_DEV_AUTH=1 for development.")
    
    # Dev mode: accept any non-empty credentials, but NEVER return role=admin
    # Admin tokens must be created via environment variables
    token = jwt.encode(
        {"sub": req.username, "role": "user", "exp": int(time.time()) + 1800},  # 30 min expiry
        JWT_SECRET,
        algorithm=JWT_ALG
    )
    return {"access_token": token, "token_type": "bearer", "expires_in": 1800}


@app.post("/analyze", dependencies=[Depends(analyze_limiter.dependency())])
async def analyze(
    req: AnalyzeRequest,
    claims: dict = Depends(get_current_claims),
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

    visit_vitals = None
    if req.vitals:
        if isinstance(req.vitals, dict):
            mapped_vitals = {
                "systolic_bp": req.vitals.get("systolic_bp") if req.vitals.get("systolic_bp") is not None else req.vitals.get("bloodPressureSystolic"),
                "diastolic_bp": req.vitals.get("diastolic_bp") if req.vitals.get("diastolic_bp") is not None else req.vitals.get("bloodPressureDiastolic"),
                "heart_rate": req.vitals.get("heart_rate") if req.vitals.get("heart_rate") is not None else req.vitals.get("heartRate"),
                "respiratory_rate": req.vitals.get("respiratory_rate") if req.vitals.get("respiratory_rate") is not None else req.vitals.get("respiratoryRate"),
                "temperature": req.vitals.get("temperature"),
                "oxygen_saturation": req.vitals.get("oxygen_saturation") if req.vitals.get("oxygen_saturation") is not None else req.vitals.get("oxygenSaturation"),
                "blood_glucose": req.vitals.get("blood_glucose") if req.vitals.get("blood_glucose") is not None else req.vitals.get("bloodGlucose"),
            }
            mapped_vitals = {k: v for k, v in mapped_vitals.items() if v is not None}
            visit_vitals = Vitals(**mapped_vitals) if mapped_vitals else Vitals()
        elif isinstance(req.vitals, Vitals):
            visit_vitals = req.vitals
        else:
            visit_vitals = Vitals()

    visit = Visit(vitals=visit_vitals) if visit_vitals else None

    aeb, resp = pipeline.analyze(req.query, visit=visit, patient=patient)

    # Prepare response dict
    response = {
        "primary_diagnosis": resp.primary_diagnosis,
        "confidence": round(resp.confidence, 4),
        "uncertainty": round(resp.uncertainty, 4),
        "hallucination_score": round(resp.hallucination_score, 4),
        "risk_level": (resp.risk_level.value if hasattr(resp.risk_level, "value") else str(resp.risk_level)).lower(),
        "decision": (resp.decision.value if hasattr(resp.decision, "value") else str(resp.decision)).lower(),
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
        "triage_reasons": getattr(resp, 'triage_reasons', []),
        "followup_questions": getattr(resp, 'followup_questions', []),
    }

    # Log query to database (optional - skip if db not available)
    if db is not None:
        try:
            query_hash = _hash_query(req.query)
            patient_dict = {
                "age": req.age,
                "gender": req.gender,
                "comorbidities": req.comorbidities,
                "medications": req.medications,
                "allergies": req.allergies,
                "vitals": visit_vitals.model_dump() if visit_vitals else (
                    req.vitals.model_dump() if hasattr(req.vitals, "model_dump") else req.vitals
                ),
            }
            user_id = claims.get("sub", "anonymous")
            await log_query(db, req.query, query_hash, response, patient_dict, user_id)
        except Exception as e:
            logging.getLogger(__name__).warning("Failed to log query to database: %s", e)

    return response


@app.post("/retrieve")
async def retrieve(
    req: AnalyzeRequest,
    claims: dict = Depends(get_current_claims),
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
async def metrics(claims: dict = Depends(get_current_claims)) -> dict[str, Any]:
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
    claims: dict = Depends(get_current_claims),
) -> dict[str, str]:
    _feedback.append({"query": req.query, "helpful": req.helpful, "comment": req.comment, "ts": time.time()})
    return {"status": "recorded"}


# === Admin-only endpoints ===

@app.post("/guidelines/upload", response_model=GuidelineUploadResponse, dependencies=[Depends(require_role("admin", get_current_claims))])
async def upload_guideline(
    source: str = Form(...),
    title: str | None = Form(None),
    year: int | None = Form(None),
    trust_score: float = Form(0.8),  # IGNORED - server assigns trust
    file: UploadFile = File(...),
    claims: dict = Depends(get_current_claims),
    db=Depends(get_db_session),
) -> GuidelineUploadResponse:
    """Ingest a PDF or text document into the corpus at runtime. Admin only."""
    pipeline = get_pipeline()
    kb = pipeline.kb

    content = await file.read()
    # Validate upload
    validate_upload(file.filename, content)

    text = ""
    if file.content_type == "application/pdf":
        import io
        import pdfplumber
        pdf_file = io.BytesIO(content)
        text_parts = []
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        text = "\n\n".join(text_parts)
        if not text.strip():
            text = content.decode("utf-8", errors="ignore")
    else:
        text = content.decode("utf-8", errors="ignore")

    if not text.strip():
        return GuidelineUploadResponse(
            status="error",
            chunks_added=0,
            document_id=title or file.filename,
        )

    # Server-side trust assignment (ignore client-sent trust_score)
    assigned_trust = trust_for_source(source, {"WHO": 0.95, "CDC": 0.95, "NICE": 0.93, "NIH": 0.93, "PubMed": 0.80})
    
    chunks = kb.add_documents(
        [
            {
                "text": text,
                "source": source,
                "title": title or file.filename,
                "year": year,
                "trust_score": assigned_trust,  # Use server-assigned trust
            }
        ]
    )

    return GuidelineUploadResponse(
        status="success",
        chunks_added=chunks,
        document_id=title or file.filename,
    )


@app.get("/query/{query_id}", response_model=QueryHistoryResponse)
async def get_query_history(
    query_id: UUID,
    claims: dict = Depends(get_current_claims),
    db=Depends(get_db_session),
) -> QueryHistoryResponse:
    """Retrieve a past query's full response and retrieved evidence. Users can only access their own queries; admins can access all."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    user_id = claims.get("sub", "anonymous")
    roles = claims.get("roles") or [claims.get("role", "user")]
    is_admin = "admin" in roles
    query_log = await get_query_log(db, query_id, user_id=user_id, is_admin=is_admin)
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
        followup_questions=getattr(query_log, 'followup_questions', []),
        triage_reasons=getattr(query_log, 'triage_reasons', []),
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


@app.post("/evaluation/run", response_model=EvaluationRunResponse, dependencies=[Depends(eval_limiter.dependency()), Depends(require_role("admin", get_current_claims))])
async def run_evaluation(
    request: EvaluationRunRequest,
    background_tasks: BackgroundTasks,
    claims: dict = Depends(get_current_claims),
    db=Depends(get_db_session),
) -> EvaluationRunResponse:
    """Trigger a benchmark evaluation run in the background. Admin only, rate limited."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    eval_id = await create_evaluation_run(
        db, request.method, request.benchmark, request.n_samples, request.config
    )

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
        started_at=datetime.now(timezone.utc),
        completed_at=None,
    )


@app.get("/evaluation/{eval_id}/results", response_model=EvaluationResultsResponse)
async def get_evaluation_results(
    eval_id: UUID,
    claims: dict = Depends(get_current_claims),
    db=Depends(get_db_session),
) -> EvaluationResultsResponse:
    """Fetch evaluation run results."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
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
    claims: dict = Depends(get_current_claims),
    db=Depends(get_db_session),
) -> ReviewResponse:
    """Submit human annotation for a query response or evaluation. Users can only review their own queries; admins can review all."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    from sqlalchemy import select

    user_id = claims.get("sub", "anonymous")
    roles = claims.get("roles") or [claims.get("role", "user")]
    is_admin = "admin" in roles

    query_exists = False
    eval_exists = False

    if request.hallucination_flag is not None:
        if response_id:
            stmt = select(QueryLog).where(QueryLog.id == response_id)
            if not is_admin:
                stmt = stmt.where(QueryLog.user_id == user_id)
            query_result = await db.execute(stmt)
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
        created_at=datetime.now(timezone.utc),
    )


@app.get("/corpus/stats", response_model=CorpusStatsResponse)
async def get_corpus_stats(
    claims: dict = Depends(get_current_claims),
    db=Depends(get_db_session),
) -> CorpusStatsResponse:
    """Get corpus statistics: size, chunk count, specialty coverage."""
    from sqlalchemy import select

    if db is not None:
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

    pipeline = get_pipeline()
    store = pipeline.kb.store

    if hasattr(store, "_chunks") and store._chunks:
        total_chunks = len(store._chunks)
        specialty_dist: dict[str, int] = {}
        source_dist: dict[str, int] = {}
        sources: set[str] = set()
        for c in store._chunks:
            spec = c.specialty or "general"
            src = c.source or "unknown"
            specialty_dist[spec] = specialty_dist.get(spec, 0) + 1
            source_dist[src] = source_dist.get(src, 0) + 1
            sources.add(src)

        if db is not None:
            await update_corpus_stats(db, total_chunks, len(sources), specialty_dist, source_dist)

        return CorpusStatsResponse(
            total_chunks=total_chunks,
            total_documents=len(sources),
            specialty_distribution=specialty_dist,
            source_distribution=source_dist,
            last_updated=datetime.now(timezone.utc),
        )

    if hasattr(store, "_collection"):
        total_chunks = store._collection.count()
        results = store._collection.get(include=["metadatas"])
        specialty_dist = {}
        source_dist = {}
        for meta in results.get("metadatas", []):
            spec = meta.get("specialty", "unknown")
            src = meta.get("source", "unknown")
            specialty_dist[spec] = specialty_dist.get(spec, 0) + 1
            source_dist[src] = source_dist.get(src, 0) + 1

        if db is not None:
            await update_corpus_stats(db, total_chunks, len(set(r.get("source") for r in results.get("metadatas", []))), specialty_dist, source_dist)

        return CorpusStatsResponse(
            total_chunks=total_chunks,
            total_documents=len(set(r.get("source") for r in results.get("metadatas", []))),
            specialty_distribution=specialty_dist,
            source_distribution=source_dist,
            last_updated=datetime.now(timezone.utc),
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