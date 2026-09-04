"""PostgreSQL database layer for CAAR-CDSS API."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    primary_diagnosis: Mapped[str | None] = mapped_column(String(255))
    confidence: Mapped[float | None] = mapped_column()
    uncertainty: Mapped[float | None] = mapped_column()
    hallucination_score: Mapped[float | None] = mapped_column()
    risk_level: Mapped[str | None] = mapped_column(String(50))
    decision: Mapped[str | None] = mapped_column(String(50))
    escalated_reason: Mapped[str | None] = mapped_column(Text)
    retrieval_k_used: Mapped[int | None] = mapped_column()
    retrieval_steps: Mapped[int | None] = mapped_column()
    confidence_curve: Mapped[list[float] | None] = mapped_column(JSONB)
    differential: Mapped[list[dict] | None] = mapped_column(JSONB)
    evidence: Mapped[list[dict] | None] = mapped_column(JSONB)
    reasoning: Mapped[str | None] = mapped_column(Text)
    patient_age: Mapped[int | None] = mapped_column()
    patient_gender: Mapped[str | None] = mapped_column(String(50))
    patient_comorbidities: Mapped[list[str] | None] = mapped_column(JSONB)
    patient_medications: Mapped[list[str] | None] = mapped_column(JSONB)
    patient_allergies: Mapped[list[str] | None] = mapped_column(JSONB)
    vitals: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    reviews: Mapped[list[Review]] = relationship(back_populates="query_log")


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    method: Mapped[str] = mapped_column(String(50), nullable=False)
    benchmark: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    n_samples: Mapped[int] = mapped_column(default=0)
    config: Mapped[dict | None] = mapped_column(JSONB)
    results: Mapped[dict | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    reviews: Mapped[list[Review]] = relationship(back_populates="evaluation")


class Review(Base):
    __tablename__ = "reviews"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    query_log_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("query_logs.id"), nullable=True
    )
    evaluation_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("evaluation_runs.id"), nullable=True
    )
    hallucination_flag: Mapped[bool | None] = mapped_column()
    notes: Mapped[str | None] = mapped_column(Text)
    reviewer: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    query_log: Mapped[QueryLog | None] = relationship(back_populates="reviews")
    evaluation: Mapped[EvaluationRun | None] = relationship(back_populates="reviews")


class CorpusStats(Base):
    __tablename__ = "corpus_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    total_chunks: Mapped[int] = mapped_column(default=0)
    total_documents: Mapped[int] = mapped_column(default=0)
    specialty_distribution: Mapped[dict] = mapped_column(JSONB, default={})
    source_distribution: Mapped[dict] = mapped_column(JSONB, default={})
    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


async def create_engine_and_session(
    database_url: str | None = None,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """Create async PostgreSQL engine and session maker."""
    if database_url is None:
        database_url = os.environ.get(
            "DATABASE_URL",
            "postgresql+asyncpg://postgres:postgres@localhost:5432/caar_cdss",
        )

    engine = create_async_engine(
        database_url,
        echo=os.environ.get("SQL_ECHO", "false").lower() == "true",
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )

    async_session_maker = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    return engine, async_session_maker


async def init_db(engine: AsyncEngine) -> None:
    """Create all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session(
    session_maker: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database session."""
    async with session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def log_query(
    session: AsyncSession,
    query_text: str,
    query_hash: str,
    response: dict,
    patient: dict | None = None,
) -> UUID:
    """Log a query and its response to the database."""
    diff = response.get("differential", [])
    ev = response.get("evidence", [])

    query_log = QueryLog(
        query_text=query_text,
        query_hash=query_hash,
        primary_diagnosis=response.get("primary_diagnosis"),
        confidence=response.get("confidence"),
        uncertainty=response.get("uncertainty"),
        hallucination_score=response.get("hallucination_score"),
        risk_level=response.get("risk_level"),
        decision=response.get("decision"),
        escalated_reason=response.get("escalated_reason"),
        retrieval_k_used=response.get("retrieval_k_used"),
        retrieval_steps=response.get("retrieval_steps"),
        confidence_curve=response.get("confidence_curve"),
        differential=diff,
        evidence=ev,
        reasoning=response.get("reasoning"),
        patient_age=patient.get("age") if patient else None,
        patient_gender=patient.get("gender") if patient else None,
        patient_comorbidities=patient.get("comorbidities") if patient else None,
        patient_medications=patient.get("medications") if patient else None,
        patient_allergies=patient.get("allergies") if patient else None,
        vitals=patient.get("vitals") if patient else None,
    )
    session.add(query_log)
    await session.flush()
    return query_log.id


async def get_query_log(session: AsyncSession, query_id: UUID) -> QueryLog | None:
    """Retrieve a query log by ID."""
    from sqlalchemy import select
    result = await session.execute(select(QueryLog).where(QueryLog.id == query_id))
    return result.scalar_one_or_none()


async def create_evaluation_run(
    session: AsyncSession,
    method: str,
    benchmark: str,
    n_samples: int,
    config: dict | None = None,
) -> UUID:
    """Create a new evaluation run record."""
    eval_run = EvaluationRun(
        method=method,
        benchmark=benchmark,
        n_samples=n_samples,
        config=config,
        status="pending",
    )
    session.add(eval_run)
    await session.flush()
    return eval_run.id


async def update_evaluation_run(
    session: AsyncSession,
    eval_id: UUID,
    status: str,
    results: dict | None = None,
    error: str | None = None,
) -> None:
    """Update evaluation run status and results."""
    from sqlalchemy import select
    result = await session.execute(
        select(EvaluationRun).where(EvaluationRun.id == eval_id)
    )
    eval_run = result.scalar_one_or_none()
    if eval_run:
        eval_run.status = status
        if results:
            eval_run.results = results
        if error:
            eval_run.error = error
        if status == "completed":
            eval_run.completed_at = datetime.utcnow()
        await session.flush()


async def add_review(
    session: AsyncSession,
    query_log_id: UUID | None = None,
    evaluation_id: UUID | None = None,
    hallucination_flag: bool | None = None,
    notes: str | None = None,
    reviewer: str | None = None,
) -> UUID:
    """Add a human review annotation."""
    review = Review(
        query_log_id=query_log_id,
        evaluation_id=evaluation_id,
        hallucination_flag=hallucination_flag,
        notes=notes,
        reviewer=reviewer,
    )
    session.add(review)
    await session.flush()
    return review.id


async def update_corpus_stats(
    session: AsyncSession,
    total_chunks: int,
    total_documents: int,
    specialty_dist: dict,
    source_dist: dict,
) -> None:
    """Update or create corpus statistics."""
    from sqlalchemy import select
    result = await session.execute(select(CorpusStats).limit(1))
    stats = result.scalar_one_or_none()
    if stats is None:
        stats = CorpusStats()
        session.add(stats)
    stats.total_chunks = total_chunks
    stats.total_documents = total_documents
    stats.specialty_distribution = specialty_dist
    stats.source_distribution = source_dist
    await session.flush()
