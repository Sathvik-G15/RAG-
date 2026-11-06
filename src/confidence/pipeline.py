"""Pipeline orchestrator: ties KB + retrieval + AEB + verification + triage together."""
from __future__ import annotations

import logging
from pathlib import Path

from ..agents.context_builder import infer_profile
from ..agents.reasoner import make_reasoner
from ..config import get_app_config, set_global_seeds
from ..kb.embedder import make_embedder
from ..kb.ingest import KnowledgeBase
from ..kb.vector_store import make_vector_store
from ..models import ClinicalResponse, PatientProfile, Visit, TriageDecision, RiskLevel
from ..retrieval.hybrid import HybridConfig, HybridRetriever
from ..retrieval.rerank import make_reranker
from ..verification.verifier import make_verifier
from .aeb import AEBPipeline, AEBResult
from .estimation import ConfidenceModel
from .triage import triage, TriageInput, TriageConfig, TriageResult

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        kb: KnowledgeBase,
        retriever: HybridRetriever,
        aeb: AEBPipeline,
        triage_config: TriageConfig,
        prefer_real_llm: bool = False,
        model_name: str | None = None,
    ):
        self.kb = kb
        self.retriever = retriever
        self.aeb = aeb
        self.triage_config = triage_config
        self.reasoner = aeb.reasoner
        self.model_name = model_name

    @classmethod
    def from_unified_kb(cls, kb_path: str | Path | None = None, prefer_real: bool = False) -> Pipeline:
        """Build pipeline from the single unified clinical knowledge base."""
        from ..data.unified_kb import load_unified_kb

        set_global_seeds(42)
        cfg = get_app_config()

        documents = load_unified_kb(kb_path)
        embedder = make_embedder(cfg.embedding.model_name, prefer_real=prefer_real)
        store = make_vector_store(backend="memory")
        kb = KnowledgeBase(embedder=embedder, store=store)
        kb.add_documents(documents)

        reranker = make_reranker(prefer_real=prefer_real)
        retriever = HybridRetriever(
            embedder=embedder,
            store=store,
            reranker=reranker,
            config=HybridConfig(final_top_k=cfg.retrieval.final_top_k),
        )
        retriever.ensure_bm25_indexed()

        reasoner = make_reasoner(prefer_real=prefer_real, model_name=cfg.llm.model_name)
        verifier = make_verifier(prefer_real=prefer_real)
        confidence_model = ConfidenceModel()
        aeb = AEBPipeline(
            retriever=retriever,
            reasoner=reasoner,
            confidence_model=confidence_model,
            verifier=verifier,
            config=cfg.aeb,
        )
        # Convert triage config from dict to TriageConfig
        triage_cfg = TriageConfig(**cfg.triage) if isinstance(cfg.triage, dict) else cfg.triage
        return cls(
            kb=kb,
            retriever=retriever,
            aeb=aeb,
            triage_config=triage_cfg,
            prefer_real_llm=prefer_real,
            model_name=cfg.llm.model_name,
        )

    @classmethod
    def from_seed(cls, prefer_real: bool = False) -> Pipeline:
        """Build pipeline (delegates to the single unified clinical knowledge base)."""
        return cls.from_unified_kb(prefer_real=prefer_real)

    @classmethod
    def from_corpus(
        cls,
        corpus_name: str = "epfl-llm/guidelines",
        domain: str | None = "infectious_disease",
        prefer_real: bool = True,
        chroma_dir: str | Path = "data/chroma_db",
        limit: int | None = None,
    ) -> Pipeline:
        """Build pipeline from real guidelines corpus with ChromaDB and BGE-M3."""
        from ..data.guidelines_loader import load_and_chunk_guidelines

        set_global_seeds(42)
        cfg = get_app_config()

        embedder = make_embedder(cfg.embedding.model_name, prefer_real=prefer_real)
        store = make_vector_store(backend="chroma", persist_dir=chroma_dir)
        kb = KnowledgeBase(embedder=embedder, store=store)

        # Ingest chunks into Chroma if empty
        if len(store) == 0:
            logger.info("ChromaDB is empty. Loading guidelines from %s (domain=%s)...", corpus_name, domain)
            chunks = load_and_chunk_guidelines(domain=domain, limit=limit)
            if chunks:
                texts = [c.text for c in chunks]
                embs = embedder.embed(texts)
                store.add(chunks, embs)
                logger.info("Indexed %d chunks into ChromaDB at %s", len(chunks), chroma_dir)

        reranker = make_reranker(prefer_real=prefer_real)
        retriever = HybridRetriever(
            embedder=embedder,
            store=store,
            reranker=reranker,
            config=HybridConfig(final_top_k=cfg.retrieval.final_top_k),
        )
        retriever.ensure_bm25_indexed()

        reasoner = make_reasoner(
            backend=cfg.llm.serving_backend if prefer_real else "mock",
            prefer_real=prefer_real,
            model_name=cfg.llm.model_name,
        )
        verifier = make_verifier(prefer_real=prefer_real)
        confidence_model = ConfidenceModel()
        aeb = AEBPipeline(
            retriever=retriever,
            reasoner=reasoner,
            confidence_model=confidence_model,
            verifier=verifier,
            config=cfg.aeb,
        )
        triage_cfg = TriageConfig(**cfg.triage) if isinstance(cfg.triage, dict) else cfg.triage
        return cls(
            kb=kb,
            retriever=retriever,
            aeb=aeb,
            triage_config=triage_cfg,
            prefer_real_llm=prefer_real,
            model_name=cfg.llm.model_name,
        )

    def analyze(
        self,
        query: str,
        visit: Visit | None = None,
        patient: PatientProfile | None = None,
    ) -> tuple[AEBResult, ClinicalResponse]:
        """Run the full pipeline: profile -> AEB -> triage."""
        if patient is None:
            patient = infer_profile(query)
        
        aeb_result = self.aeb.run(query, visit=visit, patient=patient)
        response = aeb_result.response
        
        # Build triage input from patient profile and visit
        vitals = visit.vitals if visit else None
        triage_input = TriageInput(
            age_years=patient.age,
            sbp=vitals.systolic_bp if vitals else None,
            hr=vitals.heart_rate if vitals else None,
            spo2=vitals.oxygen_saturation if vitals else None,
            rr=vitals.respiratory_rate if vitals else None,
            temp_c=vitals.temperature if vitals else None,
            avpu=getattr(vitals, 'avpu', None) if vitals else None,
            pregnant=patient.pregnancy_status,
            symptom_text=query,
        )
        
        # Get differential as list of (diagnosis, probability) tuples
        differential = [(d.diagnosis, d.probability) for d in response.differential]
        
        # Run new triage
        triage_result: TriageResult = triage(
            triage_input,
            differential,
            response.confidence,
            self.triage_config,
        )
        
        # Update response with triage results
        if isinstance(triage_result.risk_level, str):
            try:
                response.risk_level = RiskLevel(triage_result.risk_level.lower())
            except ValueError:
                response.risk_level = RiskLevel.ROUTINE
        else:
            response.risk_level = triage_result.risk_level

        if isinstance(triage_result.decision, str):
            try:
                response.decision = TriageDecision(triage_result.decision.lower())
            except ValueError:
                response.decision = TriageDecision.ESCALATE
        else:
            response.decision = triage_result.decision

        response.triage_reasons = triage_result.reasons
        response.followup_questions = triage_result.followup_questions
        
        # Set escalated_reason from triage reasons if triage escalated and AEB didn't set one
        if triage_result.decision == TriageDecision.ESCALATE and not response.escalated_reason:
            response.escalated_reason = "; ".join(triage_result.reasons) + " — clinician review advised."
        
        return aeb_result, response
