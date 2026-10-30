"""Chroma-based pipeline for CAAR-CDSS."""

from __future__ import annotations

from pathlib import Path

from ..agents.context_builder import infer_profile
from ..agents.reasoner import make_reasoner
from ..config import get_app_config, set_global_seeds
from ..kb.embedder import make_embedder
from ..kb.vector_store import make_vector_store
from ..models import ClinicalResponse, PatientProfile, TriageDecision, Visit
from ..retrieval.hybrid import HybridConfig, HybridRetriever
from ..retrieval.rerank import make_reranker
from ..retrieval.sparse import BM25Index
from ..verification.verifier import make_verifier
from .aeb import AEBPipeline, AEBResult
from .estimation import ConfidenceModel
from .triage import assign_risk


class ChromaPipeline:
    """Pipeline that uses Chroma DB for retrieval."""

    def __init__(
        self,
        retriever: HybridRetriever,
        aeb: AEBPipeline,
    ):
        self.retriever = retriever
        self.aeb = aeb
        self.reasoner = aeb.reasoner

    @classmethod
    def from_chroma(cls, prefer_real: bool = True, reasoner_backend: str = "mock", chroma_dir: str | Path = "data/chroma_db") -> ChromaPipeline:
        """Build pipeline from Chroma DB.

        Args:
            prefer_real: If True, use real SentenceTransformer embedder to match ChromaDB dimension.
                        If False, use HashEmbedder (512-dim) - will fail if ChromaDB has different dim.
            reasoner_backend: Backend for the reasoner ("mock", "local_4bit", "kaggle_fp16", "hf_api").
            chroma_dir: Path to ChromaDB persistence directory.
        """
        set_global_seeds(42)
        cfg = get_app_config()

        # Use ChromaVectorStore for dense search (via factory)
        chroma_store = make_vector_store(backend="chroma", persist_dir=chroma_dir)

        # Use the configured embedder (must match the dimension of existing ChromaDB)
        # Default to prefer_real=True to get the real SentenceTransformer embedder
        embedder = make_embedder(cfg.embedding.model_name, prefer_real=True)

        # BM25 index for sparse retrieval (we'll create from Chroma data)
        bm25 = BM25Index()

        # Get all chunks from Chroma for BM25
        all_results = chroma_store._collection.get()
        if all_results and all_results["documents"]:
            chunks = []
            for doc, metadata in zip(all_results["documents"], all_results["metadatas"], strict=False):
                import uuid

                from ..models import EvidenceChunk
                chunk = EvidenceChunk(
                    chunk_id=uuid.uuid4(),
                    text=doc,
                    source=metadata.get("source", "unknown"),
                    title=metadata.get("title"),
                    year=metadata.get("year"),
                    specialty=metadata.get("specialty"),
                    trust_score=metadata.get("trust_score", 0.8),
                )
                chunks.append(chunk)
            bm25.add_chunks(chunks)

        # Hybrid retriever combining Chroma (dense) + BM25 (sparse)
        reranker = make_reranker(prefer_real=prefer_real)
        retriever = HybridRetriever(
            embedder=embedder,
            store=chroma_store,  # ChromaVectorStore IS a BaseVectorStore
            bm25=bm25,
            reranker=reranker,
            config=HybridConfig(final_top_k=cfg.retrieval.final_top_k),
        )

        reasoner = make_reasoner(backend=reasoner_backend, prefer_real=(reasoner_backend != "mock"), model_name=cfg.llm.model_name)
        verifier = make_verifier(prefer_real=prefer_real)
        confidence_model = ConfidenceModel()
        aeb = AEBPipeline(
            retriever=retriever,
            reasoner=reasoner,
            confidence_model=confidence_model,
            verifier=verifier,
        )
        return cls(
            retriever=retriever,
            aeb=aeb,
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
        response = assign_risk(
            aeb_result.response,
            profile=patient,
            vitals=visit.vitals if visit else None,
            query=query,
        )
        return aeb_result, response


def run_chroma_evaluation(max_samples: int = 20):
    """Run evaluation on Chroma pipeline."""
    from src.data.ingest_external import ExternalDataIngestor, IngestionConfig
    from src.evaluation.harness import EvaluationHarness

    print("=== Initializing Chroma Pipeline ===")
    pipeline = ChromaPipeline.from_chroma()

    print("=== Initializing Ingestor ===")
    config = IngestionConfig()
    ingestor = ExternalDataIngestor(config)

    print("=== Creating Evaluation Harness ===")
    harness = EvaluationHarness(pipeline, ingestor)

    # Run on our seed queries (clinical triage cases)
    from src.data.seed_corpus import SEED_PATIENT_QUERIES

    print(f"\n=== Evaluating on {len(SEED_PATIENT_QUERIES)} seed clinical queries ===")
    results = []
    for i, q in enumerate(SEED_PATIENT_QUERIES[:max_samples]):
        query = q["query"]
        expected = q["expected_dx"]
        print(f"[{i+1}/{min(max_samples, len(SEED_PATIENT_QUERIES))}] {query[:80]}...")

        try:
            aeb_result, response = pipeline.analyze(query)

            from src.evaluation.harness import EvalResult
            result = EvalResult(
                query=query,
                expected_answer=expected,
                generated_answer=response.reasoning,
                retrieved_contexts=[c.text for c in response.evidence],
                confidence=response.confidence,
                hallucination_score=response.hallucination_score,
                abstained=response.decision == TriageDecision.ESCALATE,
            )
            result.answer_correct = (response.primary_diagnosis == expected)
            result.abstention_correct = harness._check_abstention_correct(result.abstained, result.answer_correct)
            result.citation_accuracy = harness._compute_citation_accuracy(response.reasoning, [c.text for c in response.evidence])

            results.append(result)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    agg = harness.aggregate_results(results)
    harness.save_results(results, "chroma_clinical", Path("experiments/results"))

    print("\n=== CHROMA CLINICAL RESULTS ===")
    for k, v in agg.items():
        print(f"  {k}: {v}")

    return agg


if __name__ == "__main__":
    run_chroma_evaluation(max_samples=20)
