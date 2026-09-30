# CAAR-CDSS: Complete Project Analysis

## Executive Summary

**CAAR-CDSS (Confidence-Aware Adaptive Retrieval Clinical Decision Support System)** is a research-grade clinical AI system that implements **Adaptive Evidence Budgeting (AEB)** — a novel approach where retrieval iteratively expands until calibrated confidence reaches a threshold, rather than using a fixed top-k. The system combines hybrid retrieval (dense + sparse + cross-encoder rerank), NLI-based claim verification, confidence calibration, and risk-aware triage into a production-ready pipeline with FastAPI backend and React frontend.

**Key Innovation**: The AEB loop (src/confidence/aeb.py:62-174) dynamically adjusts evidence budget based on confidence signals, answering the clinically critical question: *"When does the AI have enough evidence to make a safe recommendation?"*

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CAAR-CDSS SYSTEM ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌─────────────────┐    ┌──────────────────────────┐  │
│  │   Frontend   │◄──►│    FastAPI      │◄──►│       Pipeline           │  │
│  │  (React/TS)  │    │     Backend     │    │  (src/confidence/pipeline)│  │
│  └──────────────┘    └─────────────────┘    └───────────┬──────────────┘  │
│                                                          │                 │
│                      ┌───────────────────────────────────┼───────────────┐ │
│                      ▼                                   ▼               ▼ │
│              ┌───────────────┐                  ┌───────────────┐  ┌─────────┐│
│              │  Knowledge    │                  │    AEB Loop   │  │ Verifier││
│              │  Base (KB)    │                  │  (Adaptive    │  │ (NLI)   ││
│              │  - ChromaDB   │                  │   Retrieval)  │  └─────────┘│
│              │  - BGE-M3     │                  └───────┬───────┘           │
│              │  - BM25       │                          │                 │
│              └───────────────┘                          ▼                 │
│                                                         ┌───────────────┐  │
│                                                         │    Reasoner   │  │
│                                                         │  (LLM/Mock)   │  │
│                                                         └───────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Modules Deep Dive

### 1. Knowledge Base (`src/kb/`)

#### 1.1 Chunker (`chunker.py`)
- **Purpose**: Splits clinical documents into overlapping chunks while preserving metadata
- **Algorithms**: 
  - Recursive word-based (default): ~512 tokens with 100 token overlap (approximates tokens by 1.3 words/token)
  - Sentence-boundary aware: maintains sentence integrity
- **Specialty Detection**: Keyword-based lightweight classifier (lines 82-101) mapping clinical terms to specialties (cardiology, pulmonology, etc.)
- **PDF Cleaning**: Strips headers, footers, page numbers, references sections (lines 140-147)

#### 1.2 Embedder (`embedder.py`)
Three embedding strategies with factory pattern (`make_embedder`):
| Embedder | Use Case | Dimensions | Why This Choice |
|----------|----------|------------|-----------------|
| `HashEmbedder` | Tests/CI/No GPU | 512 | Deterministic, dependency-free, preserves token overlap for keyword-heavy clinical queries |
| `SentenceTransformerEmbedder` | General biomedical | Varies | Uses `pritamdeka/S-PubMedBert-MS-MARCO` — PubMed-optimized |
| `BGE_M3Embedder` | **Production (default)** | 1024 | **BAAI/bge-m3** via FlagEmbedding — hybrid dense+sparse+colbert, SOTA on MTEB medical |

**Why BGE-M3 over alternatives?**
- `bge-m3` supports **hybrid retrieval natively** (dense + lexical + colbert) unlike BGE-small/v2
- 1024-dim provides better semantic resolution for clinical nuances
- FlagEmbedding's optimized inference (~2x faster than sentence-transformers)
- Falls back to sentence-transformers if FlagEmbedding unavailable

#### 1.3 Vector Store (`vector_store.py`)
Pluggable backends via `BaseVectorStore` interface:
| Backend | Storage | Search | Use Case |
|---------|---------|--------|----------|
| `InMemoryVectorStore` | RAM (pickle + numpy) | Cosine via matrix mul | Tests, mock pipeline, CI |
| `FAISSVectorStore` | Disk (FAISS index) | Inner product on normalized | Medium-scale, single-node |
| `ChromaVectorStore` | **ChromaDB (PersistentClient)** | HNSW cosine | **Production default** — persistent, metadata filtering, scales |

**Why ChromaDB over Qdrant/FAISS/Pinecone?**
- **No separate server** — runs embedded via `PersistentClient`
- **Metadata filtering built-in** — crucial for specialty/trust_score boosting
- **HNSW index** — fast approximate search at scale
- **Zero-config persistence** — ideal for research reproducibility

#### 1.4 Ingestion (`ingest.py`)
`KnowledgeBase` class orchestrates: documents → chunks → embeddings → vector store. Supports streaming from HF datasets, local directories, and seed corpus.

---

### 2. Retrieval (`src/retrieval/`)

#### 2.1 Dense Retrieval (`dense.py`)
Simple wrapper: query → embedder → vector store search. Handles `ChromaRetriever` compatibility.

#### 2.2 Sparse Retrieval (`sparse.py`)
**BM25Index** — pure Python Okapi BM25 implementation:
- Custom tokenizer with clinical stopword removal
- No external dependency (unlike Elasticsearch/rank-bm25)
- Standard parameters: k1=1.5, b=0.75
- In-memory, deterministic — critical for reproducibility

#### 2.3 Hybrid Retrieval (`hybrid.py`) — **Core Innovation**
`HybridRetriever` implements **Reciprocal Rank Fusion (RRF)**:

```python
# RRF Formula (lines 149-176)
score(i) = Σ_r 1 / (rrf_k + rank_r(i))  where rrf_k=60
```

**Why RRF over weighted sum?**
- Dense (cosine similarity ~0-1) and sparse (BM25 unbounded) scores are **incommensurable scales**
- RRF is **rank-based**, requiring no score normalization or hyperparameter tuning
- Proven robust in TREC/IR literature

**Multiplicative Boosting** (lines 73-102):
```python
new_score = base_score × (1 + α_trust × trust_score) × (1 + α_patient × patient_match)
```

**Why multiplicative not additive?**
- RRF base scores are small (~0.03); additive boost would drown relevance signal
- Multiplicative preserves RRF ranking order while tilting toward trusted/patient-relevant sources
- `α_trust=0.10`, `α_patient=0.05` — calibrated to not overpower relevance

**Patient Context Boosting**: Extracts demographic/comorbidity terms from profile (geriatric, pediatric, renal, pregnancy) and boosts chunks containing those terms.

#### 2.4 Reranking (`rerank.py`)
Two implementations via factory:
| Reranker | Model | Use Case |
|----------|-------|----------|
| `OverlapReranker` | Token overlap + trust | Mock/CI, deterministic |
| `CrossEncoderReranker` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Production — real cross-attention |

**Why this cross-encoder?**
- 6-layer MiniLM — fast (~5ms/query on GPU)
- MS-MARCO trained — strong on passage ranking
- Better than heuristic overlap for semantic matching

---

### 3. Agents (`src/agents/`)

#### 3.1 Context Builder (`context_builder.py`)
Deterministic rule-based patient profile extraction from free-text query:
- **Age**: Regex `(\d{1,3})\s*[-\s]?year[-\s]old`
- **Gender**: Male/female pronoun detection
- **Comorbidities**: Keyword matching against 10 common conditions
- **Pregnancy**: "pregnan" substring
- **Vitals → Risk Tokens**: Systolic <90 → "hypotension", O2 <92 → "hypoxia", HR >120 → "tachycardia"

**Why deterministic rule-based?**
- Reproducible without LLM calls
- No hallucination risk in profile extraction
- Fast (<1ms) — no GPU needed

#### 3.2 Reasoner (`reasoner.py`) — **Multi-Backend LLM Interface**

**Three Reasoner Implementations:**
| Backend | Class | Model | Quantization | Use Case |
|---------|-------|-------|--------------|----------|
| `mock` | `MockReasoner` | N/A | N/A | **Default** — CI, tests, no GPU |
| `local_4bit` | `OpenSourceLLMReasoner` | Llama-3.1-8B-Med | NF4 (bitsandbytes) | Local dev (RTX 4050 6GB) |
| `kaggle_fp16` | `OpenSourceLLMReasoner` | Llama-3.1-8B-Med | fp16 | Kaggle T4 (16GB) |
| `hf_inference_api` | `HFInferenceReasoner` | Llama-3.1-8B-Med | Serverless | Zero VRAM, RAGAS judging |

**MockReasoner Design (lines 45-206)** — *Deterministic clinical reasoning without LLM*:
- 36 diagnosis patterns with keyword regexes and prior probabilities
- Co-occurrence disambiguation (e.g., fever + flank pain → pyelonephritis boost 1.7x)
- Negation handling ("without flank pain" suppresses pyelonephritis)
- Evidence support scoring: `0.7 × coverage + 0.3 × relevance`
- Query-signal flag: diagnosis must match query text to commit primary
- Confidence = `0.5 + margin×0.6 + evidence_richness×0.15`

**Why this mock design?**
- Enables **full AEB pipeline testing without GPU**
- Produces differential diagnoses tied to evidence keywords
- Verification layer has meaningful claims to score
- Matches clinical reasoning patterns (pattern recognition + evidence weighting)

**OpenSourceLLMReasoner** (lines 247-299):
- 4-bit NF4 quantization via `BitsAndBytesConfig` for 6GB VRAM
- Multi-GPU memory split: GPU0=9GiB (Llama), GPU1=7GiB (DeBERTa verifier)
- Robust 3-tier JSON parser for unreliable small-model outputs

---

### 4. Verification (`src/verification/verifier.py`)

**Two Verifier Implementations:**

| Verifier | Model | Threshold | VRAM | Use Case |
|----------|-------|-----------|------|----------|
| `LexicalVerifier` | Token overlap (Jaccard + recall) | 0.26 entailment, 0.40 similarity | 0 GB | **Default** — CI, local, deterministic |
| `DeBERTaVerifier` | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | 0.7 entailment | ~1.8 GB fp16 | Kaggle T4, paper experiments |

**LexicalVerifier Design** (lines 72-127):
- Splits reasoning into atomic claims (sentence-level)
- Filters clinical stopwords ("patient presentation consistent diagnosis...")
- Scores: `0.8 × recall + 0.2 × Jaccard` against each evidence chunk
- Hallucination score = `1 - supported_claims / total_claims`

**DeBERTaVerifier** (lines 130-226):
- Batched inference: all (premise, hypothesis) pairs in single forward pass
- Loads on GPU1 (multi-GPU) to avoid OOM with Llama on GPU0
- Finds entailment label index from model config dynamically

**Factory** (`VerifierFactory.get_verifier`, line 229): Auto-selects DeBERTa if VRAM ≥ 10GB, else Lexical.

---

### 5. Confidence & AEB (`src/confidence/`)

#### 5.1 Estimation (`estimation.py`)
**Five Confidence Signals** (lines 14-21):
1. `retriever_top1` — Top-1 vs Top-5 similarity delta (flatness penalty)
2. `evidence_sufficiency` — Fraction of differentials with supporting evidence
3. `support_agreement` — Top-2 probability margin
4. `hallucination_penalty` — From NLI verification (capped at 0.5)
5. `raw_confidence` — LLM's self-reported confidence

**Why cap hallucination penalty at 0.5?** (line 91)
- Off-domain corpus can produce near-zero token overlap even for correct responses
- Prevents false-positive hallucination signals from driving confidence to zero
- Real hallucinations still produce meaningful penalty

**ConfidenceModel** (lines 72-110) — Weighted fusion:
```python
weights = [0.15, 0.30, 0.25, 0.15, 0.15]  # retriever, evidence, agreement, LLM, hallucination
values  = [retriever_top1, evidence_sufficiency, agreement, raw_confidence, 1-hallucination]
confidence = weighted_sum / sum(weights)
# Optional temperature scaling
```

**TemperatureScaler** (lines 113-132): Fits temperature via L-BFGS-B minimizing NLL on validation set.

#### 5.2 AEB Loop (`aeb.py`) — **Primary Research Contribution**
`AEBPipeline.run()` (lines 84-173):

```python
k = initial_k (3)
while True:
    evidence = retriever.retrieve(query, top_k=k)
    response = reasoner.generate(query, evidence)
    verification = verifier.verify(response.reasoning, evidence)
    signals = compute_signals(response, evidence, verification)
    confidence = confidence_model.score(signals)
    
    if confidence >= threshold (0.70):  # Stop: sufficient evidence
        break
    if k >= max_k (15) or rounds >= max_rounds (5):  # Stop: budget exhausted
        break
    k = min(k + step_k (3), max_k)
```

**Key Design Decisions:**
- **Initial k=3**: Start small — many simple queries need only 3 chunks
- **Step k=3**: Incremental expansion balances latency vs confidence
- **Threshold=0.70** (configurable): Calibrated achievable across MedQA + PubMedQA; 0.85 caused near-100% budget exhaustion
- **Hard cap at 5 rounds**: Prevents infinite loops
- **Accumulated evidence**: Each round adds to previous (not replace) — builds evidence base

#### 5.3 Triage (`triage.py`)
Rule-based safety overlay (lines 8-68):
1. **Hard Emergency Rules**: Systolic <90, O2 <90%, HR >150, high-risk chest pain, stroke signs, anaphylaxis, sepsis → **EMERGENCY + ESCALATE**
2. **Confidence < 0.70** → URGENT + ESCALATE
3. **Confidence 0.70-0.85** → ROUTINE + ASK_FOLLOWUP
4. **Confidence ≥ 0.85** → HOME_CARE + ANSWER

---

### 6. Pipeline Orchestrator (`src/confidence/pipeline.py`)

`Pipeline` class ties everything together:
- `from_seed()`: In-memory KB + MockReasoner + LexicalVerifier (zero setup)
- `from_corpus()`: ChromaDB + BGE-M3 + real LLM + DeBERTaVerifier (production)
- `analyze()`: Profile inference → AEB → Triage → ClinicalResponse

---

### 7. Data Layer (`src/data/`)

| Module | Purpose |
|--------|---------|
| `seed_corpus.py` | 24 curated clinical documents (WHO/CDC/NICE/PubMed) + 8 seed queries with ground truth |
| `guidelines_loader.py` | Streams `epfl-llm/guidelines` from HF, domain filtering, structure-aware chunking |
| `benchmarks.py` | Loaders for MedQA, PubMedQA, MedMCQA (JSONL from local cache) |
| `ingest_external.py` | Full ingestion pipeline: EPFL guidelines → ChromaDB + benchmark downloads |

---

### 8. Experiments & Evaluation (`src/experiments/`, `src/evaluation/`)

#### 8.1 Runner (`runner.py`)
Compares three methods on same queries:
| Method | Retrieval | Verification | AEB Loop |
|--------|-----------|--------------|----------|
| `vanilla` | Top-5 dense only | ❌ | ❌ |
| `hybrid` | Top-10 hybrid (dense+sparse+rerank) | ❌ | ❌ |
| `aeb` | Adaptive (3→15) | ✅ | ✅ |

**Robust Diagnosis Matching** (`_dx_matches`, lines 53-128):
- Exact string, substring, option-letter, differential scan, token overlap (Jaccard ≥0.5), binary decision cues (yes/no/maybe)
- **Critical**: Abstentions (no primary diagnosis) are **never correct** — prevents gaming via escalation

#### 8.2 RAGAS Evaluation (`ragas_eval.py`)
- **Metrics**: Faithfulness, Answer Relevancy, Context Precision/Recall
- **Judge**: Local pipeline (Kaggle) or HF Inference API (zero VRAM)
- **Embeddings**: BGE-small-en-v1.5 on **CPU** (fixes T4 GPU contention)
- **Critical Fix**: No `max_memory` in `pipeline()` call (prevents generate() crash)

#### 8.3 Evaluation Harness (`harness.py`)
Custom metrics without RAGAS dependency:
- Citation accuracy (claim-token overlap with contexts)
- Answer correctness (medical keyword overlap ≥30%)
- Abstention correctness (abstained when wrong, answered when right)

---

### 9. API Layer (`src/api/`)

**FastAPI Application** (`app.py`) with endpoints:
| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness probe |
| `POST /login` | Dev JWT (any credentials) |
| `POST /analyze` | **Main** — full AEB pipeline |
| `POST /retrieve` | Raw evidence only |
| `GET /metrics` | Aggregate evaluation on seed queries |
| `POST /feedback` | User helpfulness recording |
| `POST /guidelines/upload` | Runtime PDF/text ingestion |
| `GET /query/{id}` | Query history with full response |
| `POST /evaluation/run` | Background benchmark runs |
| `GET /evaluation/{id}/results` | Fetch evaluation results |
| `POST /review/{id}` | Human annotation (hallucination flag) |
| `GET /corpus/stats` | ChromaDB chunk/specialty/source counts |

**Database** (`database.py`): SQLAlchemy async + PostgreSQL (or SQLite for dev)
- Tables: `query_logs`, `evaluation_runs`, `reviews`, `corpus_stats`
- JSONB for flexible structured data (differential, evidence, config)

---

### 10. Frontend (`frontend/src/`)

**Stack**: React 18 + TypeScript + Vite + Tailwind CSS + TanStack Query + React Router + Zod

**Architecture**:
```
src/
├── components/
│   ├── ui/           # Reusable: Button, Card, Table, Modal, Input, Select, Badge, Skeleton
│   ├── layout/       # Header, Footer, PageContainer
│   ├── medical/      # Domain: ConfidenceBadge, ConsentModal, DisclaimerBanner
│   └── feedback/     # ErrorBoundary, LoadingOverlay
├── contexts/         # AuthProvider, ThemeProvider
├── hooks/            # useAnalyze, useAuth, useTheme, useCorpusStats
├── lib/              # api.ts (Axios), auth.ts, config.ts, queryClient.ts, validations.ts (Zod)
├── pages/            # PatientPortal, DoctorDashboard, AdminDashboard, CorpusStats, Login
├── styles/           # globals.css (Tailwind + medical color palette)
└── types/            # Shared TypeScript interfaces
```

**Key Pages**:
- **PatientPortal**: Symptom input form → analysis results with confidence badge, differential table, evidence audit trail
- **DoctorDashboard**: Preset clinical cases → AEB pipeline visualization (confidence curve, retrieval steps, evidence)
- **AdminDashboard**: Metrics dashboard (accuracy, hallucination, avg K, latency) with refresh

**Design System**: Medical-themed Tailwind palette (`medical-primary`, `medical-danger`, `medical-warning`, `medical-neutral`) with dark mode support.

---

## Technology Choices & Rationale

### Backend (Python)

| Category | Choice | Why Not Alternatives |
|----------|--------|---------------------|
| **Framework** | FastAPI | Async, auto OpenAPI, type-safe with Pydantic; Flask/Django too heavy/sync |
| **Vector DB** | ChromaDB | Embedded, no server, metadata filtering, HNSW; Qdrant/Pinecone need servers |
| **Dense Embeddings** | BGE-M3 (FlagEmbedding) | Hybrid native, 1024-dim, SOTA medical; BGE-small weaker, E5 needs more VRAM |
| **Sparse Retrieval** | Custom BM25 | No Elasticsearch server; rank-bm25 has C extensions (install issues) |
| **Cross-Encoder** | ms-marco-MiniLM-L-6-v2 | 6-layer, fast, MS-MARCO trained; larger models (BERT-base) too slow |
| **NLI Verification** | DeBERTa-v3-large-MNLI | Best open NLI; BERT-large-MNLI weaker on fever/anli |
| **LLM** | Llama-3.1-8B-Med-Instruct | Medical fine-tune, 8B fits 6GB@4-bit/16GB@fp16, permissive license; Phi-3/MedAlpaca weaker clinical |
| **Quantization** | bitsandbytes NF4 | 4-bit with double quant preserves quality; GPTQ/AWQ need calibration data |
| **Config** | Pydantic + YAML | Type-safe, validation, env override; raw dict/env-only lacks validation |
| **Seeds/Reproducibility** | Global seed function | torch/numpy/random + cudnn deterministic; hydra/omegaconf overkill |
| **Disk/GPU Utils** | Custom | Kaggle-specific (ENOSPC, T4 OOM); generic tools don't handle these |
| **Experiment Tracking** | MLflow + W&B | Both in reqs; MLflow local, W&B cloud — covers all needs |

### Frontend (TypeScript/React)

| Category | Choice | Why |
|----------|--------|-----|
| **Build** | Vite | Fast HMR, ES modules, smaller than CRA/Next.js for SPA |
| **State/Server** | TanStack Query | Caching, deduping, retries, devtools; Redux/SWR more complex/less features |
| **Forms** | React Hook Form + Zod | Type-safe validation, minimal re-renders; Formik/Yup heavier |
| **Styling** | Tailwind CSS | Utility-first, dark mode, medical palette via config; CSS-in-JS slower |
| **Routing** | React Router v6 | Standard, loader/action patterns; TanStack Router newer/less ecosystem |
| **Charts** | Custom SVG bars | Lightweight, no chart.js/recharts bundle; confidence curves simple |
| **Notifications** | Sonner | Toast library, accessible, Promise API; react-hot-toast less polished |
| **Icons** | Lucide React | Tree-shakable, consistent, 1000+ icons; Heroicons fewer medical icons |
| **Animation** | Framer Motion | Declarative, layout animations, exit transitions; CSS-only limited |
| **Testing** | Vitest + Playwright | Vite-native unit, real browser E2E; Jest/Cypress config overhead |

---

## Configuration System

### `configs/seeds.yaml` — Single Source of Truth
All hyperparameters, seeds, model names version-controlled:
```yaml
aeb:
  initial_k: 3
  step_k: 3
  max_k: 15
  confidence_threshold: 0.80  # Note: code default 0.70, seeds.yaml 0.80
retrieval:
  dense_top_k: 50
  sparse_top_k: 50
  rerank_top_k: 15
  final_top_k: 10
llm:
  model_name: "meta-llama/Llama-3.1-8B-Instruct"
  serving_backend: "kaggle_fp16"
embedding:
  model_name: "BAAI/bge-m3"
```

### Hardware Auto-Detection (`config.py:129-207`)
```python
# VRAM-based backend selection:
≥ 20 GB  → kaggle_fp16 (or multi-GPU)
12-19 GB → kaggle_fp16 (single T4)
< 10 GB  → local_4bit + hf_inference_api for verification/RAGAS
No GPU   → hf_inference_api for everything
```

### Kaggle Safe Defaults (`config.py:34-47`)
Reduced workload for T4 time/disk limits:
- `n_queries: 50` (down from 200)
- `max_new_tokens: 128`
- `ragas_max_workers: 1` (sequential prevents OOM)
- `ingest_limit: 2000` (down from 5000)

---

## Key Design Patterns

### 1. Factory Pattern with Graceful Degradation
Every major component has `make_X(prefer_real=bool)` factory:
- `make_embedder`, `make_reranker`, `make_reasoner`, `make_verifier`, `make_vector_store`
- **Mock/fallback always works** — real backend failures raise only when explicitly requested
- Enables `python -m src.cli query "..." --mode mock` anywhere

### 2. Dependency Injection in Pipeline
`AEBPipeline` takes `retriever`, `reasoner`, `confidence_model`, `verifier`, `config` — all swappable for testing/ablation.

### 3. Dataclass Configuration
`AEBConfig`, `RetrievalConfig`, `HybridConfig`, `ConfidenceSignals` — immutable, typed, self-documenting.

### 4. Strategy Pattern for Verification/Reasoning
`BaseVerifier`/`BaseReasoner` interfaces with multiple implementations selected at runtime.

---

## Data Flow: Query → Response

```
User Query: "55yo diabetic with chest pain, sweating, SOB"
        │
        ▼
┌────────────────────────────────────────┐
│  Context Builder (infer_profile)       │
│  → age=55, gender=male, comorbidities  │
│    =["diabetes"]                       │
└────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────┐
│  Patient Context Query Augmentation    │
│  "55yo diabetic... 55 year old adult   │
│   male diabetes"                       │
└────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────┐
│  AEB ROUND 1 (k=3)                     │
│  HybridRetriever.retrieve()            │
│  → Dense (BGE-M3) Top-50               │
│  → Sparse (BM25) Top-50                │
│  → RRF Fusion                          │
│  → Trust/Patient Boosting              │
│  → CrossEncoder Rerank Top-15          │
│  → Final Top-3 Evidence Chunks         │
└────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────┐
│  Reasoner (Mock/LLM)                   │
│  → Pattern match evidence              │
│  → Differential: ACS (0.82), etc.      │
│  → Confidence: 0.78                    │
│  → Reasoning text with citations       │
└────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────┐
│  Verification (Lexical/DeBERTa)        │
│  → Split reasoning into claims         │
│  → Each claim vs evidence chunks       │
│  → Hallucination Score: 0.12           │
└────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────┐
│  Confidence Fusion                     │
│  Signals: retriever=0.72, evidence=1.0,│
│           agreement=0.85, LLM=0.78,    │
│           hallucination=0.88 (1-0.12)  │
│  Weighted → 0.81                       │
└────────────────────────────────────────┘
        │
        ▼
   Confidence 0.81 ≥ Threshold 0.70? → YES → STOP
        │
        ▼
┌────────────────────────────────────────┐
│  Triage                                │
│  No emergency vitals                   │
│  Confidence 0.81 ∈ [0.70, 0.85)        │
│  → Risk: ROUTINE, Decision: ASK_FOLLOWUP│
└────────────────────────────────────────┘
        │
        ▼
ClinicalResponse with full audit trail
```

---

## Evaluation Methodology

### Benchmarks
| Benchmark | Size | Type | Source |
|-----------|------|------|--------|
| MedQA | ~1200 | USMLE 4-option MCQ | openlifescienceai/medqa |
| PubMedQA | ~500 | Yes/No/Maybe QA | qiaojin/PubMedQA |
| MedMCQA | ~4000 | Indian medical MCQ | openlifescienceai/medmcqa |
| Seeds | 8 | Curated clinical cases | Internal |

### Metrics
| Category | Metrics |
|----------|---------|
| **Clinical** | Top-1 Accuracy, Top-3 Accuracy |
| **Safety** | Hallucination Rate, Escalation Precision, False Reassurance Rate |
| **Calibration** | ECE, Brier Score |
| **Efficiency** | Avg Retrieval K, Latency (ms), Cost/Query |
| **RAGAS** | Faithfulness, Answer Relevancy, Context Precision/Recall |

### Ablation Studies (Planned)
1. **Threshold Sweep**: 0.50→0.95 → Abstention-Accuracy curve
2. **Embedding Ablation**: BGE-M3 vs BGE-small vs PubMedBERT
3. **Retrieval Ablation**: Vanilla vs Hybrid vs AEB
4. **Verification Ablation**: Lexical vs DeBERTa vs None

---

## Reproducibility Features

1. **Locked Dependencies**: `requirements-locked.txt` + `uv.lock`
2. **Global Seeds**: `set_global_seeds(42)` called at pipeline entry points
3. **Deterministic Mocks**: `HashEmbedder`, `MockReasoner`, `LexicalVerifier`, `OverlapReranker`
4. **Config Versioning**: All hyperparameters in `configs/seeds.yaml` (git-tracked)
5. **Hardware-Adaptive**: `detect_hardware()` selects optimal backend automatically
6. **Kaggle Notebook Template**: `notebooks/kaggle_eval_template.ipynb` for paper experiments

---

## Safety & Compliance

### Hard Constraints (Enforced in Code)
- **Never prescribes** — no drug dosing/output
- **Never replaces physician** — escalation pathway mandatory
- **Always disclaimers** — `safety_disclaimer` on every response (models.py:93-96)
- **Emergency detection** — Rule-based vitals/symptom triggers override everything

### Risk Levels
| Level | Criteria | Action |
|-------|----------|--------|
| EMERGENCY | Hard vitals rules OR high-risk patterns | Immediate ESCALATE |
| URGENT | Confidence < 0.70 | ESCALATE |
| ROUTINE | Confidence 0.70-0.85 | ASK_FOLLOWUP |
| HOME_CARE | Confidence ≥ 0.85 | ANSWER |

---

## Known Limitations & TODOs

From `Docs/Basic design doc.md` and code comments:

1. **RL Adaptive Policy** — Currently rule-based; learnable PPO policy planned (Phase 5)
2. **Drug Safety Agent** — Not yet implemented (Agent 3 in detailed doc)
3. **Multi-Agent Consistency** — Single reasoner; consistency agent planned
4. **PostgreSQL Dependency** — API requires DB; SQLite fallback for dev would help
5. **PDF Ingestion** — `/guidelines/upload` treats PDF as text; needs pdfplumber integration
6. **Threshold Calibration** — Default 0.70 vs seeds.yaml 0.80 discrepancy
7. **Real Corpus Scale** — EPFL guidelines ingestion not yet run at full scale (5000 docs)
8. **RAGAS Stability** — HF Inference API provider/model availability unreliable

---

## Running the System

### Quick Start (Mock, No GPU)
```bash
# 1. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run query
python -m src.cli query "55yo diabetic with chest pain" --mode mock

# 3. Launch API
uvicorn src.api.app:app --reload

# 4. Launch Frontend
cd frontend && npm install && npm run dev
```

### Real LLM (Local RTX 4050)
```bash
python -m src.cli query "55yo diabetic..." --mode local_4bit
# or
uvicorn src.api.app:app --reload  # with prefer_real=True in pipeline
```

### Paper Experiments (Kaggle T4)
1. Upload repo to Kaggle
2. Run `notebooks/kaggle_eval_template.ipynb`
3. Download results from `/kaggle/working/results/`

### RAGAS Evaluation
```bash
# Requires HF_TOKEN in .env or configs/secrets.yaml
python -m src.experiments.ragas_eval --benchmark medqa --n 50 --mode kaggle_fp16
```

---

## File Structure Summary

```
Capstone/
├── src/
│   ├── agents/           # Context builder, Reasoner (Mock + LLM backends)
│   ├── api/              # FastAPI app, DB models, Pydantic schemas
│   ├── confidence/       # AEB loop, Confidence fusion, Triage, Chroma pipeline
│   ├── data/             # Seed corpus, Guidelines loader, Benchmarks, Ingestion
│   ├── evaluation/       # Custom harness (no RAGAS dependency)
│   ├── experiments/      # Runner (vanilla/hybrid/aeb), RAGAS evaluation
│   ├── kb/               # Chunker, Embedder (Hash/BGE-M3), Ingest, Vector stores
│   ├── retrieval/        # Dense, Sparse (BM25), Hybrid (RRF), Rerank, Chroma retriever
│   ├── utils/            # Disk/GPU memory management, progress formatting
│   ├── verification/     # NLI verifier (Lexical + DeBERTa)
│   ├── cli.py            # Command-line interface
│   ├── config.py         # Config loading, hardware detection, seeds
│   └── models.py         # Pydantic domain models
├── frontend/             # React + TypeScript + Tailwind dashboard
├── configs/              # seeds.yaml, kb.yaml, secrets.example.yaml
├── data/                 # External datasets (gitignored), chroma_db (gitignored)
├── notebooks/            # Kaggle evaluation template
├── experiments/          # Results, ablations (gitignored)
├── paper/                # IEEE paper draft
├── Docs/                 # Design docs, timeline
├── tests/                # Unit tests
└── requirements.txt      # Python dependencies
```

---

## Conclusion

CAAR-CDSS is a **well-architected research prototype** that demonstrates Adaptive Evidence Budgeting as a practical, clinically-motivated alternative to fixed-top-k RAG. The codebase exhibits:

- **Clean separation of concerns** — each module has single responsibility
- **Graceful degradation** — works fully in mock mode, scales to real models
- **Reproducibility-first design** — seeds, deterministic mocks, hardware auto-config
- **Safety by default** — emergency rules, escalation, disclaimers baked in
- **Evaluation rigor** — multiple baselines, custom + RAGAS metrics, ablation framework

The system is ready for paper experiments on Kaggle T4 and provides a solid foundation for the planned RL-based adaptive retrieval policy and drug safety agent extensions.