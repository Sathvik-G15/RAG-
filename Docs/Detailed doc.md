Here is your complete design document formatted as a **Markdown file**.

You can copy this directly into `PROJECT_DESIGN.md`.

---

# Adaptive Multi-Agent Confidence-Aware Clinical RAG System  
## Full Technical Design Document

---

# 1. Project Overview

## 1.1 Objective

Develop a **Confidence-Aware Adaptive Retrieval Clinical Decision Support System (CAAR-CDSS)** with:

- Adaptive retrieval based on uncertainty
- Hybrid RAG (dense + sparse + re-ranking)
- Multi-agent reasoning
- Hallucination detection
- Confidence calibration
- Risk-aware triage
- Escalation to clinician when unsafe

---

# 2. System Architecture

## 2.1 High-Level Architecture

```
Frontend / UI
      │
      ▼
API Gateway
      │
      ▼
Pipeline Orchestrator
      │
      ▼
Patient Profiler
      │
      ▼
Disease Probability Estimator
      │
      ▼
Uncertainty Estimator
      │
      ▼
Adaptive Retrieval Policy
      │
      ▼
Hybrid Retrieval Layer
(Dense + Sparse + Re-rank + Boost)
      │
      ▼
Multi-Agent Reasoning
      │
      ▼
Fact Verification Engine
      │
      ▼
Hallucination Detector
      │
      ▼
Calibration + Risk Engine
      │
      ▼
Decision & Escalation Layer
```

---

# 3. Database Design

## 3.1 PostgreSQL Schema

### 3.1.1 Patients Table

```sql
CREATE TABLE patients (
    patient_id UUID PRIMARY KEY,
    age INT,
    gender VARCHAR(20),
    ethnicity VARCHAR(50),
    weight FLOAT,
    height FLOAT,
    pregnancy_status BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

### 3.1.2 Visits Table

```sql
CREATE TABLE visits (
    visit_id UUID PRIMARY KEY,
    patient_id UUID REFERENCES patients(patient_id),
    symptoms JSONB,
    vitals JSONB,
    labs JSONB,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

### 3.1.3 Predictions Table

```sql
CREATE TABLE predictions (
    prediction_id UUID PRIMARY KEY,
    visit_id UUID REFERENCES visits(visit_id),
    disease_distribution JSONB,
    uncertainty_score FLOAT,
    final_diagnosis TEXT,
    confidence FLOAT,
    hallucination_score FLOAT,
    escalation BOOLEAN,
    retrieval_k INT,
    sources_used JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

### 3.1.4 Retrieval Logs Table

```sql
CREATE TABLE retrieval_logs (
    id SERIAL PRIMARY KEY,
    visit_id UUID,
    chunk_id UUID,
    similarity_score FLOAT,
    boosted_score FLOAT,
    rank INT
);
```

---

# 4. Retrieval System

## 4.1 Hybrid Retrieval Components

- Dense vector search (FAISS / Pinecone)
- Sparse search (Elasticsearch BM25)
- Cross-encoder re-ranking
- Guideline priority boosting
- Patient-specific filtering

---

## 4.2 Final Ranking Formula

\[
Score =
\alpha \cdot DenseSim
+ \beta \cdot BM25
+ \gamma \cdot CrossEncoder
+ \delta \cdot GuidelineBoost
+ \epsilon \cdot PatientMatch
\]

---

## 4.3 Patient-Specific Boosting

Boost documents if:

- Age > 65 → geriatric guidelines
- Pregnant → obstetric guidelines
- CKD → renal-adjusted drug docs
- Pediatric → child guidelines

---

# 5. Multi-Agent Reasoning Layer

## Agent 1 – Differential Diagnosis Agent
Generates top-5 diagnoses.

## Agent 2 – Evidence Mapping Agent
Maps diagnoses to specific evidence chunks.

## Agent 3 – Drug Safety Agent
Checks contraindications & interactions.

## Agent 4 – Consistency Agent
Ensures no contradiction across reasoning steps.

---

# 6. Uncertainty Modeling

## 6.1 Classifier-Based Uncertainty

- BioBERT fine-tuned classifier
- Logistic regression baseline

### Predictive Entropy

\[
H(p) = - \sum p_i \log p_i
\]

---

## 6.2 LLM Self-Consistency

Generate N responses:

```
temperature = 0.7
N = 7
```

Measure diagnosis agreement % and variance.

---

## 6.3 Retrieval Uncertainty

\[
\Delta = Sim_1 - Sim_5
\]

Low delta → uncertain retrieval.

---

## 6.4 Final Uncertainty Score

\[
U = w_1 H_{classifier}
+ w_2 Var_{LLM}
+ w_3 RetrievalFlatness
\]

---

# 7. Hallucination Detection

## 7.1 Structured Claim Extraction

Force LLM output:

```
For each diagnosis:
- List supporting evidence
- Cite document source
```

---

## 7.2 NLI Verification

- Model: DeBERTa-v3-large MNLI
- Premise: Retrieved evidence
- Hypothesis: LLM claim
- Threshold: entailment probability > 0.7

---

## 7.3 Similarity Gate

If cosine similarity < 0.75 → unsupported claim.

---

## 7.4 Hallucination Score

\[
H = 1 - \frac{\text{Supported Claims}}{\text{Total Claims}}
\]

---

# 8. Confidence Calibration

## 8.1 Temperature Scaling

\[
p' = \text{softmax}(z / T)
\]

Tune T on validation set.

---

## 8.2 Metrics

- Expected Calibration Error (ECE)
- Brier Score

---

# 9. Triage Risk Model

## 9.1 Output Classes

- Emergency
- Urgent
- Routine
- Home Care

## 9.2 Features

- Age
- BP
- HR
- Oxygen
- Symptom severity
- Comorbidities

Emergency probability > threshold → override system.

---

# 10. API Design

## Authentication

JWT-based:

```
Authorization: Bearer <token>
```

---

## 10.1 POST /infer/full

Runs full pipeline.

### Request

```json
{
  "visit_id": "uuid"
}
```

### Response

```json
{
  "differential": [...],
  "primary_diagnosis": "Pneumonia",
  "confidence": 0.91,
  "uncertainty": 0.34,
  "hallucination_score": 0.08,
  "risk_level": "Urgent",
  "escalation": false,
  "evidence": [...]
}
```

---

## 10.2 POST /policy/update

Update adaptive retrieval thresholds.

---

## 10.3 GET /metrics

Returns:

- Average hallucination rate
- Average ECE
- Average retrieval_k
- Escalation rate

---

# 11. LLM Stack (Resolved)

## Selected Model: `microsoft/Llama3-Med-8B-Instruct`

Chosen because:
- Medical domain fine-tuning (best fit for clinical QA)
- 8B parameters — fits in 6 GB VRAM at 4-bit, 16 GB VRAM at fp16
- Llama 3 Community License — clear for research
- Already referenced in `configs/seeds.yaml`

---

## Serving Backends (Three modes, auto-selected)

### Mode 1: `local_4bit` — RTX 4050 (6 GB)

```python
from transformers import BitsAndBytesConfig, pipeline
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
pipe = pipeline("text-generation", model=model_name,
    model_kwargs={"quantization_config": quantization_config},
    device_map="auto", max_new_tokens=256)
```

Used for: local smoke tests, development, small-scale sanity checks.

---

### Mode 2: `kaggle_fp16` — Kaggle T4 (16 GB)

```python
pipe = pipeline("text-generation", model=model_name,
    torch_dtype=torch.float16, device_map="auto", max_new_tokens=512)
```

Used for: full MedQA benchmark runs (200 questions), ablation sweeps. **Free 30h GPU/week.**

---

### Mode 3: `hf_inference_api` — HuggingFace Inference API

```python
from huggingface_hub import InferenceClient
client = InferenceClient(model="microsoft/Llama3-Med-8B-Instruct", token=HF_TOKEN)
response = client.text_generation(prompt, max_new_tokens=256)
```

Used for: RAGAS judge calls. Zero local VRAM. Free tier: ~1000 req/day — sufficient for 200-sample eval.

---

## Hardware Auto-Config (`detect_hardware()` in `src/config.py`)

| Detected VRAM | Auto-selected mode |
|---|---|
| ≥ 20 GB (Kaggle T4/A100) | `kaggle_fp16` |
| 10–19 GB | `local_4bit` (fp16-capable but conservative) |
| < 10 GB (RTX 4050) | `local_4bit` + RAGAS via `hf_inference_api` |
| No GPU | `hf_inference_api` for all calls |

---

# 12. Reinforcement Learning (Optional Advanced)

## State

```
[entropy, LLM_variance, retrieval_delta, age, risk_score]
```

## Action

```
k ∈ {3,5,7,10}
source_mix ∈ {guideline, mixed}
ask_followup ∈ {0,1}
```

## Reward

```
+1 correct diagnosis
-1 hallucination
-0.05*k
-2 unsafe decision
```

Train with PPO.

---

# 13. Experimental Design

## Compare:

1. Static RAG
2. Hybrid RAG
3. Adaptive RAG (no hallucination layer)
4. Full System

---

## Metrics

### Clinical

- Top-1 Accuracy
- Top-3 Accuracy

### Safety

- Hallucination Rate
- Escalation Precision

### Calibration

- ECE
- Brier Score

### Efficiency

- Avg retrieval_k
- Latency
- Cost per query

---

# 14. Deployment Architecture

## Stack

| Component | Technology | Notes |
|---|---|---|
| Backend | FastAPI + Uvicorn | Already implemented |
| Vector DB | **ChromaDB** | Replaces FAISS — persistent, no server needed |
| Search | BM25 (rank-bm25) | No Elasticsearch server required |
| LLM (local dev) | HuggingFace `pipeline` 4-bit NF4 | RTX 4050 compatible |
| LLM (paper eval) | Kaggle T4 notebook fp16 | Free 30h/week |
| LLM (RAGAS judge) | HF Inference API | Free tier sufficient |
| Query logs | **SQLite** | Lightweight — no PostgreSQL setup |
| Experiment tracking | MLflow / W&B | Already in requirements |
| Containerization | Docker | For local demo and reproducibility |

---

## Actual Hardware (Resolved)

| Environment | Hardware | Purpose |
|---|---|---|
| Development | RTX 4050 6 GB (local) | Code iteration, unit tests, smoke tests |
| Paper experiments | Kaggle T4 16 GB (free) | Full MedQA eval, ablation sweeps |
| RAGAS judging | HF Inference API (managed) | Zero VRAM, free tier |
| Fallback | Google Colab Pro A100 | If Kaggle quota runs out |

> **Note**: PostgreSQL, FAISS, and Elasticsearch are **not required** for the research prototype. SQLite + ChromaDB + BM25 is the correct lean research stack.

---

# 15. Security & Compliance

- AES-256 encryption at rest
- TLS 1.3 in transit
- Role-based access control
- Full audit logging
- De-identification support

---

# 16. Research Contributions

1. Confidence-aware adaptive retrieval
2. Hybrid retrieval with guideline prioritization
3. Clinical hallucination suppression layer
4. Confidence calibration in medical RAG
5. Risk-aware triage integration

---

# 17. Deliverables

- Full working system
- Experimental evaluation
- Ablation study
- Reproducible codebase
- Research paper draft
- Architecture documentation

---

# End of Document

