> **Confidence-Aware Adaptive Retrieval Clinical Decision Support System (CAAR-CDSS)**

This is written as an engineering + research implementation blueprint. You can directly use this for thesis documentation or project planning.

---

# 1. System Overview

## 1.1 Objective

Design and implement a **safe clinical decision support system** that:

- Dynamically adapts retrieval strategy based on uncertainty
- Estimates diagnostic confidence
- Detects hallucinations
- Escalates to clinician when uncertain
- Provides evidence-grounded explanations

---

## 1.2 High-Level Architecture

```
Patient Input (Symptoms + Profile)
            │
            ▼
Preprocessing & Feature Builder
            │
            ▼
Initial Disease Probability Estimator
            │
            ▼
Uncertainty Estimator
            │
            ▼
Adaptive Retrieval Policy
      ┌───────────────┬───────────────┐
      ▼               ▼               ▼
Guideline DB     Literature DB   Drug DB
      └───────────────┴───────────────┘
            ▼
Evidence Aggregator
            │
            ▼
LLM Reasoning Module
            │
            ▼
Fact Verification Module
            │
            ▼
Confidence Calibration
            │
            ▼
Decision Module
     ├── Answer
     ├── Ask Follow-up
     └── Escalate to Doctor
```

---

# 2. Module-Level Design

---

# 2.1 Patient Input Module

## Inputs

- Symptoms (text)
- Duration
- Vitals
- Age
- Gender
- Comorbidities
- Medications
- Allergies

## Output

Structured patient object:

```json
{
  "symptoms": ["fever", "cough"],
  "duration": {"fever": "3 days"},
  "age": 65,
  "comorbidities": ["diabetes"],
  "medications": ["metformin"],
  "allergies": ["penicillin"]
}
```

## Implementation

- Use clinical NER model (e.g., BioClinicalBERT)
- Map symptoms to SNOMED codes (optional but recommended)
- Normalize units

---

# 2.2 Initial Disease Probability Estimator

Purpose:
Provide a rough disease probability distribution before heavy retrieval.

## Options

- Fine-tuned classifier (BioBERT)
- Gradient Boosting on structured features
- LLM zero-shot with calibrated logits

## Output

```
{
  "pneumonia": 0.55,
  "tuberculosis": 0.12,
  "covid19": 0.18,
  "other": 0.15
}
```

---

# 2.3 Uncertainty Estimation Module

This is the core research contribution.

We compute:

### 1️⃣ Predictive Entropy

\[
H(p) = - \sum p_i \log p_i
\]

### 2️⃣ Top-2 Margin

\[
p_1 - p_2
\]

Low margin → high uncertainty

### 3️⃣ Monte Carlo Dropout (if neural classifier)

Run classifier N times.

Variance = epistemic uncertainty.

### 4️⃣ LLM Self-Consistency Variance

Generate multiple reasoning chains.
Measure answer variance.

---

## Output

```json
{
  "entropy": 1.21,
  "margin": 0.08,
  "variance": 0.17,
  "uncertainty_score": 0.72
}
```

Normalize to 0–1.

---

# 2.4 Adaptive Retrieval Policy

Core decision logic.

## Input

- Patient profile
- Disease distribution
- Uncertainty score

## Policy Logic (Initial Rule-Based Version)

Example:

| Uncertainty | Retrieval Strategy |
|------------|-------------------|
| Low (<0.3) | Top-3 guidelines |
| Medium (0.3–0.6) | Top-7 guidelines + 3 papers |
| High (>0.6) | Top-10 + rare disease filter |
| Very High (>0.8) | Ask follow-up or escalate |

---

## Later: Learnable Policy

Train policy:

State:
```
[query_embedding, uncertainty, top_disease_prob]
```

Action:
```
k ∈ {3,5,10}
source ∈ {guideline, literature, mixed}
ask_followup ∈ {yes/no}
```

Reward:
- + accuracy
- − hallucination
- − cost

Use RL (PPO or Q-learning).

---

# 2.5 Document Retrieval System

## 2.5.1 Databases

1. WHO / CDC / NICE Guidelines
2. PubMed abstracts
3. Drug interaction database
4. Hospital protocols

---

## 2.5.2 Indexing

- Chunk size: 300–500 tokens
- Overlap: 50 tokens
- Embedding model: Bio-embedding model
- Vector DB: FAISS / Pinecone / Weaviate

---

## 2.5.3 Retrieval Scoring

Final score:

\[
Score = \alpha \cdot Similarity
+ \beta \cdot GuidelinePriority
+ \gamma \cdot PatientMatch
\]

Guidelines get boost weight.

---

# 2.6 Evidence Aggregator

Combine retrieved chunks.

Remove duplicates.

Rank by:
- Source priority
- Relevance
- Recency

Output:

Top N evidence blocks passed to LLM.

---

# 2.7 LLM Reasoning Module

Prompt template:

```
You are a clinical decision support assistant.

Patient Information:
{patient_profile}

Retrieved Evidence:
{evidence}

Instructions:
1. Provide differential diagnosis.
2. Support each diagnosis with evidence.
3. Provide confidence score (0–100%).
4. List alternative diagnoses.
5. If insufficient evidence, say so.
```

---

## Output Format (Structured JSON)

```json
{
  "primary_diagnosis": "Pneumonia",
  "confidence": 0.87,
  "evidence": [
    {"symptom": "fever", "source": "WHO guideline p18"}
  ],
  "alternatives": [
    {"disease": "TB", "prob": 0.12}
  ]
}
```

---

# 2.8 Fact Verification Module

For each claim:

- Check if claim sentence embedding matches retrieved evidence.
- Use NLI model to verify entailment.

If unsupported:

Flag hallucination.

---

# 2.9 Confidence Calibration

Raw LLM confidence is often overconfident.

Apply:

- Temperature scaling
- Isotonic regression
- Platt scaling

Train on validation set.

Measure:

- Brier Score
- Expected Calibration Error (ECE)

---

# 2.10 Decision Module

Logic:

If:

- Hallucination detected → regenerate or escalate
- Confidence < threshold → ask follow-up
- Risk score high → recommend emergency
- Confidence very low → escalate to doctor

---

# 3. Safety Mechanisms

## 3.1 Hard Constraints

Never:

- Prescribe controlled drugs
- Replace physician
- Provide definitive diagnosis without disclaimer

## 3.2 Emergency Detection

Rule-based high-risk triggers:

- Chest pain + sweating + age > 60
- BP < 90 systolic
- Oxygen < 90%

Immediate escalation.

---

# 4. Evaluation Plan

---

## 4.1 Datasets

- MIMIC-IV (if available)
- MedQA
- Synthetic structured cases
- Custom annotated dataset

---

## 4.2 Baselines

- Static Top-5 RAG
- Plain LLM
- Large-k RAG
- Med-specific LLM

---

## 4.3 Metrics

### Clinical

- Diagnostic Accuracy
- Top-3 Accuracy

### Safety

- Hallucination Rate
- Escalation Precision
- False Reassurance Rate

### Calibration

- ECE
- Brier Score

### Efficiency

- Avg retrieval count
- Latency
- Cost per query

---

# 5. Implementation Stack

## Backend

- Python 3.11
- FastAPI + Uvicorn
- PyTorch 2.3
- HuggingFace `transformers` + `accelerate`
- `bitsandbytes` (4-bit NF4 quantization)

## Knowledge Base / Retrieval

- **ChromaDB** — persistent vector store (replaces in-memory)
- **BGE-M3** (FlagEmbedding) — primary embedding model
- **BM25** (rank-bm25) — sparse retrieval
- **DeBERTa-v3-MNLI** — NLI claim verifier

## LLM Stack (Resolved)

| Environment | Model | Mode |
|---|---|---|
| Local dev (RTX 4050 / 6 GB) | `Llama3-Med-8B-Instruct` | 4-bit NF4 via `bitsandbytes` |
| Kaggle T4 eval runs | `Llama3-Med-8B-Instruct` | fp16 in-process `pipeline()` |
| RAGAS judging (zero VRAM) | `Llama3-Med-8B-Instruct` | HuggingFace Inference API (free tier) |

## Database

- **SQLite** — query logs (lightweight, no PostgreSQL setup needed for research)
- **ChromaDB** — vector store

## Compute Strategy

- **Local (RTX 4050 6 GB)** → Development, unit tests, MockReasoner
- **Kaggle Notebooks (T4 × 1–2, 30h free/week)** → Full MedQA benchmarks, ablations
- **HuggingFace Inference API** → RAGAS judge calls (~1000 req/day free tier)
- **Google Colab Pro** *(optional fallback)* → A100 if Kaggle quota runs out

## Deployment

- Docker (local + demo)
- Kaggle Notebooks (paper experiments)
- Secure server if clinical validation added later

---

# 6. Development Phases

---

## Phase 1 – Core Pipeline ✅ COMPLETE

✅ Adaptive Evidence Budgeting (AEB) loop
✅ Hybrid retrieval (dense + BM25 + RRF)
✅ NLI hallucination verifier (DeBERTa)
✅ Confidence calibration + triage
✅ FastAPI backend
✅ MockReasoner (deterministic, no GPU needed)

---

## Phase 2 – Real Corpus & LLM Stack 🔴 IN PROGRESS

- [ ] Load `epfl-llm/guidelines` → ChromaDB (Task 1.1–1.2)
- [ ] Wire BGE-M3 embedder (Task 1.3)
- [ ] `OpenSourceLLMReasoner` with 4-bit NF4 — local dev (Task 1.7)
- [ ] `HFInferenceReasoner` — RAGAS judge via HF API (Task 1.7–1.8)
- [ ] Hardware auto-detection + `seeds.yaml` LLM config (Task 1.9–1.10)
- [ ] Kaggle notebook template for T4 eval runs (Task 1.10.1)

---

## Phase 3 – Evaluation Infrastructure 🔴 PENDING

- [ ] MedQA + PubMedQA benchmark loaders
- [ ] RAGAS pipeline wired into `run_all`
- [ ] Abstention-rate curve (threshold sweep)
- [ ] Missing API endpoints (Sprint 2)

---

## Phase 4 – Paper Experiments 🟡 PENDING

- [ ] 3 baseline experiments on Kaggle T4
- [ ] 2 ablation studies (threshold sweep + embedding comparison)
- [ ] Write Methods → Results → Discussion → Introduction → Abstract

---

## Phase 5 – Polish 🟢 OPTIONAL

- [ ] Frontend confidence badge
- [ ] Drug Safety Agent
- [ ] RL adaptive retrieval policy  

---

# 7. Expected Research Contributions

1. Formal uncertainty-aware retrieval framework
2. Adaptive retrieval policy
3. Clinical hallucination suppression
4. Confidence-calibrated decision system

---

# 8. Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| Overfitting | Cross-hospital validation |
| LLM hallucination | Strict evidence verification |
| High cost | Adaptive model routing |
| Dataset bias | Diverse clinical cases |

---

# 9. Deliverables

- Working system
- Research paper
- Evaluation report
- Open benchmark (optional)
- Code repository
- Architecture diagram
