I'm going to act as your **research supervisor**. The goal is not just to finish the project, but to produce something that can realistically become an **IEEE conference paper**.

Assumptions (updated to reflect actual state as of Sep 2026):

* **Remaining Duration:** 8 weeks (core pipeline is ✅ already built)
* **Team:** 2 members
* **Working Time:** 20–25 hours/week/team member
* **Goal:** Real corpus + LLM + benchmarks + paper
* **Compute:** RTX 4050 local + Kaggle T4 (free) + HF Inference API (free RAGAS judging)

---

## ✅ Already Complete (Weeks 1–8 equivalent)

The following were built before this plan was written:

| Component | Status | File |
|---|---|---|
| Adaptive Evidence Budgeting (AEB) loop | ✅ Done | `src/confidence/` |
| Hybrid retrieval (Dense + BM25 + RRF) | ✅ Done | `src/retrieval/hybrid.py` |
| NLI hallucination verifier (DeBERTa) | ✅ Done | `src/verification/verifier.py` |
| Confidence calibration + triage | ✅ Done | `src/confidence/estimation.py` |
| FastAPI backend (core endpoints) | ✅ Done | `src/api/app.py` |
| MockReasoner (deterministic, no GPU) | ✅ Done | `src/agents/reasoner.py` |
| LLMReasoner (fp16, basic) | ✅ Done | `src/agents/reasoner.py` |

---

## 🔴 Remaining Work (8 weeks)

```text
Week 1   Sprint 1: epfl-llm corpus + ChromaDB + BGE-M3 + OpenSourceLLMReasoner +
                   HFInferenceReasoner + RAGAS judge + Kaggle notebook template
Week 2   Sprint 2: Missing API endpoints (upload, query-log, eval-run, review, corpus-stats)
Week 3   Sprint 3: MedQA + PubMedQA loaders + RAGAS pipeline + Abstention-rate curve
Week 4   Sprint 4: Frontend confidence badge  |  Sprint 5: Drug Safety Agent (optional)
Week 5   Kaggle T4: Run 3 baseline experiments + 2 ablation studies (200-question MedQA)
Week 6   Write Methods + Results sections
Week 7   Write Discussion + Introduction + Abstract
Week 8   IEEE formatting + GitHub cleanup + demo video + conference selection
```

---

# Original 16-Week Plan (Reference)
*The sections below document the original full plan. Weeks 1–8 are already done.*

---

# Overall Timeline (Original)

```text
Month 1 → Research + Knowledge Base + Retrieval

Month 2 → Adaptive RAG + Novel Contribution

Month 3 → Full System Development

Month 4 → Experiments + Paper + Demo
```

---

# WEEK 1 — Literature Survey & Research Gap

## Goal

Understand existing work and **finalize your novelty**.

This week is the MOST IMPORTANT.

Most students immediately start coding.

Don't.

Spend one week understanding research.

---

## Read Papers

Read around 20 papers.

Split work.

Member 1

10 papers

Member 2

10 papers

Focus on

* Medical RAG
* Clinical Decision Support
* AI Triage
* Hallucination Detection
* Adaptive Retrieval
* Confidence Estimation

---

## Create Literature Review Spreadsheet

Columns

| Paper | Year | Dataset | Model | Contribution | Limitation |
| ----- | ---- | ------- | ----- | ------------ | ---------- |

Example

| MedRAG | 2024 | PubMedQA | Llama | Retrieval | Static Top-k |

By end of week

20–25 papers summarized.

---

## Deliverables

✓ Literature Review Spreadsheet

✓ Research Gap

✓ Final Project Title

✓ Initial Architecture

---

# WEEK 2 — Dataset Collection

Goal

Collect all data.

---

## Download

### Medical Guidelines

WHO

CDC

NICE

NIH

Store

```
datasets/

guidelines/

who/

cdc/

nice/

nih/
```

---

### Medical QA

Download

PubMedQA

BioASQ

MedQA

MedMCQA

---

### Clinical Dataset

Apply for

MIMIC-IV

MIMIC-IV ED

While approval takes time,

continue using

PubMedQA

BioASQ

---

## Clean PDFs

Remove

Headers

Footers

References

Store cleaned text.

---

## Deliverables

✓ Dataset Folder

✓ Clean Documents

✓ Guidelines Repository

---

# WEEK 3 — Knowledge Base Construction

Goal

Build searchable database.

---

Install

```text
Python

LangChain

LlamaIndex

Qdrant

Sentence Transformers

FastAPI
```

---

Chunk documents.

Example

```
WHO Guideline

↓

Paragraphs

↓

512 Tokens

↓

Overlap 100
```

Generate embeddings.

Store

```
Embedding

Text

Metadata

Source
```

Metadata

```
WHO

Cardiology

2025

Guideline

Trust Score
```

---

Deliverables

✓ Vector Database

✓ Metadata

✓ Search Working

---

# WEEK 4 — Baseline RAG

Goal

Build simplest RAG.

Architecture

```
Question

↓

Embedding

↓

Vector Search

↓

LLM

↓

Answer
```

Don't optimize.

Just make it work.

---

Experiments

Test

10 questions

Measure

Latency

Precision

Correctness

---

Deliverables

✓ Working Medical Chatbot

✓ Baseline Results

---

# WEEK 5 — Hybrid Retrieval

Current

Vector only.

Upgrade.

Architecture

```
Query

↓

Vector Search

+

BM25

+

Metadata Filter

↓

Merge

↓

Top Documents
```

Implement

Dense Retrieval

BM25

Hybrid Ranking

---

Experiments

Compare

Vector

vs

Hybrid

---

Deliverables

✓ Hybrid Search

✓ Retrieval Comparison

---

# WEEK 6 — Adaptive Retrieval (Novelty Begins)

This is your first publication contribution.

Instead of

```
Top 5
```

Always

Use

```
Simple Case

↓

Top 3

Medium

↓

Top 7

Complex

↓

Top 15
```

How?

Create

Complexity Estimator

Inputs

* Symptom Count

* Rare Disease

* Ambiguity

* Confidence

Output

Simple

Medium

Complex

---

Experiments

Measure

Accuracy

Latency

Tokens Used

---

Deliverables

✓ Adaptive Retrieval

✓ First Novel Component

---

# WEEK 7 — Context Building

Instead of

```
Chest Pain
```

Search

```
55-year-old diabetic male

Chest Pain

High BP

Family History
```

Implement

Patient Context Builder

Use

Age

Gender

Vitals

History

Medication

---

Deliverables

✓ Personalized Retrieval

---

# WEEK 8 — Explainability Layer

Output

Instead of

```
Diagnosis

Pneumonia
```

Show

```
Possible Condition

Pneumonia

Confidence

91%

Evidence

WHO

CDC

PubMed
```

Also implement

Evidence Highlighting

Source Links

---

Deliverables

✓ Explainable Reports

---

# WEEK 9 — Evidence Verification

One of your biggest research contributions.

Pipeline

```
LLM

↓

Split Claims

↓

Evidence Matching

↓

Unsupported?

↓

Remove
```

Example

LLM says

```
WHO recommends Drug X
```

Verifier

↓

Not found

↓

Delete.

---

Deliverables

✓ Verification Module

---

# WEEK 10 — Confidence Estimation

Now calculate

Confidence

Example

```
Retriever

0.94

Evidence

0.91

Agreement

0.88

↓

Confidence

91%
```

Thresholds

```
>90%

Safe

70-90%

Moderate

<70%

Doctor Consultation
```

---

Deliverables

✓ Confidence Module

---

# WEEK 11 — Backend Development

FastAPI

Endpoints

```
/login

/analyze

/upload

/feedback

/retrieve

/admin
```

Database

PostgreSQL

Vector DB

Authentication

---

Deliverables

✓ Backend

---

# WEEK 12 — Frontend

Patient Portal

Doctor Dashboard

Admin Dashboard

Use

React

Tailwind

Charts

---

Deliverables

✓ Complete UI

---

# WEEK 13 — Integration

Everything together.

Pipeline

```
Frontend

↓

Backend

↓

Adaptive Retrieval

↓

LLM

↓

Verification

↓

Confidence

↓

Explainable Report
```

---

Testing

50 Queries

Fix bugs.

---

Deliverables

✓ Working System

---

# WEEK 14 — Research Experiments

This week is ONLY experiments.

Run

Vanilla RAG

Hybrid RAG

Adaptive RAG

Compare

Precision

Recall

Hallucination

Latency

Tokens

Confidence

---

Also perform

Ablation

Remove

Adaptive Retrieval

↓

Measure

Remove

Verification

↓

Measure

Remove

Confidence

↓

Measure

This becomes the Results section of your paper.

---

# WEEK 15 — Paper Writing

Split work.

## Member 1

Write

Introduction

Literature Review

Methodology

Experiments

Discussion

---

## Member 2

Architecture

Implementation

Datasets

Evaluation

Figures

References

---

Prepare

Architecture Diagram

Flowchart

Tables

Graphs

---

Deliverables

✓ Complete Paper Draft

---

# WEEK 16 — Final Week

Tasks

Grammar Check

IEEE Formatting

GitHub Cleanup

README

Demo Video

Presentation

Poster

Conference Selection

---

Deliverables

✅ IEEE Paper

✅ GitHub Repository

✅ Final Demo

✅ PPT

✅ Poster

---

# Weekly Meeting Checklist (Every Sunday)

Every week, ask these questions:

### Research

* Did we read the planned papers?
* Did we identify any new related work?

### Development

* What modules are complete?
* What is blocked?

### Experiments

* What metrics were collected?
* Are results reproducible?

### Documentation

* Is the GitHub repository updated?
* Is the report/paper updated with this week's work?

---

# Team Task Allocation (Remaining 8 Weeks)

| Week | Member 1 (Research & AI) | Member 2 (Software & Integration) |
| ---- | ----------------------------------- | ---------------------------------------- |
| 1 | `OpenSourceLLMReasoner` (4-bit NF4) + `HFInferenceReasoner` | ChromaDB backend + BGE-M3 embedder + epfl-llm loader |
| 1 | RAGAS eval script + HF API judge config | Kaggle notebook template + `detect_hardware()` |
| 2 | MedQA + PubMedQA benchmark loaders | Missing API endpoints (upload, query-log, eval) |
| 3 | Abstention-rate curve sweep | RAGAS pipeline wired into `run_all` |
| 4 | Drug Safety Agent (optional) | Frontend confidence badge + corpus stats view |
| 5 | Kaggle: 3 baseline experiments | Kaggle: 2 ablation studies |
| 6 | Write Methods section | Results tables + Figures 1 & 2 |
| 7 | Write Discussion + Introduction | Write Abstract + format references |
| 8 | IEEE formatting + final review | GitHub cleanup + demo video + presentation |


---

# One Important Change I'd Make Before You Start

After thinking through the whole project, I would **slightly refine the research direction**.

Instead of making **Adaptive Retrieval** your primary novelty, make **Adaptive Evidence Budgeting** the centerpiece.

### Why?

Most existing systems decide:

> **"How many documents should I retrieve?"**

Your system would decide:

> **"Do I have enough evidence to answer safely?"**

Example:

```text
Patient Query
      │
      ▼
Retrieve 3 Documents
      │
      ▼
Confidence = 58%
      │
Need More Evidence?
      │
     YES
      ▼
Retrieve 3 More Documents
      │
      ▼
Confidence = 84%
      │
Need More Evidence?
      │
     YES
      ▼
Retrieve 2 More Documents
      │
      ▼
Confidence = 93%
      │
STOP
      ▼
Generate Response
```

This is **adaptive evidence budgeting**, and I believe it is a stronger, cleaner, and more publishable research contribution than simply changing Top-K based on query complexity. It directly answers the clinically important question:

> **"When does the AI have enough evidence to make a safe recommendation?"**

That single idea can become the title, the algorithm, the experiments, and the core contribution of your IEEE paper. It also remains feasible for a two-person team within four months because it builds on the same architecture you've already planned, rather than requiring entirely new components.
