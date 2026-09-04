# CAAR-CDSS: Confidence-Aware Adaptive Retrieval Clinical Decision Support System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Paper: IEEE](https://img.shields.io/badge/Paper-IEEE%20Target-blue)](paper/)

A **trustworthy clinical decision support system** featuring Adaptive Evidence Budgeting (AEB), evidence-grounded retrieval, and confidence-calibrated escalation for safe medical AI assistance.

---

## 🎯 Core Contributions

1. **Adaptive Evidence Budgeting (AEB)** — Iteratively retrieves documents until calibrated confidence ≥ θ, rather than fixed top-k.
2. **Hybrid Retrieval** — Dense (biomedical embeddings) + Sparse (BM25) + Cross-encoder rerank + guideline priority boosting.
3. **Evidence Verification (NLI)** — Claim-level entailment checking with DeBERTa-v3-MNLI.
4. **Confidence Calibration** — Multi-signal fusion with temperature scaling; well-calibrated ECE.
5. **Risk-aware Triage** — Hard-coded emergency escalation + soft escalation when evidence insufficient.

---

## 📁 Project Structure

```
Capstone/
├── src/
│   ├── data/           # Data loading & preprocessing
│   ├── kb/             # Knowledge base construction (chunking, embedding, storage)
│   ├── retrieval/      # Hybrid retrieval (dense + sparse + rerank)
│   ├── agents/         # Multi-agent reasoning layer
│   ├── verification/   # NLI-based claim verification
│   ├── confidence/     # Confidence calibration & AEB loop
│   ├── api/            # FastAPI backend
│   └── experiments/    # Experiment runners & evaluation
├── notebooks/          # Kaggle T4 eval notebooks (paper experiments)
├── frontend/           # React + Tailwind dashboards
├── data/
│   ├── guidelines/     # epfl-llm/guidelines corpus (ChromaDB)
│   └── raw/            # Raw datasets (PubMedQA, MedQA, etc.)
├── experiments/        # Results, ablations, figures
├── paper/              # IEEE paper draft
├── configs/            # Reproducibility configs + seeds
├── tests/              # Unit & integration tests
└── Docs/               # Design documents
```

---

## 🚀 Quick Start

```bash
# 1. Clone & setup
git clone <repo-url> && cd Capstone
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Configure environment
cp configs/secrets.example.yaml configs/secrets.yaml
# Edit secrets.yaml — add HF_TOKEN for HuggingFace Inference API (free at hf.co/settings/tokens)

# 3. Auto-detect your hardware (prints recommended serving_backend)
python -c "from src.config import detect_hardware; print(detect_hardware())"

# 4. Build the knowledge base (epfl-llm/guidelines, all clinical domains)
python -m src.cli ingest --corpus epfl-llm/guidelines

# 5. Run a query — mock (no GPU) or real (local 4-bit or HF Inference API)
python -m src.cli query "55yo diabetic with chest pain" --mode mock
python -m src.cli query "55yo diabetic with chest pain" --mode real   # uses detected backend

# 6. Launch the API
uvicorn src.api.app:app --reload

# 7. Launch the frontend
cd frontend && npm install && npm run dev
```

---

## 🧪 Running Experiments

```bash
# Local: smoke test on 10 samples (MockReasoner, no GPU needed)
python -m src.experiments.run_all --benchmark medqa --n 10 --mode mock

# Local: smoke test with real LLM (RTX 4050, 4-bit NF4)
python -m src.experiments.run_all --benchmark medqa --n 10 --mode real

# RAGAS evaluation (calls HF Inference API as judge — set HF_TOKEN in secrets.yaml)
python -m src.experiments.ragas_eval --benchmark medqa --n 50

# Full paper experiments → run notebooks/kaggle_eval_template.ipynb on Kaggle T4
# (Upload repo, run notebook, download results CSV from /kaggle/working/results/)
```

All experiments log to `experiments/results/` with timestamps.

---

## 📊 Reproducibility

See `configs/seeds.yaml` for all random seeds.
See `configs/versions.txt` for locked library versions.

---

## 📖 Documentation

- [`Docs/Basic design doc.md`](Docs/Basic%20design%20doc.md) — System design overview
- [`Docs/Detailed doc.md`](Docs/Detailed%20doc.md) — Full technical specification (LLM stack §11, deployment §14)
- [`Docs/timeline.md`](Docs/timeline.md) — 8-week remaining execution plan
- [`notebooks/kaggle_eval_template.ipynb`](notebooks/) — Kaggle T4 benchmark notebook
- [`paper/`](paper/) — IEEE paper draft

## 💻 Compute Setup

| Environment | What to run there |
|---|---|
| **Local (RTX 4050 6 GB)** | Development, tests, `--mode mock`, smoke tests with `--mode real` (4-bit NF4) |
| **Kaggle Notebook (T4, free)** | Full MedQA benchmarks, ablation sweeps — use `notebooks/kaggle_eval_template.ipynb` |
| **HuggingFace Inference API** | RAGAS judge calls — set `HF_TOKEN` in `configs/secrets.yaml` |

---

## 📝 License

MIT License — see [`LICENSE`](LICENSE) for details.

---

## ⚠️ Disclaimer

This system is a **research prototype** for academic publication. It is **not a medical device** and must not be used for actual clinical decision-making. All outputs include safety disclaimers and escalation pathways to qualified clinicians.
