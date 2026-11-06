"""Run benchmark evaluations directly against the real server pipeline.

Uses the real server models:
  - Vector Store: ChromaDB (18,480 chunks) or Unified Clinical KB
  - Embedding: BAAI/bge-small-en-v1.5
  - Reasoner: m42-health/Llama3-Med42-8B (4-bit quantized) / Clinical Diagnostic Reasoner
  - Verifier: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
  - Confidence Engine: Calibrated multi-signal AEB (Adaptive Evidence Budgeting)
  - Triage Engine: 4-tier risk classification (Emergency, Urgent, Routine, Home Care)

Usage:
  python scripts/run_real_benchmarks.py --benchmark prompts
  python scripts/run_real_benchmarks.py --benchmark triage
  python scripts/run_real_benchmarks.py --benchmark medqa --limit 20
  python scripts/run_real_benchmarks.py --benchmark pubmedqa --limit 20
  python scripts/run_real_benchmarks.py --benchmark ddxplus --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure offline cache lookups and CUDA configuration
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from src.api.app import get_pipeline
from src.models import Visit, Vitals


# ─── 22 CLINICAL BENCHMARK PROMPTS ───────────────────────────────────────────
CLINICAL_BENCHMARK_PROMPTS = [
    # Category 1: ANSWER (High Confidence, Home Care)
    {
        "id": "ANS_01",
        "condition": "Primary Hypothyroidism",
        "query": "fatigue, cold intolerance, weight gain, constipation, elevated tsh",
        "age": 40, "gender": "female",
        "expected_decision": "answer", "expected_tier": "home_care",
    },
    {
        "id": "ANS_02",
        "condition": "Influenza",
        "query": "sudden fever, myalgia, headache, dry cough, fatigue, chills, influenza season",
        "age": 28, "gender": "male",
        "expected_decision": "answer", "expected_tier": "home_care",
    },
    {
        "id": "ANS_03",
        "condition": "Tension Headache",
        "query": "bilateral tension headache, dull pressure, band around head, no nausea",
        "age": 32, "gender": "female",
        "expected_decision": "answer", "expected_tier": "home_care",
    },
    {
        "id": "ANS_04",
        "condition": "GERD",
        "query": "heartburn, acid regurgitation, epigastric burning, worse after meals, lying down, gerd",
        "age": 45, "gender": "male",
        "expected_decision": "answer", "expected_tier": "home_care",
    },
    {
        "id": "ANS_05",
        "condition": "COPD Exacerbation",
        "query": "worsening dyspnea, increased sputum, purulent sputum, copd, wheeze, cough",
        "age": 65, "gender": "male",
        "expected_decision": "answer", "expected_tier": "home_care",
    },

    # Category 2: ASK_FOLLOWUP (Moderate Confidence, Routine Risk)
    {
        "id": "FLW_01",
        "condition": "Cluster Headache",
        "query": "severe unilateral orbital headache, lacrimation, nasal congestion, cluster headache",
        "age": 35, "gender": "male",
        "expected_decision": "ask_followup", "expected_tier": "routine",
    },
    {
        "id": "FLW_02",
        "condition": "Hypertension",
        "query": "high blood pressure, hypertension, bp 148/92, no headache, no chest pain",
        "age": 56, "gender": "male",
        "expected_decision": "ask_followup", "expected_tier": "routine",
        "vitals": Vitals(systolic_bp=148.0, diastolic_bp=92.0, heart_rate=76.0),
    },
    {
        "id": "FLW_03",
        "condition": "Migraine Spectrum",
        "query": "throbbing headache, mild nausea, recurrent headache, migraine",
        "age": 34, "gender": "female",
        "expected_decision": "ask_followup", "expected_tier": "routine",
    },
    {
        "id": "FLW_04",
        "condition": "HIV Initial Infection",
        "query": "fever, lymphadenopathy, pharyngitis, rash, fatigue, possible HIV exposure, acute retroviral",
        "age": 29, "gender": "male",
        "expected_decision": "ask_followup", "expected_tier": "routine",
    },
    {
        "id": "FLW_05",
        "condition": "C. difficile Infection",
        "query": "watery diarrhea, recent antibiotics, abdominal cramps, Clostridioides difficile",
        "age": 62, "gender": "female",
        "expected_decision": "ask_followup", "expected_tier": "routine",
    },

    # Category 3: ESCALATE - Emergencies (Risk: EMERGENCY)
    {
        "id": "ESC_EMERG_01",
        "condition": "Acute Coronary Syndrome",
        "query": "crushing chest pain, left arm pain, sweating, diaphoresis, shortness of breath",
        "age": 58, "gender": "male",
        "expected_decision": "escalate", "expected_tier": "emergency",
        "vitals": Vitals(systolic_bp=140.0, heart_rate=110.0, oxygen_saturation=94.0),
    },
    {
        "id": "ESC_EMERG_02",
        "condition": "Acute Ischemic Stroke",
        "query": "sudden facial droop, slurred speech, right arm weakness, acute stroke",
        "age": 67, "gender": "female",
        "expected_decision": "escalate", "expected_tier": "emergency",
    },
    {
        "id": "ESC_EMERG_03",
        "condition": "Anaphylaxis",
        "query": "acute urticaria, lip swelling, angioedema, throat tightness, peanut allergy",
        "age": 22, "gender": "female",
        "expected_decision": "escalate", "expected_tier": "emergency",
    },
    {
        "id": "ESC_EMERG_04",
        "condition": "Bacterial Meningitis / Sepsis",
        "query": "high fever 40c, severe stiff neck, photophobia, headache",
        "age": 24, "gender": "male",
        "expected_decision": "escalate", "expected_tier": "emergency",
    },
    {
        "id": "ESC_EMERG_05",
        "condition": "Pulmonary Embolism with Hypoxia",
        "query": "sudden pleuritic chest pain, severe shortness of breath, hypoxia, spo2 86",
        "age": 52, "gender": "female",
        "expected_decision": "escalate", "expected_tier": "emergency",
        "vitals": Vitals(oxygen_saturation=86.0, heart_rate=115.0),
    },

    # Category 3: ESCALATE - Urgent Presentations (Risk: URGENT)
    {
        "id": "ESC_URG_01",
        "condition": "Community-Acquired Pneumonia",
        "query": "fever, productive cough, purulent sputum, pleuritic chest pain, dyspnea, lobar infiltrate",
        "age": 68, "gender": "male",
        "expected_decision": "escalate", "expected_tier": "urgent",
    },
    {
        "id": "ESC_URG_02",
        "condition": "Acute Asthma / Bronchospasm",
        "query": "acute wheeze, shortness of breath, chest tightness, bronchospasm, inhaler use",
        "age": 31, "gender": "female",
        "expected_decision": "escalate", "expected_tier": "urgent",
    },
    {
        "id": "ESC_URG_03",
        "condition": "Atrial Fibrillation",
        "query": "palpitations, irregular pulse, shortness of breath, fatigue, atrial fibrillation",
        "age": 71, "gender": "male",
        "expected_decision": "escalate", "expected_tier": "urgent",
    },
    {
        "id": "ESC_URG_04",
        "condition": "Spontaneous Pneumothorax",
        "query": "sudden pleuritic chest pain, absent breath sounds, dyspnea, tall thin male, pneumothorax",
        "age": 21, "gender": "male",
        "expected_decision": "escalate", "expected_tier": "urgent",
    },
    {
        "id": "ESC_URG_05",
        "condition": "Acute Pericarditis",
        "query": "sharp chest pain, worse lying down, better leaning forward, friction rub, pericarditis",
        "age": 36, "gender": "male",
        "expected_decision": "escalate", "expected_tier": "urgent",
    },

    # Category 3: ESCALATE - Abstentions (Low confidence or empty evidence)
    {
        "id": "ESC_ABS_01",
        "condition": "Type 2 Diabetes (Out of Scope / Atypical)",
        "query": "polyuria, polydipsia, weight loss, random glucose 240, type 2 diabetes",
        "age": 49, "gender": "male",
        "expected_decision": "escalate", "expected_tier": "routine",
    },
    {
        "id": "ESC_ABS_02",
        "condition": "Uncomplicated UTI (Atypical)",
        "query": "dysuria, urinary frequency, urgency, suprapubic pain, no fever, uncomplicated uti",
        "age": 27, "gender": "female",
        "expected_decision": "escalate", "expected_tier": "urgent",
    },
]


def run_prompts_benchmark(pipeline: Any, limit: int | None = None) -> dict[str, Any]:
    prompts = CLINICAL_BENCHMARK_PROMPTS[:limit] if limit else CLINICAL_BENCHMARK_PROMPTS
    results = []
    
    print("\n" + "=" * 105)
    print("                     CAAR-CDSS REAL SERVER BENCHMARK EVALUATION")
    print("=" * 105)
    print(f"{'ID':<12} | {'Condition':<25} | {'Decision':<14} | {'Conf':<6} | {'Risk':<12} | {'Time':<6} | {'Status'}")
    print("-" * 105)

    n_correct_decision = 0
    emergency_caught = 0
    emergency_total = 0
    escalation_caught = 0
    escalation_total = 0

    for item in prompts:
        vitals = item.get("vitals")
        visit = Visit(vitals=vitals) if vitals else None
        
        t0 = time.perf_counter()
        aeb, resp = pipeline.analyze(item["query"], visit=visit)
        latency = time.perf_counter() - t0

        assigned_decision = resp.decision.value
        assigned_risk = resp.risk_level.value
        conf = resp.confidence
        expected_decision = item["expected_decision"]
        expected_tier = item["expected_tier"]

        if expected_tier == "emergency":
            emergency_total += 1
            if assigned_risk == "emergency":
                emergency_caught += 1

        if expected_decision == "escalate":
            escalation_total += 1
            if assigned_decision == "escalate":
                escalation_caught += 1

        # Match check
        decision_match = (assigned_decision == expected_decision)
        if decision_match:
            n_correct_decision += 1

        status = "MATCH" if decision_match else "FLAG"
        print(f"{item['id']:<12} | {item['condition'][:25]:<25} | {assigned_decision:<14} | {conf*100:>5.1f}% | {assigned_risk:<12} | {latency:>5.2f}s | {status}")

        results.append({
            "id": item["id"],
            "condition": item["condition"],
            "query": item["query"],
            "expected_decision": expected_decision,
            "assigned_decision": assigned_decision,
            "expected_tier": expected_tier,
            "assigned_tier": assigned_risk,
            "confidence": conf,
            "primary_diagnosis": resp.primary_diagnosis,
            "retrieval_k": resp.retrieval_k_used,
            "retrieval_steps": resp.retrieval_steps,
            "latency_s": round(latency, 3),
            "match": decision_match,
        })

    decision_accuracy = (n_correct_decision / len(prompts)) * 100 if prompts else 0.0
    emergency_recall = (emergency_caught / emergency_total) * 100 if emergency_total else 100.0
    escalation_recall = (escalation_caught / escalation_total) * 100 if escalation_total else 100.0
    avg_latency = sum(r["latency_s"] for r in results) / len(results) if results else 0.0

    print("-" * 105)
    print(f"Total Test Cases Evaluated : {len(prompts)}")
    print(f"Decision Tier Alignment   : {decision_accuracy:.1f}% ({n_correct_decision}/{len(prompts)})")
    print(f"Emergency Detection Recall : {emergency_recall:.1f}% ({emergency_caught}/{emergency_total})")
    print(f"Escalation Safety Recall   : {escalation_recall:.1f}% ({escalation_caught}/{escalation_total})")
    print(f"Average Pipeline Latency   : {avg_latency:.2f}s per query")
    print("=" * 105 + "\n")

    return {
        "benchmark": "clinical_prompts",
        "total_cases": len(prompts),
        "decision_alignment_pct": decision_accuracy,
        "emergency_recall_pct": emergency_recall,
        "escalation_recall_pct": escalation_recall,
        "average_latency_s": avg_latency,
        "results": results,
    }


def run_external_benchmark(pipeline: Any, benchmark_name: str, limit: int = 20) -> dict[str, Any]:
    file_path = PROJECT_ROOT / "data" / "external" / f"{benchmark_name}.jsonl"
    if not file_path.exists():
        print(f"❌ Error: Dataset {file_path} not found.")
        return {}

    queries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            # Standardize query extraction
            q_text = item.get("question") or item.get("query") or item.get("transcription") or item.get("case_description") or ""
            expected = item.get("answer") or item.get("final_diagnosis") or item.get("label") or ""
            if q_text:
                queries.append({"query": q_text[:500], "expected": expected, "raw": item})
            if len(queries) >= limit:
                break

    print(f"\n=== Evaluating on {benchmark_name.upper()} ({len(queries)} samples) ===")
    results = []
    latencies = []
    confidences = []

    for i, q in enumerate(queries, 1):
        t0 = time.perf_counter()
        aeb, resp = pipeline.analyze(q["query"])
        latency = time.perf_counter() - t0

        latencies.append(latency)
        confidences.append(resp.confidence)

        print(f"[{i:02d}/{len(queries):02d}] Dx: {resp.primary_diagnosis or 'None':<28} | Conf: {resp.confidence*100:>5.1f}% | Decision: {resp.decision.value:<14} | Risk: {resp.risk_level.value:<10} | {latency:.2f}s")

        results.append({
            "query": q["query"],
            "predicted_diagnosis": resp.primary_diagnosis,
            "confidence": resp.confidence,
            "decision": resp.decision.value,
            "risk_level": resp.risk_level.value,
            "retrieval_k": resp.retrieval_k_used,
            "latency_s": round(latency, 3),
        })

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

    print(f"\n--- {benchmark_name.upper()} Summary ---")
    print(f"Samples Evaluated : {len(queries)}")
    print(f"Average Confidence: {avg_conf*100:.1f}%")
    print(f"Average Latency   : {avg_lat:.2f}s")

    return {
        "benchmark": benchmark_name,
        "n_samples": len(queries),
        "avg_confidence": avg_conf,
        "avg_latency_s": avg_lat,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks with the real server pipeline")
    parser.add_argument("--benchmark", type=str, default="prompts",
                        choices=["prompts", "triage", "medqa", "pubmedqa", "ddxplus", "mmlumed", "mtsamples", "all"],
                        help="Benchmark dataset to run")
    parser.add_argument("--limit", type=int, default=20, help="Max queries to evaluate for external datasets")
    parser.add_argument("--output", type=str, default="experiments/results/real_server_benchmark_report.json")
    args = parser.parse_args()

    print("🔧 Initializing Real Server Pipeline (ChromaDB + DeBERTa NLI + AEB Confidence)...")
    pipeline = get_pipeline()
    print("✅ Pipeline loaded successfully!\n")

    report: dict[str, Any] = {}

    if args.benchmark in ("prompts", "all"):
        report["prompts"] = run_prompts_benchmark(pipeline, limit=args.limit)

    if args.benchmark == "triage":
        from scripts.evaluate_triage import run_triage_evaluation
        report["triage"] = run_triage_evaluation()

    if args.benchmark in ("medqa", "pubmedqa", "ddxplus", "mmlumed", "mtsamples"):
        report[args.benchmark] = run_external_benchmark(pipeline, args.benchmark, limit=args.limit)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n💾 Saved evaluation report to: {out_path}")


if __name__ == "__main__":
    main()
