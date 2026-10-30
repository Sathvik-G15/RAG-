"""Evaluation script for the CAAR-CDSS Risk-Aware Triage System.

Evaluates triage safety, emergency detection sensitivity (recall),
confidence-based soft escalation, and decision tiering across clinical cases.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.confidence.pipeline import Pipeline
from src.models import (
    RiskLevel,
    TriageDecision,
    Visit,
    Vitals,
)


@dataclass
class TriageTestCase:
    case_id: str
    description: str
    query: str
    expected_tier: str  # "emergency", "urgent", "routine", "home_care"
    expected_decision: str  # "escalate", "ask_followup", "answer"
    vitals: Vitals | None = None


# A curated battery of representative clinical triage vignettes
CLINICAL_TRIAGE_BATTERY: list[TriageTestCase] = [
    # --- Category 1: Clinical Emergencies (Rule-Based Hard Triggers) ---
    TriageTestCase(
        case_id="EMERG_01",
        description="Geriatric ACS presentation with diaphoresis",
        query="62-year-old male smoker with sudden crushing retrosternal chest pain, cold sweating, and shortness of breath.",
        expected_tier="emergency",
        expected_decision="escalate",
    ),
    TriageTestCase(
        case_id="EMERG_02",
        description="Acute stroke with hemiparesis and dysarthria",
        query="68-year-old female presents with sudden weakness on right side of body and slurred speech 45 minutes ago.",
        expected_tier="emergency",
        expected_decision="escalate",
    ),
    TriageTestCase(
        case_id="EMERG_03",
        description="Severe anaphylaxis with circulatory collapse",
        query="28-year-old female with widespread urticaria, severe lip swelling, and hypotension after bee sting.",
        expected_tier="emergency",
        expected_decision="escalate",
    ),
    TriageTestCase(
        case_id="EMERG_04",
        description="Suspected bacterial sepsis",
        query="75-year-old nursing home resident with rigors, lethargy, fever 39.5C, and suspected sepsis.",
        expected_tier="emergency",
        expected_decision="escalate",
    ),
    TriageTestCase(
        case_id="EMERG_05",
        description="Critically low oxygen saturation (Vitals Hypoxemia)",
        query="70-year-old with productive cough and fatigue.",
        vitals=Vitals(oxygen_saturation=84.0, heart_rate=98.0, systolic_bp=120.0),
        expected_tier="emergency",
        expected_decision="escalate",
    ),
    TriageTestCase(
        case_id="EMERG_06",
        description="Cardiogenic / Septic shock (Vitals Hypotension)",
        query="55-year-old with general malaise and weakness.",
        vitals=Vitals(systolic_bp=82.0, diastolic_bp=50.0, heart_rate=110.0),
        expected_tier="emergency",
        expected_decision="escalate",
    ),
    TriageTestCase(
        case_id="EMERG_07",
        description="Extreme tachyarrhythmia (Vitals HR > 150)",
        query="42-year-old with sudden rapid pounding in chest and lightheadedness.",
        vitals=Vitals(heart_rate=165.0, systolic_bp=105.0),
        expected_tier="emergency",
        expected_decision="escalate",
    ),
    TriageTestCase(
        case_id="EMERG_08",
        description="Severe acute respiratory distress (Vitals RR > 30)",
        query="35-year-old with severe asthma exacerbation.",
        vitals=Vitals(respiratory_rate=36.0, oxygen_saturation=91.0),
        expected_tier="emergency",
        expected_decision="escalate",
    ),

    # --- Category 2: Low-Confidence / Ambiguous Cases (Soft Escalation) ---
    TriageTestCase(
        case_id="URGENT_01",
        description="Atypical complex out-of-scope presentation",
        query="Patient with sudden severe abdominal pain after starting an experimental immunosuppressant.",
        expected_tier="urgent",
        expected_decision="escalate",
    ),
    TriageTestCase(
        case_id="URGENT_02",
        description="Vague undifferentiated constitutional symptoms",
        query="Young adult with 3-month fluctuating night sweats, migratory joint pains, and recurring hair loss.",
        expected_tier="urgent",
        expected_decision="escalate",
    ),

    # --- Category 3: Ambulatory & Well-Characterized Cases ---
    TriageTestCase(
        case_id="AMB_01",
        description="Classic uncomplicated lower UTI (NICE guideline)",
        query="45-year-old woman with dysuria and urinary frequency without flank pain or fever.",
        vitals=Vitals(systolic_bp=118.0, heart_rate=72.0, oxygen_saturation=99.0, respiratory_rate=16.0),
        expected_tier="home_care",
        expected_decision="answer",
    ),
    TriageTestCase(
        case_id="AMB_02",
        description="Classic primary hypothyroidism presentation",
        query="32-year-old woman with fatigue, weight gain, cold intolerance, and TSH of 8.5.",
        vitals=Vitals(systolic_bp=122.0, heart_rate=68.0, oxygen_saturation=98.0, respiratory_rate=14.0),
        expected_tier="home_care",
        expected_decision="answer",
    ),
    TriageTestCase(
        case_id="ROUTINE_01",
        description="Pyelonephritis in diabetic outpatient (requires followup / clinical evaluation)",
        query="50-year-old diabetic female with fever, unilateral flank pain, and dysuria.",
        vitals=Vitals(systolic_bp=125.0, heart_rate=88.0, oxygen_saturation=97.0, respiratory_rate=18.0),
        expected_tier="routine",
        expected_decision="ask_followup",
    ),
]


def run_triage_evaluation(output_path: Path | None = None) -> dict:
    pipeline = Pipeline.from_seed()
    
    results = []
    emergency_expected_count = 0
    emergency_caught_count = 0
    escalation_expected_count = 0
    escalation_caught_count = 0
    
    print("\n" + "=" * 80)
    print("           CAAR-CDSS CLINICAL TRIAGE SYSTEM COMPREHENSIVE EVALUATION")
    print("=" * 80)
    print(f"{'Case ID':<10} | {'Expected Tier':<12} | {'Assigned Tier':<12} | {'Decision':<14} | {'Conf':<6} | {'Status'}")
    print("-" * 80)

    for case in CLINICAL_TRIAGE_BATTERY:
        visit = Visit(vitals=case.vitals) if case.vitals else None
        _, resp = pipeline.analyze(case.query, visit=visit)
        
        assigned_tier = resp.risk_level.value
        assigned_decision = resp.decision.value
        conf = resp.confidence
        
        is_emergency_expected = (case.expected_tier == "emergency")
        if is_emergency_expected:
            emergency_expected_count += 1
            if assigned_tier == "emergency":
                emergency_caught_count += 1
        
        is_escalation_expected = (case.expected_decision == "escalate")
        if is_escalation_expected:
            escalation_expected_count += 1
            if assigned_decision == "escalate":
                escalation_caught_count += 1

        tier_match = (assigned_tier == case.expected_tier)
        decision_match = (assigned_decision == case.expected_decision)
        status = "PASS" if (decision_match or (is_emergency_expected and assigned_decision == "escalate")) else "FLAG"

        print(f"{case.case_id:<10} | {case.expected_tier:<12} | {assigned_tier:<12} | {assigned_decision:<14} | {conf:>5.2f} | {status}")
        
        results.append({
            "case_id": case.case_id,
            "description": case.description,
            "query": case.query,
            "expected_tier": case.expected_tier,
            "assigned_tier": assigned_tier,
            "expected_decision": case.expected_decision,
            "assigned_decision": assigned_decision,
            "confidence": conf,
            "escalated_reason": resp.escalated_reason,
            "primary_diagnosis": resp.primary_diagnosis,
            "passed": status == "PASS",
        })

    emergency_recall = (emergency_caught_count / emergency_expected_count) * 100 if emergency_expected_count else 100.0
    escalation_recall = (escalation_caught_count / escalation_expected_count) * 100 if escalation_expected_count else 100.0

    print("-" * 80)
    print(f"Total Test Cases Evaluated : {len(CLINICAL_TRIAGE_BATTERY)}")
    print(f"Emergency Detection Recall  : {emergency_recall:.1f}% ({emergency_caught_count}/{emergency_expected_count})")
    print(f"Escalation Safety Recall    : {escalation_recall:.1f}% ({escalation_caught_count}/{escalation_expected_count})")
    print("=" * 80 + "\n")

    report = {
        "total_cases": len(CLINICAL_TRIAGE_BATTERY),
        "emergency_recall_pct": emergency_recall,
        "escalation_recall_pct": escalation_recall,
        "emergency_cases_caught": f"{emergency_caught_count}/{emergency_expected_count}",
        "results": results,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved evaluation report to {output_path}")

    return report


if __name__ == "__main__":
    out_file = Path("experiments/results/triage_evaluation_report.json")
    run_triage_evaluation(out_file)
