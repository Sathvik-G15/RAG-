"""Severity-aware triage (replaces confidence-only triage).

Principle: ACUITY comes from vitals, red flags and the can't-miss differential.
CONFIDENCE only decides ANSWER vs ASK_FOLLOWUP vs human review. Confidence can
never lower acuity.

All numeric thresholds below are ILLUSTRATIVE defaults. A clinician must review
them against a published protocol (ESI / NEWS2 / PEWS / WHO) and cite it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

# Use the existing enums from models.py for compatibility
from ..models import RiskLevel, TriageDecision


@dataclass
class TriageConfig:
    high_confidence: float = 0.85
    low_confidence: float = 0.70
    cantmiss_emergency_prob: float = 0.20  # p >= this for an emergent dx -> EMERGENCY
    cantmiss_urgent_prob: float = 0.05     # p >= this for any can't-miss dx -> >= URGENT
    low_confidence_risk: RiskLevel = RiskLevel.ROUTINE  # acuity when only confidence is low


@dataclass
class TriageInput:
    age_years: Optional[float] = None
    sbp: Optional[float] = None
    hr: Optional[float] = None
    spo2: Optional[float] = None
    rr: Optional[float] = None
    temp_c: Optional[float] = None
    avpu: Optional[str] = None  # "A","V","P","U"
    pregnant: bool = False
    symptom_text: str = ""


@dataclass
class TriageResult:
    risk_level: RiskLevel
    decision: TriageDecision
    reasons: list = field(default_factory=list)
    followup_questions: list[str] = field(default_factory=list)


# (max_age_exclusive, hr_low, hr_high, rr_low, rr_high)
_AGE_BANDS = [
    (1, 80, 200, 10, 60),
    (5, 60, 180, 10, 40),
    (12, 50, 160, 8, 35),
    (18, 45, 150, 8, 30),
    (999, 40, 150, 8, 30),
]


def _band(age: Optional[float]):
    a = 30.0 if age is None else age
    for row in _AGE_BANDS:
        if a < row[0]:
            return row
    return _AGE_BANDS[-1]


def _sbp_low(age: Optional[float]) -> float:
    a = 30.0 if age is None else age
    if a < 1:
        return 70
    if a < 10:
        return 70 + 2 * a
    return 90


EMERGENCY_PHRASES = [
    "face droop", "facial droop", "slurred speech", "one-sided weakness", "sudden weakness",
    "worst headache", "can't breathe", "cannot breathe", "unable to breathe", "throat swelling",
    "anaphylaxis", "severe bleeding", "vomiting blood", "coughing blood", "suicidal",
    "suicide", "seizure", "unconscious", "unresponsive", "overdose", "crushing chest",
    "lip swelling", "urticaria", "angioedema",
]
URGENT_PHRASES = [
    "chest pain", "shortness of breath", "difficulty breathing", "confusion", "stiff neck",
    "black stool", "blood in stool", "severe abdominal pain", "fainting", "syncope",
    "suspected sepsis",
]
_NEG = re.compile(r"(\bno\b|\bnot\b|\bdenies\b|\bdenied\b|\bwithout\b|\bnegative for\b|\bnever\b)[^.;,]{0,25}$")


def _phrase_present(text: str, phrase: str) -> bool:
    t = text.lower()
    for m in re.finditer(re.escape(phrase), t):
        if not _NEG.search(t[max(0, m.start() - 30): m.start()]):
            return True
    return False


# diagnosis keyword -> "emergent" | "urgent"  (extend with clinician input)
CANT_MISS = {
    "myocardial infarction": "emergent", "acute coronary": "emergent", "pulmonary embolism": "emergent",
    "stroke": "emergent", "sepsis": "emergent", "aortic dissection": "emergent",
    "subarachnoid": "emergent", "meningitis": "emergent", "ectopic pregnancy": "emergent",
    "anaphylaxis": "emergent", "diabetic ketoacidosis": "emergent",
    "appendicitis": "urgent", "testicular torsion": "emergent", "gi bleed": "urgent",
    "pneumonia": "urgent", "deep vein thrombosis": "urgent", "pyelonephritis": "urgent",
}


def emergency_rules(inp: TriageInput) -> list:
    r = []
    _, hr_lo, hr_hi, rr_lo, rr_hi = _band(inp.age_years)
    if inp.sbp is not None and inp.sbp < _sbp_low(inp.age_years):
        r.append(f"hypotension: SBP {inp.sbp} < {_sbp_low(inp.age_years)}")
    if inp.spo2 is not None and inp.spo2 < 90:
        r.append(f"hypoxia: SpO2 {inp.spo2} < 90")
    if inp.hr is not None and inp.hr > hr_hi:
        r.append(f"tachycardia: HR {inp.hr} > {hr_hi}")
    if inp.hr is not None and inp.hr < hr_lo:
        r.append(f"bradycardia: HR {inp.hr} < {hr_lo}")
    if inp.rr is not None and (inp.rr > rr_hi or inp.rr < rr_lo):
        r.append(f"abnormal respiratory rate: {inp.rr}")
    if inp.temp_c is not None and (inp.temp_c > 40 or inp.temp_c < 35):
        r.append(f"extreme temperature: {inp.temp_c}")
    if inp.age_years is not None and inp.age_years < 0.25 and inp.temp_c is not None and inp.temp_c >= 38:
        r.append("fever in infant < 3 months")
    if inp.avpu and inp.avpu.upper() != "A":
        r.append(f"altered consciousness (AVPU={inp.avpu.upper()})")
    for p in EMERGENCY_PHRASES:
        if _phrase_present(inp.symptom_text, p):
            r.append(f"red-flag symptom: {p}")
    if inp.pregnant and _phrase_present(inp.symptom_text, "vaginal bleeding"):
        r.append("pregnancy with vaginal bleeding")
    return r


def urgent_rules(inp: TriageInput) -> list:
    return [f"urgent symptom: {p}" for p in URGENT_PHRASES if _phrase_present(inp.symptom_text, p)]


def cantmiss_hits(differential: Sequence[Tuple[str, float]], cfg: TriageConfig):
    emergency, urgent = [], []
    for name, prob in differential:
        n = name.lower()
        for kw, tier in CANT_MISS.items():
            if kw in n:
                if tier == "emergent" and prob >= cfg.cantmiss_emergency_prob:
                    emergency.append(f"can't-miss dx '{name}' p={prob:.2f}")
                elif prob >= cfg.cantmiss_urgent_prob:
                    urgent.append(f"can't-miss dx '{name}' p={prob:.2f}")
                break
    return emergency, urgent


# Model-independent symptom-pattern triggers (for can't-miss floor)
# These fire even if the model never lists the diagnosis
SYMPTOM_CANT_MISS = [
    # (pattern, diagnosis_name, tier, min_prob)
    # ACS
    ({"chest pain", "diaphoresis"}, "acute_coronary_syndrome", "emergent", 0.20),
    ({"chest pain", "shortness of breath"}, "acute_coronary_syndrome", "emergent", 0.20),
    ({"chest pain", "age > 45"}, "acute_coronary_syndrome", "emergent", 0.20),
    ({"crushing chest", "radiating"}, "acute_coronary_syndrome", "emergent", 0.20),
    # Stroke
    ({"face droop", "facial droop"}, "stroke", "emergent", 0.20),
    ({"slurred speech", "weakness"}, "stroke", "emergent", 0.20),
    ({"sudden weakness", "one-sided"}, "stroke", "emergent", 0.20),
    # PE
    ({"unilateral leg swelling", "dyspnea"}, "pulmonary_embolism", "emergent", 0.20),
    ({"leg swelling", "shortness of breath"}, "pulmonary_embolism", "emergent", 0.20),
    ({"calf pain", "shortness of breath"}, "pulmonary_embolism", "emergent", 0.20),
    # Anaphylaxis
    ({"lip swelling", "urticaria"}, "anaphylaxis", "emergent", 0.20),
    ({"angioedema", "hypotension"}, "anaphylaxis", "emergent", 0.20),
    # Sepsis
    ({"fever", "confusion", "hypotension"}, "sepsis", "emergent", 0.20),
    # Appendicitis
    ({"right lower quadrant pain", "fever"}, "appendicitis", "urgent", 0.05),
    ({"migratory pain", "rebound"}, "appendicitis", "urgent", 0.05),
    # Ectopic pregnancy
    ({"pregnant", "vaginal bleeding", "abdominal pain"}, "ectopic pregnancy", "emergent", 0.20),
    # Testicular torsion
    ({"testicular pain", "sudden"}, "testicular torsion", "emergent", 0.20),
    # GI bleed
    ({"hematemesis", "melena"}, "gi bleed", "urgent", 0.05),
    # Meningitis
    ({"stiff neck", "fever", "photophobia"}, "meningitis", "emergent", 0.20),
    # DKA
    ({"diabetic", "ketoacidosis"}, "diabetic ketoacidosis", "emergent", 0.20),
]


def _check_symptom_patterns(inp: TriageInput) -> list:
    """Check for model-independent symptom patterns that imply can't-miss diagnoses.
    
    Returns list of (diagnosis, tier) tuples that match.
    """
    text = inp.symptom_text.lower()
    hits = []
    for pattern, dx_name, tier, min_prob in SYMPTOM_CANT_MISS:
        if all(p in text for p in pattern):
            hits.append((dx_name, tier, min_prob))
    return hits


def triage(inp: TriageInput, differential: Sequence[Tuple[str, float]], confidence: float,
           cfg: Optional[TriageConfig] = None) -> TriageResult:
    cfg = cfg or TriageConfig()
    em = emergency_rules(inp)
    cm_em, cm_ur = cantmiss_hits(differential, cfg)
    
    # Add symptom-pattern triggers (model-independent)
    symptom_hits = _check_symptom_patterns(inp)
    for dx_name, tier, min_prob in symptom_hits:
        if tier == "emergent":
            em.append(f"symptom pattern: {dx_name} p={min_prob}")
        elif tier == "urgent":
            cm_ur.append(f"symptom pattern: {dx_name} p={min_prob}")
    
    if em or cm_em:
        return TriageResult(RiskLevel.EMERGENCY, TriageDecision.ESCALATE, em + cm_em)
    ur = urgent_rules(inp) + cm_ur
    if ur:
        return TriageResult(RiskLevel.URGENT, TriageDecision.ESCALATE, ur)
    if not differential:
        return TriageResult(RiskLevel.ROUTINE, TriageDecision.ESCALATE, ["no differential produced; human review"])
    if confidence >= cfg.high_confidence:
        return TriageResult(RiskLevel.HOME_CARE, TriageDecision.ANSWER, [f"confidence {confidence:.2f} and no red flags"])
    if confidence >= cfg.low_confidence:
        questions = generate_followup_questions(
            symptom_text=inp.symptom_text,
            differential=differential,
            patient_age=inp.age_years,
            pregnant=inp.pregnant,
        )
        return TriageResult(
            RiskLevel.ROUTINE,
            TriageDecision.ASK_FOLLOWUP,
            [f"moderate confidence {confidence:.2f}"],
            followup_questions=questions,
        )
    return TriageResult(cfg.low_confidence_risk, TriageDecision.ESCALATE, [f"low confidence {confidence:.2f}; human review"])


# =====================================================================
# Clinical Follow-Up Questions (for moderate confidence / triage clarification)
# =====================================================================

FOLLOWUP_DOMAINS: list[tuple[re.Pattern, list[str]]] = [
    (
        re.compile(r"headache|migraine|cephalea|cluster|tension", re.IGNORECASE),
        [
            "Are you experiencing nausea, vomiting, or sensitivity to light (photophobia) or sound?",
            "How long do the headache episodes last, and is the pain pulsating/throbbing on one side or a steady dull ache?",
            "Does physical activity, stress, or lack of sleep trigger or worsen the headache?",
        ],
    ),
    (
        re.compile(r"cough|bronchitis|asthma|wheez|dyspnea|shortness of breath|copd|pharyngitis|sore throat|sinusitis", re.IGNORECASE),
        [
            "How many days have you had this cough, and is it dry or producing phlegm/mucus (and if so, what color)?",
            "Do you notice shortness of breath, wheezing, or chest tightness, particularly at night or with exertion?",
            "Have you had fever, chills, body aches, or nasal congestion accompanying this?",
        ],
    ),
    (
        re.compile(r"chest pain|angina|palpitation|coronary|arrhythmia|heart", re.IGNORECASE),
        [
            "Does the chest discomfort radiate to your left arm, jaw, neck, or back, or change when you breathe deeply?",
            "Does the discomfort occur primarily during physical activity, and does resting relieve it?",
            "Are you experiencing associated symptoms such as cold sweats, dizziness, or shortness of breath?",
        ],
    ),
    (
        re.compile(r"abdominal|epigastric|stomach|gerd|reflux|nausea|vomit|diarrhea|constipation|peptic|gastritis|bowel", re.IGNORECASE),
        [
            "Where in your abdomen is the discomfort located, and does it seem related to eating or specific types of foods?",
            "Have you noticed heartburn, acid regurgitation, nausea, or changes in bowel movements?",
            "Have you seen any dark or tarry stools, visible blood, or experienced unintended weight loss?",
        ],
    ),
    (
        re.compile(r"urinary|dysuria|uti|cystitis|flank|pyelonephritis|kidney|bladder", re.IGNORECASE),
        [
            "Are you experiencing pain or burning during urination, or an urge to urinate much more frequently than usual?",
            "Have you noticed any lower back/flank pain, fever, chills, or pink/cloudy urine?",
            "Do you have a history of recurrent urinary tract infections or kidney stones?",
        ],
    ),
    (
        re.compile(r"joint|arthritis|back pain|lumbar|knee|shoulder|neck pain|strain|sprain|sciatica|stiffness", re.IGNORECASE),
        [
            "Did this pain begin following a specific injury or heavy lifting, or did it start gradually over time?",
            "Is there noticeable joint stiffness when you wake up in the morning, and how long does it take to loosen?",
            "Does the pain shoot or radiate down your legs or arms, or cause any numbness/tingling?",
        ],
    ),
    (
        re.compile(r"rash|itching|pruritus|eczema|dermatitis|urticaria|lesion|hives|skin", re.IGNORECASE),
        [
            "Is the rash itchy, tender, or warm, and where on your body did it first appear?",
            "Have you recently used new medications, soaps, cosmetics, or been exposed to unusual plants or insect bites?",
            "Have you had a fever or any swelling of your face, lips, or eyelids along with the skin symptoms?",
        ],
    ),
    (
        re.compile(r"fatigue|tired|weakness|thyroid|hypothyroid|diabetes|hyperglycemia", re.IGNORECASE),
        [
            "How long have you felt this fatigue, and does restful sleep or taking time off improve your energy?",
            "Have you noticed changes in weight, sensitivity to cold or heat, dry skin, or hair thinning?",
            "Have you had increased thirst, frequent nighttime urination, or changes in your vision?",
        ],
    ),
]

GENERAL_FOLLOWUP_QUESTIONS: list[str] = [
    "When did these symptoms first begin, and have they been constant, intermittent, or getting progressively worse?",
    "On a scale of 1 to 10, how severe is the discomfort, and does anything specific make it better or worse?",
    "Are you taking any medications or supplements, or do you have any relevant medical conditions or allergies?",
]


def generate_followup_questions(
    symptom_text: str = "",
    differential: Sequence[Tuple[str, float]] | None = None,
    patient_age: Optional[float] = None,
    pregnant: bool = False,
    max_questions: int = 3,
) -> list[str]:
    """Generate clinically targeted follow-up clarifying questions.

    When triage assigns ASK_FOLLOWUP (moderate confidence, non-emergent presentation),
    these discriminating questions gather diagnostic precision on the differential
    diagnoses and presenting symptoms to guide clinical decision-making.
    """
    corpus = (symptom_text or "").lower()
    if differential:
        dx_texts = " ".join(dx for dx, _ in differential)
        corpus = f"{corpus} {dx_texts.lower()}"

    questions: list[str] = []

    if pregnant:
        questions.append("How many weeks pregnant are you, and have you experienced any cramping, spotting, or headaches?")

    for pattern, domain_questions in FOLLOWUP_DOMAINS:
        if pattern.search(corpus):
            for q in domain_questions:
                if q not in questions:
                    questions.append(q)
                if len(questions) >= max_questions:
                    return questions[:max_questions]

    # If domain matching produced fewer than max_questions, fill with general clarifying questions
    for gq in GENERAL_FOLLOWUP_QUESTIONS:
        if gq not in questions:
            questions.append(gq)
        if len(questions) >= max_questions:
            break

    return questions[:max_questions]
