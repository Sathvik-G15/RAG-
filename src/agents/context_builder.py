"""Patient Context Builder.

Augments a free-text clinical query with structured patient profile data
to enable personalized retrieval (e.g., geriatric / diabetic / renal / obstetric
boosting) downstream.

The builder is intentionally deterministic and rule-based so it is reproducible
and does not require an LLM to run.
"""

from __future__ import annotations

import re

from ..models import PatientProfile, Visit

_AGE_RE = re.compile(r"(\d{1,3})\s*[-\s]?year[-\s]old", re.IGNORECASE)
_GENDER_M = re.compile(r"\b(male|man|gentleman|he|him|his)\b", re.IGNORECASE)
_GENDER_F = re.compile(r"\b(female|woman|lady|she|her)\b", re.IGNORECASE)

_COMORBIDITY_TOKENS = (
    "diabetes", "hypertension", "copd", "asthma", "ckd", "kidney disease",
    "hypothyroidism", "anemia", "anaphylaxis", "stroke", "mi",
)


def infer_profile(query: str, explicit: PatientProfile | None = None) -> PatientProfile:
    """Best-effort extraction of demographic features from the query text.

    Any explicitly supplied profile field takes precedence.
    """
    text = " " + (query or "").lower() + " "
    if explicit is None:
        explicit = PatientProfile()

    age = explicit.age
    if age is None:
        m = _AGE_RE.search(text)
        if m:
            age = max(0, min(120, int(m.group(1))))

    gender: str | None = explicit.gender
    if gender is None:
        if _GENDER_F.search(text):
            gender = "female"
        elif _GENDER_M.search(text):
            gender = "male"

    comorbid = set(explicit.comorbidities)
    for tk in _COMORBIDITY_TOKENS:
        if tk in text and tk not in comorbid:
            comorbid.add(tk)

    pregnancy = explicit.pregnancy_status
    if pregnancy is None and "pregnan" in text:
        pregnancy = True

    return PatientProfile(
        age=age,
        gender=gender,
        pregnancy_status=pregnancy,
        comorbidities=sorted(comorbid),
        medications=list(explicit.medications),
        allergies=list(explicit.allergies),
    )


def build_context_query(query: str, profile: PatientProfile | None = None, visit: Visit | None = None) -> str:
    """Produce an augmented retrieval string that combines the clinical question
    with patient-specific tokens (demographics + comorbidities + allergies).
    """
    if profile is None:
        profile = infer_profile(query)

    extras: list[str] = []
    if profile.age is not None:
        band = ("pediatric" if profile.age < 16
                else "adult"
                if 16 <= profile.age < 65
                else "geriatric")
        extras += [format(profile.age, "d"), "year", "old", band]
    if profile.gender:
        extras.append(profile.gender)
    if profile.pregnancy_status:
        extras.append("pregnancy")
    for c in profile.comorbidities:
        extras.append(c)
    for a in profile.allergies:
        extras.append("allergy to " + a)

    vital_tokens: list[str] = []
    if visit is not None and visit.vitals is not None:
        v = visit.vitals
        if v.systolic_bp and v.systolic_bp >= 180:
            vital_tokens.append("hypertensive crisis")
        elif v.systolic_bp and v.systolic_bp < 90:
            vital_tokens.append("hypotension")
        if v.oxygen_saturation and v.oxygen_saturation < 92:
            vital_tokens.append("hypoxia")
        if v.heart_rate and v.heart_rate >= 120:
            vital_tokens.append("tachycardia")

    extras.extend(vital_tokens)

    augmented = query.strip()
    if extras:
        augmented = augmented + " " + " ".join(extras)
    return augmented
