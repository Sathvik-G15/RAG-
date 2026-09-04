"""LLM reasoning agent with template-based deterministic mock fallback.

The mock reasoner is invoked when no real LLM is available; it enables
end-to-end end-to-end testing of the AEB pipeline (confidence, escalation,
verification) without GPU/network access. The mock is intentionally designed
to produce differential diagnoses tied to keywords observed in retrieved chunks,
giving the verification layer something meaningful to score.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..models import ClinicalResponse, DifferentialDiagnosis, EvidenceChunk, TriageDecision

_DX_PATTERNS: list[tuple[str, str, float]] = [
    ("acute coronary|chest pain|stemi|myocardial infarct|cardiac|angina", "acute_coronary_syndrome", 0.90),
    ("pneumonia|consolidation|curb-65|lobar|pulmonary infiltrat|crackle|productive cough", "community_acquired_pneumonia", 0.86),
    ("urinary tract|cystitis|uti|dysuria", "uncomplicated_uti", 0.88),
    ("pyelonephritis|flank pain", "pyelonephritis", 0.82),
    ("copd|bronchitis|bronchodilator|fev1|wheeze", "copd_exacerbation", 0.83),
    ("asthma|wheeze|bronchospasm|beta-agonist|chest tightness", "asthma_exacerbation", 0.85),
    ("anaphylaxis|urticaria|angioedema", "anaphylaxis", 0.95),
    ("hypothyroid|levothyroxine|tsh|thyroid", "hypothyroidism", 0.85),
    ("stroke|cerebrovascular|thrombectomy|alteplase|slurred speech|facial droop|sudden weakness", "acute_ischemic_stroke", 0.88),
    ("covid|sars-cov|nirmatrelvir", "covid_19", 0.78),
    ("sepsis|septic|sofa|qsofa|lactate", "sepsis", 0.84),
    ("hypertension|high blood pressure", "hypertension", 0.75),
    ("type 2 diabetes|t2dm|metformin|a1c|hba1c", "type_2_diabetes", 0.74),
    ("gastritis|peptic ulcer|nsaid|h. pylori|helicobacter|epigastric", "gastritis", 0.76),
    ("aki|acute kidney|creatinine|kdigo", "acute_kidney_injury", 0.77),
    ("influenza|flu|oseltamivir", "influenza", 0.80),
]


class BaseReasoner:
    def generate(self, query: str, evidence: Sequence[EvidenceChunk]) -> ClinicalResponse:
        raise NotImplementedError


@dataclass
class MockReasoner(BaseReasoner):
    """Deterministic mock llm that scores patterns from evidence & query."""

    def generate(self, query: str, evidence: Sequence[EvidenceChunk]) -> ClinicalResponse:
        query_l = query.lower()
        scored: list[DifferentialDiagnosis] = []

        # Co-occurrence disambiguation: fever + flank pain points to pyelonephritis
        # rather than simple cystitis, even though "dysuria" is shared.
        pyelo_boost = 1.0
        if "pyelonephritis" in query_l or ("flank pain" in query_l and ("fever" in query_l or "chills" in query_l)):
            pyelo_boost = 1.7

        for pattern, dx, prior in _DX_PATTERNS:
            # Word-boundary anchors prevent partial-word matches (e.g. "flu"
            # must not match "fluoroquinolone").
            pat = re.compile(r"(?<![a-z0-9])(" + pattern + r")(?![a-z0-9])")

            matching: list[EvidenceChunk] = []
            for ch in evidence:
                text = ch.text.lower() + " " + (ch.title or "").lower()
                if pat.search(text):
                    matching.append(ch)

            if not matching:
                continue

            # Evidence support weighted by retrieval relevance (similarity score).
            relevance = sum(max(0.0, ch.similarity_score) for ch in matching) / max(len(matching), 1)
            coverage = len(matching) / max(len(evidence), 1)
            evidence_support = 0.7 * coverage + 0.3 * relevance

            query_hit = pat.search(query_l) is not None

            if query_hit:
                # Query text directly signals this diagnosis; evidence confirms it.
                # Credit exclusive (non-shared) pattern terms more than shared ones:
                # e.g. "flank pain" points to pyelonephritis; "dysuria" is shared with UTI.
                alternatives = [a.strip() for a in pattern.split("|") if a.strip()]
                matched_alts = [a for a in alternatives if a in query_l]
                # Negation handling: exclude alternatives explicitly negated in the
                # query (e.g. "without flank pain" should not trigger pyelonephritis).
                negated = re.compile(r"\b(?:without|no|denies|denies any|negative for)\b")
                negated_alts = [
                    a for a in matched_alts
                    if negated.search(query_l[max(0, query_l.find(a) - 40):query_l.find(a)])
                ]
                matched_alts = [a for a in matched_alts if a not in negated_alts]
                if not matched_alts:
                    continue
                shared: set[str] = set()
                for other_pat, _, _ in _DX_PATTERNS:
                    if other_pat == pattern:
                        continue
                    for oa in other_pat.split("|"):
                        oa = oa.strip()
                        if oa and oa in query_l:
                            shared.add(oa)
                exclusive = [a for a in matched_alts if a not in shared]
                hits = len(matched_alts)
                specificity = 0.5 + 0.5 * min(1.5, hits * 0.4 + len(exclusive))
                boost = pyelo_boost if dx == "pyelonephritis" else 1.0
                raw = prior * specificity * (0.55 + 0.45 * evidence_support) * boost
            else:
                # Evidence-only hypothesis: needs substantial corroboration to rank.
                if evidence_support < 0.35:
                    continue
                raw = prior * 0.35 * evidence_support

            support: list[str] = []
            sources_used: list[str] = []
            seen_sources: set[str] = set()
            for ch in matching:
                snippet = ch.text[:160].replace("\n", " ")
                support.append(f"[{ch.source}] {snippet}")
                if ch.source not in seen_sources:
                    seen_sources.add(ch.source)
                    sources_used.append(ch.source)

            scored.append(
                DifferentialDiagnosis(
                    diagnosis=dx,
                    probability=round(float(raw), 4),
                    supporting_evidence=support[:3],
                    sources=sources_used,
                    # Carry the query-signal flag for confidence computation below.
                    # We use a private marker key since the model is generic.
                )
            )
            scored[-1].query_signal = query_hit

        if not scored:
            return ClinicalResponse(
                primary_diagnosis=None,
                differential=[],
                confidence=0.0,
                uncertainty=1.0,
                reasoning="No supported diagnosis identified from retrieved evidence.",
                decision=TriageDecision.ESCALATE,
                escalated_reason="Insufficient evidence to determine differential diagnosis.",
            )

        scored.sort(key=lambda d: d.probability, reverse=True)
        scored = scored[:5]

        total = sum(d.probability for d in scored) or 1.0
        for d in scored:
            d.probability = round(d.probability / total, 4)

        top = scored[0]

        # Safety: if the leading hypothesis was never signaled by the query text
        # itself (only inferred from loose evidence patterns), we do NOT have a
        # confident diagnosis. An out-of-scope query must escalate rather than
        # receive a confidently wrong single-diagnosis answer.
        if not getattr(top, "query_signal", False):
            candidates = ", ".join(d.diagnosis for d in scored[:3])
            return ClinicalResponse(
                primary_diagnosis=None,
                differential=[],
                confidence=0.0,
                uncertainty=1.0,
                reasoning=(
                    "Insufficient evidence to determine a confident differential diagnosis. "
                    f"Possible conditions considered: {candidates}. "
                    "Clinician review is advised."
                ),
                decision=TriageDecision.ESCALATE,
                escalated_reason="No query-anchored diagnosis identified; evidence does not support a confident recommendation.",
            )

        second = scored[1] if len(scored) > 1 else None

        # Confidence: agreement between top-1 and runner-up plus evidence richness.
        margin = (top.probability - (second.probability if second else 0.0)) if second else top.probability
        evidence_richness = min(1.0, len(top.supporting_evidence) / 3.0)
        confidence = round(float(min(1.0, 0.5 + margin * 0.6 + evidence_richness * 0.15)), 4)

        # Reasoning cites evidence verbatim so the verification layer can score it.
        quote = ""
        if top.supporting_evidence:
            snippet = top.supporting_evidence[0].split("]", 1)[-1].strip().replace("\n", " ")
            snippet = re.split(r"(?<=[.!?])\s+", snippet)[0]
            quote = snippet[:220]

        reasoning_lines = [
            f"The patient presentation is most consistent with {top.diagnosis}.",
        ]
        if quote:
            reasoning_lines.append(f"Supporting evidence: {quote}")
        if second:
            reasoning_lines.append(
                f"An alternative differential diagnosis is {second.diagnosis}."
            )

        return ClinicalResponse(
            primary_diagnosis=top.diagnosis,
            differential=scored,
            confidence=confidence,
            uncertainty=round(1 - confidence, 4),
            reasoning="\n".join(reasoning_lines),
        )


def _parse_llm_json(text: str) -> dict:
    """Robust 3-tier JSON parser for 8B and smaller open-source LLMs."""
    # Attempt 1: direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Attempt 2: balanced curly brace regex extraction
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Attempt 3: fallback field-by-field regex extraction
    diag_match = re.search(r'"primary_diagnosis"\s*:\s*"([^"]+)"', text)
    conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
    diag = diag_match.group(1) if diag_match else None
    conf = float(conf_match.group(1)) if conf_match else 0.50

    return {
        "primary_diagnosis": diag,
        "confidence": conf,
        "uncertainty": round(1.0 - conf, 4),
        "differential": [
            {
                "diagnosis": diag or "unspecified_condition",
                "probability": conf,
                "supporting_evidence": [],
                "sources": [],
            }
        ] if diag else [],
        "reasoning": text[:600].strip(),
    }


class OpenSourceLLMReasoner(BaseReasoner):
    """Real LLM reasoner with 4-bit quantization (local) or fp16 (Kaggle T4)."""

    def __init__(
        self,
        model_name: str = "microsoft/Llama3-Med-8B-Instruct",
        load_in_4bit: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.95,
    ):
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._pipe = None

    def _load(self) -> None:
        import torch
        from transformers import BitsAndBytesConfig, pipeline

        if self.load_in_4bit and torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            )
            self._pipe = pipeline(
                "text-generation",
                model=self.model_name,
                model_kwargs={"quantization_config": quant_config},
                device_map="auto",
                max_new_tokens=self.max_new_tokens,
            )
        else:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self._pipe = pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                max_new_tokens=self.max_new_tokens,
            )

    def generate(self, query: str, evidence: Sequence[EvidenceChunk]) -> ClinicalResponse:
        if self._pipe is None:
            self._load()

        evidence_block = "\n\n".join(
            f"[{i+1}] SOURCE={c.source}; TITLE={c.title or ''}:\n{c.text}"
            for i, c in enumerate(evidence)
        )
        prompt = (
            "You are a clinical decision support assistant. Provide a structured JSON response.\n"
            "Patient query: " + query + "\n\n"
            "Retrieved evidence:\n" + evidence_block + "\n\n"
            "Output valid JSON only with keys: primary_diagnosis (string or null), confidence (float between 0.0 and 1.0), "
            "differential (list of objects with diagnosis, probability, supporting_evidence, sources), reasoning (string)."
        )

        outputs = self._pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            return_full_text=False,
        )
        text = outputs[0]["generated_text"]
        payload = _parse_llm_json(text)
        return ClinicalResponse(**payload)


class HFInferenceReasoner(BaseReasoner):
    """Cloud serverless reasoner calling Hugging Face Inference API (0 local VRAM)."""

    def __init__(
        self,
        model_name: str = "microsoft/Llama3-Med-8B-Instruct",
        hf_token: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ):
        import os

        from huggingface_hub import InferenceClient

        self.model_name = model_name
        token = hf_token or os.environ.get("HF_TOKEN")
        self.client = InferenceClient(model=model_name, token=token)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, query: str, evidence: Sequence[EvidenceChunk]) -> ClinicalResponse:
        evidence_block = "\n\n".join(
            f"[{i+1}] SOURCE={c.source}; TITLE={c.title or ''}:\n{c.text}"
            for i, c in enumerate(evidence)
        )
        prompt = (
            "You are a clinical decision support assistant. Provide a structured JSON response.\n"
            "Patient query: " + query + "\n\n"
            "Retrieved evidence:\n" + evidence_block + "\n\n"
            "Output JSON with keys: primary_diagnosis, confidence (0.0 to 1.0), differential, reasoning."
        )

        text = self.client.text_generation(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        payload = _parse_llm_json(text)
        return ClinicalResponse(**payload)


class LLMReasoner(OpenSourceLLMReasoner):
    """Backwards-compatible alias for OpenSourceLLMReasoner."""
    pass


def make_reasoner(
    backend: str = "mock",
    prefer_real: bool = False,
    model_name: str | None = None,
    hf_token: str | None = None,
) -> BaseReasoner:
    """Factory that creates Mock, OpenSource (4-bit/fp16), or HF API reasoner."""
    if backend == "hf_inference_api" or (prefer_real and backend == "hf_api"):
        try:
            return HFInferenceReasoner(model_name=model_name or "microsoft/Llama3-Med-8B-Instruct", hf_token=hf_token)
        except Exception:
            pass

    if backend in ["local_4bit", "kaggle_fp16", "real"] or (prefer_real and model_name):
        load_in_4bit = backend == "local_4bit" or (backend == "real" and prefer_real)
        try:
            return OpenSourceLLMReasoner(
                model_name=model_name or "microsoft/Llama3-Med-8B-Instruct",
                load_in_4bit=load_in_4bit,
            )
        except Exception:
            pass

    return MockReasoner()

