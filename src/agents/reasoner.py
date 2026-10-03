"""LLM reasoning agent with template-based deterministic mock fallback.

The mock reasoner is invoked when no real LLM is available; it enables
end-to-end testing of the AEB pipeline without GPU/network access.
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

        pyelo_boost = 1.0
        if "pyelonephritis" in query_l or ("flank pain" in query_l and ("fever" in query_l or "chills" in query_l)):
            pyelo_boost = 1.7

        for pattern, dx, prior in _DX_PATTERNS:
            pat = re.compile(r"(?<![a-z0-9])(" + pattern + r")(?![a-z0-9])")

            matching: list[EvidenceChunk] = []
            for ch in evidence:
                text = ch.text.lower() + " " + (ch.title or "").lower()
                if pat.search(text):
                    matching.append(ch)

            if not matching:
                continue

            relevance = sum(max(0.0, ch.similarity_score) for ch in matching) / max(len(matching), 1)
            coverage = len(matching) / max(len(evidence), 1)
            evidence_support = 0.7 * coverage + 0.3 * relevance

            query_hit = pat.search(query_l) is not None

            if query_hit:
                alternatives = [a.strip() for a in pattern.split("|") if a.strip()]
                matched_alts = [a for a in alternatives if a in query_l]
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
        if not getattr(top, "query_signal", False):
            candidates = ", ".join(d.diagnosis for d in scored[:3])
            return ClinicalResponse(
                primary_diagnosis=None,
                differential=[],
                confidence=0.0,
                uncertainty=1.0,
                reasoning=(
                    "Insufficient evidence to determine a confident differential diagnosis. "
                    f"Possible conditions considered: {candidates}. Clinician review is advised."
                ),
                decision=TriageDecision.ESCALATE,
                escalated_reason="No query-anchored diagnosis identified; evidence does not support a confident recommendation.",
            )

        second = scored[1] if len(scored) > 1 else None
        margin = (top.probability - (second.probability if second else 0.0)) if second else top.probability
        evidence_richness = min(1.0, len(top.supporting_evidence) / 3.0)
        confidence = round(float(min(1.0, 0.5 + margin * 0.6 + evidence_richness * 0.15)), 4)

        quote = ""
        if top.supporting_evidence:
            snippet = top.supporting_evidence[0].split("]", 1)[-1].strip().replace("\n", " ")
            snippet = re.split(r"(?<=[.!?])\s+", snippet)[0]
            quote = snippet[:220]

        reasoning_lines = [f"The patient presentation is most consistent with {top.diagnosis}."]
        if quote:
            reasoning_lines.append(f"Supporting evidence: {quote}")
        if second:
            reasoning_lines.append(f"An alternative differential diagnosis is {second.diagnosis}.")

        return ClinicalResponse(
            primary_diagnosis=top.diagnosis,
            differential=scored,
            confidence=confidence,
            uncertainty=round(1 - confidence, 4),
            reasoning="\n".join(reasoning_lines),
        )


def _parse_llm_json(text: str) -> dict:
    """Robust 3-tier JSON parser for open-source LLM outputs."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

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


def _strip_bad_pipe_kwargs(pipe) -> None:
    """Ensure max_memory never gets forwarded into generate()."""
    targets = [pipe]
    if hasattr(pipe, "model"):
        targets.append(pipe.model)

    for target in targets:
        gen_cfg = getattr(target, "generation_config", None)
        if gen_cfg is not None:
            if hasattr(gen_cfg, "max_memory"):
                delattr(gen_cfg, "max_memory")
            if hasattr(gen_cfg, "_extra_kwargs") and isinstance(gen_cfg._extra_kwargs, dict):
                gen_cfg._extra_kwargs.pop("max_memory", None)

        cfg = getattr(target, "config", None)
        if cfg is not None:
            if hasattr(cfg, "max_memory"):
                delattr(cfg, "max_memory")
            if hasattr(cfg, "_extra_kwargs") and isinstance(cfg._extra_kwargs, dict):
                cfg._extra_kwargs.pop("max_memory", None)

        for attr in ("model_kwargs", "_forward_params", "_preprocess_params", "_postprocess_params", "_extra_kwargs"):
            d = getattr(target, attr, None)
            if isinstance(d, dict):
                d.pop("max_memory", None)


class OpenSourceLLMReasoner(BaseReasoner):
    """Real LLM reasoner with 4-bit quantization (local) or fp16 (Kaggle)."""

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
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

        tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id

        if self.load_in_4bit and torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                low_cpu_mem_usage=True,
            )
            self._pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tok,
                return_full_text=False,
            )
            _strip_bad_pipe_kwargs(self._pipe)
            return

        # fp16/cpu path (FIX: max_memory goes to from_pretrained, NOT pipeline)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None

        max_memory = None
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            # your original intent preserved:
            max_memory = {0: "9GiB", 1: "7GiB"}

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device_map,
            max_memory=max_memory,          # OK HERE
            low_cpu_mem_usage=True,
        )
        self._pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            return_full_text=False,
        )
        _strip_bad_pipe_kwargs(self._pipe)

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

        # Note: don't pass generation_config explicitly; just pass generation args.
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
    """Factory: Mock, OpenSource (4-bit/fp16), or HF API reasoner."""
    if prefer_real:
        if backend in ("hf_inference_api", "hf_api"):
            return HFInferenceReasoner(model_name=model_name or "microsoft/Llama3-Med-8B-Instruct", hf_token=hf_token)
        load_in_4bit = (backend != "kaggle_fp16")
        return OpenSourceLLMReasoner(
            model_name=model_name or "microsoft/Llama3-Med-8B-Instruct",
            load_in_4bit=load_in_4bit,
        )

    if backend == "mock":
        return MockReasoner()

    if backend in ("hf_inference_api", "hf_api"):
        return HFInferenceReasoner(model_name=model_name or "microsoft/Llama3-Med-8B-Instruct", hf_token=hf_token)

    if backend in ["local_4bit", "kaggle_fp16", "real"]:
        load_in_4bit = (backend != "kaggle_fp16")
        return OpenSourceLLMReasoner(
            model_name=model_name or "microsoft/Llama3-Med-8B-Instruct",
            load_in_4bit=load_in_4bit,
        )

    raise ValueError(f"Unknown backend: {backend}")