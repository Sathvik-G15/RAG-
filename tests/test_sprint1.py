"""Unit tests for Sprint 1 components: hardware detection, chunker, JSON parser, and RAGAS harness."""

from src.agents.reasoner import _parse_llm_json, make_reasoner
from src.config import detect_hardware
from src.data.guidelines_loader import chunk_document


def test_detect_hardware():
    hw = detect_hardware()
    assert "device" in hw
    assert "vram_gb" in hw
    assert "free_disk_gb" in hw
    assert "serving_backend" in hw
    assert hw["serving_backend"] in ["local_4bit", "kaggle_fp16", "hf_inference_api"]


def test_robust_json_parser():
    # Test 1: Clean JSON
    res1 = _parse_llm_json('{"primary_diagnosis": "pneumonia", "confidence": 0.88}')
    assert res1["primary_diagnosis"] == "pneumonia"
    assert res1["confidence"] == 0.88

    # Test 2: Preamble text + JSON
    res2 = _parse_llm_json('Here is my clinical assessment:\n```json\n{"primary_diagnosis": "asthma", "confidence": 0.75}\n```\nHope this helps.')
    assert res2["primary_diagnosis"] == "asthma"
    assert res2["confidence"] == 0.75

    # Test 3: Malformed JSON with regex fallback
    res3 = _parse_llm_json('Result: "primary_diagnosis": "copd", "confidence": 0.65, reasoning: incomplete')
    assert res3["primary_diagnosis"] == "copd"
    assert res3["confidence"] == 0.65


def test_chunk_document():
    doc = {
        "text": "Diagnosis of community acquired pneumonia requires clinical signs and chest radiograph. First-line treatment is oral amoxicillin.",
        "title": "CAP Guideline",
        "source": "WHO",
        "specialty": "infectious_disease",
        "year": "2023",
        "url": "http://who.int/cap",
    }
    chunks = chunk_document(doc, chunk_size=20, chunk_overlap=5)
    assert len(chunks) >= 1
    assert chunks[0].title == "CAP Guideline"
    assert chunks[0].metadata["specialty"] == "infectious_disease"


def test_make_reasoner_factory():
    r_mock = make_reasoner(backend="mock")
    assert r_mock.__class__.__name__ == "MockReasoner"
