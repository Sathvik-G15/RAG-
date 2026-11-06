# CAAR-CDSS Hardening Guide

Follow the phases in order. Every phase ends with a command that must pass before you continue.
Package contents are drop-in modules (tested: `pytest tests -q` -> 22 passed). I wrote them against the
paths in your project overview; where your real code differs, the "Wire it in" step shows what to change.

## Phase 0: Install (5 min)
1. Copy this folder's `src/`, `scripts/`, `tests/`, `configs/hardening.yaml` into your repo root. Do NOT overwrite
   `src/confidence/triage.py` until you have read Phase 1 (the new one replaces the old).
2. `pip install -r requirements-hardening.txt`
3. `pytest tests -q` -> must print `22 passed`.

## Phase 1: Safety-critical fixes (do first)

### 1.1 Replace triage (fixes: confidence-only HOME_CARE, adult-only thresholds, missing vitals)
File: `src/confidence/triage.py` (new version provided).
Behaviour:
- Acuity = vitals (age-banded) + red-flag symptoms (negation-aware) + can't-miss differential.
- Confidence only picks ANSWER / ASK_FOLLOWUP / human review. It can never lower acuity.
- 90% confident PE -> EMERGENCY. Any can't-miss diagnosis with p >= 0.05 -> at least URGENT.
Wire it in:
```python
from src.confidence.triage import triage, TriageInput, TriageConfig
ti = TriageInput(age_years=profile.age, sbp=vitals.systolic_bp, hr=vitals.heart_rate, spo2=vitals.spo2,
                 rr=vitals.resp_rate, temp_c=vitals.temperature_c, avpu=getattr(vitals, "avpu", None),
                 pregnant=profile.pregnant, symptom_text=query_text)
res = triage(ti, [(d.diagnosis, d.probability) for d in differential], confidence,
             TriageConfig(**cfg["triage"]))
response.risk_level, response.decision, response.triage_reasons = res.risk_level.value, res.decision.value, res.reasons
```
If your `models.py` already has RiskLevel/Decision enums, import those instead and map by name.
Add `triage_reasons: list[str]` to `ClinicalResponse`.
**Required human step:** a clinician must review `_AGE_BANDS`, `EMERGENCY_PHRASES`, `URGENT_PHRASES`, `CANT_MISS`
and you must cite the protocol used (ESI, NEWS2, PEWS or WHO) in the paper. The numbers are illustrative defaults.
Check: `pytest tests -q -k "boundaries or cantmiss or age_adjusted"`

### 1.2 Contamination check (fixes: benchmark leakage)
1. Make sure the KB contains only guidelines/textbooks. Never ingest question sets into the retrieval KB.
   In `scripts/ingest_benchmarks.py`, ingest benchmarks into a SEPARATE collection (`eval_questions`) that
   retrieval never queries, or skip ingestion entirely.
2. Run:
```
python scripts/contamination_check.py --kb data/unified_clinical_kb.jsonl \
  --bench data/medqa_test.jsonl data/medmcqa_test.jsonl \
  --question-key question --options-key options --kb-text-key text --out reports/contamination.json
```
3. Exit code must be 0. If 1, open `reports/contamination.json`, remove the offending KB records, rebuild, rerun.
4. Put the flagged rate (should be about 0) in the paper. Add this command to CI.
Adjust `--*-key` flags to your JSONL field names. KB records need a `source` field.

### 1.3 Privacy and API hardening
File: `src/api/security.py` (new). Edit `src/api/app.py`:
```python
from src.api.security import (require_role, validate_upload, trust_for_source, RateLimiter,
                              redact_phi, wrap_evidence, MAX_UPLOAD_BYTES, UNTRUSTED_TRUST)
analyze_limiter = RateLimiter(30, 60)
eval_limiter = RateLimiter(2, 3600)

@app.post("/analyze", dependencies=[Depends(analyze_limiter.dependency())])
...
@app.post("/evaluation/run", dependencies=[Depends(eval_limiter.dependency())])
async def run_eval(claims=Depends(require_role("admin", get_current_claims))): ...

@app.post("/guidelines/upload")
async def upload(file: UploadFile, claims=Depends(require_role("admin", get_current_claims))):
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    validate_upload(file.filename, data)
    # ingest with trust_score=UNTRUSTED_TRUST and status="quarantined"; ignore any client-sent trust value
```
- `get_current_claims` = your existing JWT dependency returning the payload dict (must contain `role` or `roles`).
- In retrieval, filter out chunks whose `status == "quarantined"`. An admin approval endpoint sets
  `status="approved"` and a trust score from the `source_trust` allowlist in `configs/kb.yaml`
  (`trust_for_source(source, allowlist)`).
- Reasoner prompts: build evidence with `wrap_evidence([c.text for c in chunks])` instead of plain concatenation.
- Logging (put in `app.py` startup):
```python
import logging
from src.api.security import redact_phi
class PHIFilter(logging.Filter):
    def filter(self, record):
        record.msg = redact_phi(str(record.msg)); return True
logging.getLogger().addFilter(PHIFilter())
```
- Block external inference unless data is synthetic. First line of `HFInferenceReasoner.__init__`:
```python
if not cfg["security"]["external_inference_allowed"]:
    raise RuntimeError("External inference disabled: patient data must not leave this system.")
```
- JWT: read the secret from an environment variable, set expiry to 15-30 min, and confirm `configs/secrets.yaml`
  and `.env` are in `.gitignore` (`git check-ignore configs/secrets.yaml` must print the path).
Check: `pytest tests -q -k "upload or ratelimit or require_role"`

## Phase 2: Verification and AEB loop

### 2.1 Verification (fixes: negation blindness, div-by-zero, no contradiction tracking, silent fallback)
File: `src/verification/claims.py` (new). In `src/verification/verifier.py`:
- Delete the old claim splitting and scoring. Use `split_claims`, `verify_claims`, `lexical_nli`, `DeBERTaNLI`.
```python
from src.verification.claims import split_claims, verify_claims, lexical_nli, DeBERTaNLI
nli = DeBERTaNLI() if cfg["verification"]["backend"] == "deberta" else lexical_nli  # no try/except fallback
report = verify_claims(split_claims(reasoning_text), [c.text for c in chunks], nli,
                       cfg["verification"]["entail_thr"], cfg["verification"]["contra_thr"], cfg["verification"]["backend"])
```
- The `VerificationReport` has `hallucination_score`, `contradiction_rate`, `verifiable`. Store all in `AEBStep` and results.
- If DeBERTa fails to load, the run must crash. Never fall back silently. For headline results set `backend: deberta`;
  lexical is for CI only. Record the verifier name and model revision in every results file.
- Your two thresholds (0.26 lexical, 0.7 DeBERTa) are untuned defaults. Tune on the labelled set in Phase 4.3.

### 2.2 AEB loop (fixes: lucky-spike stopping, non-monotonic confidence, plain k growth, unverifiable stops)
File: `src/confidence/stopping.py` (new). Replace the loop in `src/confidence/aeb.py`:
```python
from src.confidence.stopping import StoppingPolicy, reformulate_query, drop_rate
policy = StoppingPolicy(threshold=cfg["threshold"], min_steps=cfg["min_steps"], patience=cfg["patience"],
                        max_contradiction_rate=cfg["max_contradiction_rate"], max_k=cfg["max_k"])
k, conf_curve, query = cfg["initial_k"], [], original_query
while True:
    chunks = retriever.retrieve(query, k)
    reasoning = reasoner.run(profile, chunks)
    report = verifier(reasoning, chunks)               # from 2.1
    conf = confidence_model(signals_from(reasoning, chunks, report))
    conf_curve.append(conf)
    stop, reason = policy.should_stop(conf_curve, k, report.contradiction_rate, report.verifiable)
    if stop: break
    k += cfg["step_k"]
    if cfg["reformulate_query"]:
        query = reformulate_query(original_query, [(d.diagnosis, d.probability) for d in reasoning.differential])
result = AEBResult(..., stopped_by_threshold=(reason == "threshold"), budget_exhausted=(reason == "budget_exhausted"),
                   stop_reason=reason, conf_drop_rate=drop_rate(conf_curve))
```
Add `stop_reason` and `conf_drop_rate` to `AEBResult`. Merge `configs/hardening.yaml` into `seeds.yaml`.
Keep `initial_k + n*step_k` landing exactly on `max_k` (3,6,9,12,15 works).
Check: `pytest tests -q -k stopping`

## Phase 3: Confidence calibration (fixes: hand-set weights, wrong temperature scaling, redundant signals)
1. Extend your experiment runner to write one JSONL record per question at the AEB STOP step:
```json
{"id":"q1","signals":{"retriever_top1":0.41,"evidence_sufficiency":0.7,"support_agreement":0.5,
 "raw_confidence":0.8,"hallucination_inverse":0.9},"correct":1}
```
   Use `hallucination_inverse = 1 - hallucination_score` (uncapped; the learned weight replaces your cap) and optionally
   add `contradiction_rate`. Compute `retriever_top1` from raw dense cosine similarity, not RRF scores.
2. Run:
```
python scripts/run_calibration_study.py --records runs/records.jsonl --out reports/calibration.json --save-model models/fusion.joblib
```
3. Read `reports/calibration.json`:
   - `signal_correlations`: if two signals correlate above 0.8 (likely `support_agreement` and `raw_confidence`), drop one.
   - `per_signal_auroc_test`: signals near 0.5 carry no information; remove them.
   - `test`: compare hand_weights vs learned_logistic vs learned_logistic+isotonic on ECE (with 95% CI), Brier, AUROC.
     Report all three in the paper. Use the best one.
4. In `src/confidence/estimation.py`, replace the fixed weighted sum with
   `joblib.load("models/fusion.joblib").predict(X)`. Remove the temperature scaling on fused scores (isotonic replaces it).
   Keep the old fixed-weight path behind a flag for the ablation table.
5. Splits: the script separates fit / calibration / test. Never reuse test questions to choose weights, thresholds or caps.
6. Reliability diagrams: plot `reliability` from the JSON (bin mean_conf vs accuracy) with matplotlib; include them in the paper.

## Phase 4: Experiments for the paper

### 4.1 Baselines (add to `src/experiments/runner.py`)
LLM-only (no retrieval); Vanilla top-5; Hybrid top-10; fixed-k matched to AEB average k; fixed-k matched to AEB average
LLM calls and tokens; one published adaptive baseline (Self-RAG, FLARE or Adaptive-RAG; a faithful re-implementation of the
policy is acceptable if stated); AEB full; AEB ablations (no patience, no verification gate, no reformulation, fixed weights).
Log per item: correctness, final confidence, stop_reason, k per round, LLM calls, tokens, latency, verifier name, KB hash.

### 4.2 Statistics (use `src/experiments/stats.py`)
```python
from src.experiments.stats import paired_bootstrap, mcnemar_exact, risk_coverage, triage_safety
paired_bootstrap(correct_aeb, correct_baseline)   # diff, 95% CI, p
mcnemar_exact(correct_aeb, correct_baseline)
risk_coverage(conf, correct)["aurc"]              # selective prediction
```
Run 3+ seeds with LLM temperature 0, same question set per method, report mean and CI. Report the confidence at the stop point
(not the average over all rounds) and `conf_drop_rate`.

### 4.3 Verifier validation
Randomly sample 150-200 claims from outputs. Have two annotators label supported / neutral / contradicted (report agreement).
Compute precision and recall of each verifier against the labels and tune `entail_thr` and `contra_thr` on half of them.
Evaluate hallucination with an independent method (a different NLI model or LLM judge with a fixed rubric), not only the verifier
used inside confidence.

### 4.4 Retrieval ablations
Replace the boosting code in `src/retrieval/hybrid.py` with `boost_and_rerank` (bounded, normalized). Run boosting on/off and
log `rank_displacement(before_ids, after_ids)`. Report per-subgroup accuracy (age bands, sex, comorbidity). Store embedder
name+version in the index metadata; at query time raise an error on mismatch and never fall back to HashEmbedder outside tests.

### 4.5 Triage evaluation
Use DDXPlus (state that it is synthetic) with its severity labels mapped to the four acuity levels (document the mapping).
```python
triage_safety(y_true_acuity, y_pred_acuity)
```
Report under-triage rate, severe under-triage rate, emergency sensitivity and specificity, and that no HOME_CARE output contains
a can't-miss diagnosis above threshold.

## Phase 5: Engineering cleanup
1. **Factory methods:** keep `from_unified_kb()`. Make `from_seed()` test-only. In experiment mode, raise if the unified KB is
   missing instead of falling back to SEED_CORPUS. Log KB file hash, chunk count, embedder, verifier and config hash per run:
```python
import hashlib, json
config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
```
2. **Hardware table (`detect_hardware`):** >=20 GB -> `kaggle_fp16`; 10-19.9 GB -> `kaggle_fp16` on Kaggle else `local_4bit`;
   <10 GB -> `local_4bit`; no GPU -> `hf_inference_api` only if `external_inference_allowed`, otherwise `mock` (tests) or raise.
3. **LLM output handling:** in `reasoner.py` wrap parsing in a retry (max 2) with a JSON-repair step; on final failure return
   an `invalid` result that counts as WRONG and is logged. Never drop it or count it correct. Add per-call timeouts.
4. **Reproducibility:** `pip freeze > requirements.lock`, pin HF model revisions (`revision="<commit sha>"`), pin embedder
   versions, LLM temperature 0, save config hash with results.
5. **Knowledge base governance:** every chunk gets `source, publication_date, version, region, license, status, supersedes`.
   Prefer the newest version when guidelines conflict; document licenses (NICE and similar may restrict reuse).
6. **Frontend (`frontend/`):**
   - Show a qualitative label ("High / Moderate / Low certainty") to patients, not a numeric percentage, until Phase 3 shows
     acceptable ECE. Keep the number in a clinician/research view.
   - Always append safety-net text to HOME_CARE: "Seek care urgently if symptoms worsen, or if you develop chest pain,
     trouble breathing, confusion, or severe pain."
   - Zod ranges:
```ts
age: z.number().min(0).max(120), sbp: z.number().min(40).max(300), hr: z.number().min(20).max(250),
spo2: z.number().min(50).max(100), temperature_c: z.number().min(30).max(45), rr: z.number().min(4).max(80)
```
7. **CI:** run `pytest`, the contamination check, and a small nightly job with real DeBERTa on 20 items.
8. **Tests to add in your repo:** triage boundaries (done), RRF correctness, malformed LLM output, API auth/role tests with
   FastAPI `TestClient`, and `accuracy=1.0` on seed queries kept only as a plumbing test (label it so).

## Phase 6: Paper framing
- Contribution 1 is AEB (calibrated, verification-gated adaptive stopping with query reformulation). Present hybrid retrieval,
  NLI verification and triage as system components, not separate contributions.
- Compare against a published adaptive-RAG method and a compute-matched fixed-k baseline.
- Include a Limitations/Ethics section: research prototype, not a medical device, no clinical validation, synthetic benchmarks,
  thresholds not clinically validated, exam-style benchmarks do not measure triage quality.

## Final acceptance checklist
- [ ] `pytest tests -q` all pass; contamination check exit code 0
- [ ] Triage: no HOME_CARE for any can't-miss diagnosis; clinician reviewed thresholds
- [ ] Verification uses DeBERTa for headline results; no silent fallbacks anywhere
- [ ] Fusion weights learned on a fit split; ECE with 95% CI reported on a held-out test split
- [ ] AEB beats compute-matched fixed-k (paired bootstrap p < 0.05) or you report that honestly
- [ ] Under-triage and emergency sensitivity reported on DDXPlus
- [ ] Upload endpoint admin-only; external inference off by default; logs redacted; secrets git-ignored
- [ ] Config hash, KB hash, model revisions saved with every result
