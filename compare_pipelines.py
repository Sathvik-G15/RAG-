import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from src.confidence.pipeline import Pipeline
from src.confidence.chroma_pipeline import ChromaPipeline
from src.data.seed_corpus import SEED_PATIENT_QUERIES

print('=== ORIGINAL PIPELINE (In-Memory) ===')
pipeline1 = Pipeline.from_seed()
for q in SEED_PATIENT_QUERIES:
    aeb, resp = pipeline1.analyze(q['query'])
    print(f'  {q["expected_dx"]}: dx={resp.primary_diagnosis} conf={resp.confidence:.3f} dec={resp.decision.value}')

print()
print('=== CHROMA PIPELINE ===')
pipeline2 = ChromaPipeline.from_chroma()
for q in SEED_PATIENT_QUERIES:
    aeb, resp = pipeline2.analyze(q['query'])
    print(f'  {q["expected_dx"]}: dx={resp.primary_diagnosis} conf={resp.confidence:.3f} dec={resp.decision.value}')