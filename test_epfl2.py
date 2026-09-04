import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
from datasets import load_dataset

print("Testing EPFL-LLM guidelines load...")
ds = load_dataset("epfl-llm/guidelines", split="train", streaming=True)
sources = set()
for i, item in enumerate(ds):
    if i >= 50:
        break
    sources.add(item.get('source', 'unknown'))
    if i < 5:
        print(f"Item {i}: keys={list(item.keys())}")
        print(f"  source: {item.get('source')}")
        print(f"  title: {item.get('title')}")
        print(f"  clean_text len: {len(item.get('clean_text', ''))}")
        print(f"  raw_text len: {len(item.get('raw_text', ''))}")
        print(f"  overview: {item.get('overview')}")
        print()

print(f"Sources found: {sources}")