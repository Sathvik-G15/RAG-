import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
from datasets import load_dataset

print("Testing EPFL-LLM guidelines load...")
ds = load_dataset("epfl-llm/guidelines", split="train", streaming=True)
for i, item in enumerate(ds):
    if i >= 3:
        break
    print(f"Item {i}: keys={list(item.keys())}")
    print(f"  text preview: {item.get('text', '')[:200]}")
    print(f"  source: {item.get('source')}")
    print(f"  title: {item.get('title')}")
    print()