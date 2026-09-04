import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
from datasets import load_dataset

print("Testing EPFL-LLM guidelines load - checking more sources...")
ds = load_dataset("epfl-llm/guidelines", split="train", streaming=True)
sources = set()
count = 0
for i, item in enumerate(ds):
    if i >= 500:
        break
    sources.add(item.get('source', 'unknown'))
    count += 1

print(f"Total items checked: {count}")
print(f"Sources found: {sources}")

# Check if there are other splits
print("\nChecking dataset info...")
ds_info = load_dataset("epfl-llm/guidelines")
print(f"Dataset splits: {list(ds_info.keys())}")
for split_name, split_data in ds_info.items():
    print(f"  {split_name}: {split_data.num_rows} rows")