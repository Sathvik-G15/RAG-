from src.data.guidelines_loader import stream_guidelines, chunk_document

doc = {
    "text": "Diagnosis of community acquired pneumonia requires clinical signs and chest radiograph. First-line treatment is oral amoxicillin.",
    "title": "CAP Guideline",
    "source": "WHO",
    "specialty": "infectious_disease",
    "year": "2023",
    "url": "http://who.int/cap",
}
chunks = chunk_document(doc, chunk_size=20, chunk_overlap=5)
print(f"Chunking test: {len(chunks)} chunks generated")
print(f"First chunk: {chunks[0].text[:80]}...")
spec = chunks[0].metadata["specialty"]
print(f"Specialty: {spec}")

print()
print("Testing stream_guidelines (first 2 items)...")
count = 0
for item in stream_guidelines(domain="infectious_disease", limit=2):
    print(f"  Item {count}: {item['title'][:60]}... (specialty: {item['specialty']})")
    count += 1

print("Guidelines loader test passed!")