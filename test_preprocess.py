import json
from elastic_search import build_documents

with open("si_dataset/review_for_analysis.json", encoding="utf-8") as f:
    records = json.load(f)[:3]

docs = build_documents(records)

for i, doc in enumerate(docs):
    print(f"\n=== Document {i+1} ===")
    print(json.dumps(doc, indent=2, ensure_ascii=False))