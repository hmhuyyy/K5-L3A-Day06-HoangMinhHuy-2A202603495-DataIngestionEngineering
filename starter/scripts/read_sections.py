"""Helper: print selected sections from corpus cache."""
import json, sys
from pathlib import Path

cache = Path(__file__).resolve().parents[1] / "data" / "corpus_cache.json"
with open(cache, encoding="utf-8") as f:
    items = json.load(f)

targets = [
    ("retrieval-augmented_generation", 0),
    ("retrieval-augmented_generation", 1),
    ("bert_language_model", 0),
    ("bert_language_model", 5),
    ("bert_language_model", 6),
    ("transformer_deep_learning", 0),
    ("transformer_deep_learning", 15),
    ("transformer_deep_learning", 19),
    ("transformer_deep_learning", 21),
    ("large_language_model", 1),
    ("large_language_model", 35),
    ("vector_database", 0),
    ("vector_database", 3),
    ("natural_language_processing", 14),
    ("word_embedding", 1),
    ("information_retrieval", 2),
    ("attention_machine_learning", 0),
]

by_key = {(it["document_id"], it["page"]): it for it in items}

for doc_id, page in targets:
    it = by_key.get((doc_id, page))
    if not it:
        print(f"\n[MISSING] {doc_id} p{page}")
        continue
    sec = it["metadata"]["section_title"]
    print(f"\n{'='*60}")
    print(f"[{doc_id}] p{page}: {sec} ({it['char_count']} chars)")
    print(it["text"][:700])
