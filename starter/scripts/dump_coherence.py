"""Extract actual chunks at 10 positions across all 3 strategies for coherence audit."""
import json
from pathlib import Path

results = Path(__file__).resolve().parents[1] / "results"
data = Path(__file__).resolve().parents[1] / "data"

with open(results / "traces.json", encoding="utf-8") as f:
    traces = json.load(f)

# We need to reconstruct chunks. Since traces only have retrieval results,
# let's re-chunk to see what the chunks actually look like at each position.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion import load_all_articles
from src.chunking import fixed_size, by_paragraph
from src.models import DocumentPage

pages = load_all_articles(data)

positions = [
    ("retrieval-augmented_generation", 0),
    ("bert_language_model", 5),
    ("transformer_deep_learning", 0),
    ("transformer_deep_learning", 19),
    ("natural_language_processing", 14),
    ("word_embedding", 1),
    ("large_language_model", 1),
    ("vector_database", 0),
    ("information_retrieval", 2),
    ("attention_machine_learning", 0),
]

by_key = {(p.document_id, p.page): p for p in pages}

for i, (doc_id, page) in enumerate(positions, 1):
    p = by_key.get((doc_id, page))
    if not p:
        continue
    sec = p.metadata["section_title"]
    print(f"\n{'='*70}")
    print(f"POSITION {i}: {doc_id} p{page} [{sec}] ({len(p.text)} chars)")
    print(f"{'='*70}")

    # Strategy A: fixed 500/50
    chunks_a = fixed_size(p, 500, 50)
    print(f"\n--- Strategy A: {len(chunks_a)} chunks ---")
    for c in chunks_a:
        print(f"  [{c.chunk_id}] ({len(c.text)} chars)")
        print(f"  START: {c.text[:120]}")
        print(f"  END:   ...{c.text[-80:]}")
        print()

    # Strategy B: paragraph 800
    chunks_b = by_paragraph(p, 800)
    print(f"\n--- Strategy B/C: {len(chunks_b)} chunks ---")
    for c in chunks_b:
        print(f"  [{c.chunk_id}] ({len(c.text)} chars)")
        print(f"  START: {c.text[:120]}")
        print(f"  END:   ...{c.text[-80:]}")
        print()
