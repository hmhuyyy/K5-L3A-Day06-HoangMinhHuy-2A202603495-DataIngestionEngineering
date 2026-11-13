"""Dump trace data needed for audit files."""
import json
from pathlib import Path

results = Path(__file__).resolve().parents[1] / "results"
data = Path(__file__).resolve().parents[1] / "data"

with open(results / "traces.json", encoding="utf-8") as f:
    traces = json.load(f)

with open(data / "corpus_cache.json", encoding="utf-8") as f:
    corpus = json.load(f)

# 1. Print test question traces for all strategies (grounding audit)
print("=" * 70)
print("TEST QUESTION TRACES (for grounding audit)")
print("=" * 70)
for qid in ["q16", "q17", "q18", "q19", "q20"]:
    for strat in ["A", "B", "C"]:
        t = [x for x in traces if x["question_id"] == qid and x["strategy"] == strat]
        if not t:
            continue
        t = t[0]
        print(f"\n--- {qid} | Strategy {strat} | hit@3={t['hit_at_3']} ---")
        print(f"  Q: {t['question']}")
        print(f"  Gold: {t['gold_source_ids']} p{t['gold_pages']}")
        for h in t["hits"]:
            m = h["metadata"]
            print(f"  r{h['rank']}: {m.get('document_id','?')} p{m.get('page','?')} "
                  f"cat={m.get('topic_category','?')} sc={h['score']:.3f} hit={h['is_hit']}")
            print(f"      {h['text'][:250]}")

# 2. Print 10 source positions for coherence audit
# Pick 10 positions spread across different articles
print("\n\n" + "=" * 70)
print("10 SOURCE POSITIONS (for coherence audit)")
print("=" * 70)

positions = [
    ("retrieval-augmented_generation", 0),   # RAG intro
    ("bert_language_model", 5),              # BERT MLM
    ("transformer_deep_learning", 0),         # Transformer intro
    ("transformer_deep_learning", 19),        # Multihead attention
    ("natural_language_processing", 14),      # Higher-level NLP apps
    ("word_embedding", 1),                     # Word embedding history
    ("large_language_model", 1),              # LLM history
    ("vector_database", 0),                  # Vector DB intro
    ("information_retrieval", 2),             # IR history
    ("attention_machine_learning", 0),        # Attention intro
]

by_key = {(it["document_id"], it["page"]): it for it in corpus}

for i, (doc_id, page) in enumerate(positions, 1):
    item = by_key.get((doc_id, page))
    if not item:
        continue
    sec = item["metadata"]["section_title"]
    cat = item["metadata"]["topic_category"]
    text = item["text"]
    print(f"\n{'='*60}")
    print(f"POSITION {i}: {doc_id} p{page} [{sec}] cat={cat} ({len(text)} chars)")
    print(f"FIRST 200 chars: {text[:200]}")
    print(f"LAST 100 chars: ...{text[-100:]}")
    # Count chunks for each strategy
    # Fixed: ceil(len/450) roughly (500-50 overlap = 450 step)
    # Paragraph: depends on paragraph structure
    step = 450
    n_fixed = max(1, (len(text) - 500 + step) // step + 1) if len(text) > 500 else 1
    print(f"  Est. fixed chunks: ~{n_fixed}")
