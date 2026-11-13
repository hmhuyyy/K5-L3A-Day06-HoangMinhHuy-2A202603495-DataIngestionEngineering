"""Analyze trace results and print misses / hits."""
import json
from pathlib import Path

traces_file = Path(__file__).resolve().parents[1] / "results" / "traces.json"
with open(traces_file, encoding="utf-8") as f:
    traces = json.load(f)

for strategy in ["A", "B", "C"]:
    strat_traces = [t for t in traces if t["strategy"] == strategy]
    hits1 = sum(1 for t in strat_traces if t["hit_at_1"])
    hits3 = sum(1 for t in strat_traces if t["hit_at_3"])
    n = len(strat_traces)
    print(f"\n=== Strategy {strategy}: {hits1}/{n} Hit@1, {hits3}/{n} Hit@3 ===")

    misses = [t for t in strat_traces if not t["hit_at_3"]]
    print(f"  Misses ({len(misses)}):")
    for t in misses:
        print(f"  [{t['question_id']} split={t['split']}] {t['question'][:65]}")
        print(f"    gold: {t['gold_source_ids']} p{t['gold_pages']}")
        for h in t["hits"]:
            doc = h["metadata"].get("document_id", "?")
            pg = h["metadata"].get("page", "?")
            sc = h["score"]
            txt = h["text"][:90].replace("\n", " ")
            print(f"    r{h['rank']}: {doc} p{pg} sc={sc:.3f} hit={h['is_hit']} | {txt}")
        print()
