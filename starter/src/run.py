"""Day 06 Data Ingestion Engineering — full pipeline orchestrator.

Usage:
    cd starter
    python -m src.run            # run full pipeline (ingest + embed + eval)
    python -m src.run --fetch     # fetch corpus only, print section summary
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

# starter/
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CHROMA_DIR = ROOT / "results" / "chroma_db"
BENCHMARK_FILE = ROOT.parent / "benchmark" / "questions.json"
ARTIFACTS_DIR = ROOT.parent / "artifacts"


def _print_sections(pages):
    """Print a summary of fetched sections for benchmark authoring."""
    from collections import defaultdict
    by_doc = defaultdict(list)
    for p in pages:
        by_doc[p.document_id].append(p)
    print("\n=== CORPUS SECTION SUMMARY ===")
    total_chars = 0
    for doc_id, ps in by_doc.items():
        chars = sum(len(p.text) for p in ps)
        total_chars += chars
        print(f"\n[{doc_id}] ({len(ps)} sections, {chars} chars)")
        for p in ps:
            sec = p.metadata.get("section_title", "?")
            print(f"  p{p.page}: {sec} ({len(p.text)} chars)")
    valid = [p for p in pages if len(p.text) >= 1500]
    print(f"\nTotal sections: {len(pages)} | >=1500 chars: {len(valid)} | total chars: {total_chars}")


def main() -> None:
    fetch_only = "--fetch" in sys.argv

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    from .ingestion import load_all_articles, build_manifest
    from .chunking import fixed_size, by_paragraph
    from .embeddings import FastEmbedEmbedder
    from .store import QdrantVectorStore
    from .evaluation import evaluate_query, summarize

    # ── 1. Corpus ──────────────────────────────────────────────────────
    print("=" * 64)
    print("PHASE 1 · Loading corpus")
    pages = load_all_articles(DATA_DIR)
    manifest = build_manifest(pages, DATA_DIR)

    valid = [p for p in pages if len(p.text) >= 1500]
    print(f"Sections: {len(pages)} | valid (>=1500 chars): {len(valid)}")

    if fetch_only:
        _print_sections(pages)
        print("\n[--fetch] Done. Review sections, write benchmark/questions.json, then run again.")
        return

    # ── 2. Benchmark ───────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("PHASE 2 · Loading benchmark")
    if not BENCHMARK_FILE.exists():
        sys.exit(f"Benchmark not found: {BENCHMARK_FILE}\nRun with --fetch first.")
    with open(BENCHMARK_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    dev_q = [q for q in questions if q["split"] == "dev"]
    test_q = [q for q in questions if q["split"] == "test"]
    print(f"Questions: {len(questions)} ({len(dev_q)} dev, {len(test_q)} test)")

    # ── 3. Embedder + store ────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("PHASE 3 · Initialising embedder & store")
    embedder = FastEmbedEmbedder()
    store = QdrantVectorStore(str(CHROMA_DIR), embedder.dimensions)

    # ── 4. Three strategies ────────────────────────────────────────────
    strategies = {
        "A": {"label": "fixed_500_50",      "chunk": lambda p: fixed_size(p, 500, 50),  "filter": False},
        "B": {"label": "paragraph_800",      "chunk": lambda p: by_paragraph(p, 800),   "filter": False},
        "C": {"label": "paragraph_800_filt", "chunk": lambda p: by_paragraph(p, 800),   "filter": True},
    }

    all_traces: list[dict] = []
    stats: dict[str, dict] = {}

    for skey, cfg in strategies.items():
        print(f"\n{'=' * 64}")
        print(f"STRATEGY {skey} · {cfg['label']}")
        t0 = time.time()

        # Chunk all pages
        chunks = []
        for page in pages:
            chunks.extend(cfg["chunk"](page))
        lengths = [len(c.text) for c in chunks]
        print(f"  chunks={len(chunks)}  len min/avg/max = {min(lengths)}/{sum(lengths)//len(lengths)}/{max(lengths)}")

        # Embed
        print(f"  embedding {len(chunks)} chunks ...")
        et0 = time.time()
        vectors = embedder.embed_documents([c.text for c in chunks])
        embed_t = time.time() - et0
        print(f"  embed {embed_t:.1f}s")

        # Store
        store.create_schema(skey)
        store.upsert(skey, chunks, vectors)

        ingest_t = time.time() - t0

        # Evaluate
        traces = []
        for q in questions:
            filt = q.get("filter") if cfg["filter"] else None
            qvec = embedder.embed_query(q["question"])
            hits = store.search(skey, qvec, top_k=3, filters=filt)
            r = evaluate_query(q, hits)
            r["strategy"] = skey
            traces.append(r)
        all_traces.extend(traces)

        summ = summarize(traces)
        stats[skey] = {
            "label": cfg["label"],
            "chunk_count": len(chunks),
            "avg_chunk_len": sum(lengths) // len(lengths),
            "ingest_time_s": round(ingest_t, 1),
            "embed_time_s": round(embed_t, 1),
            "results": summ.get(skey, {}),
        }
        for split, m in summ.get(skey, {}).items():
            print(f"  {split}: Hit@1={m['hit_at_1']}/{m['n']} ({m['hit_at_1_rate']:.0%})  "
                  f"Hit@3={m['hit_at_3']}/{m['n']} ({m['hit_at_3_rate']:.0%})")

    # ── 5. Save results ───────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("PHASE 5 · Saving results")
    with open(RESULTS_DIR / "traces.json", "w", encoding="utf-8") as f:
        json.dump(all_traces, f, ensure_ascii=False, indent=2)
    print(f"  traces -> {RESULTS_DIR / 'traces.json'}")

    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  summary -> {RESULTS_DIR / 'summary.json'}")

    _update_experiment_log(stats)

    # Final table
    print("\n" + "=" * 64)
    print("FINAL RESULTS")
    print(f"{'Strat':<6} {'Split':<6} {'Hit@1':<14} {'Hit@3':<14} {'Chunks':<8} {'Time'}")
    for sk, d in stats.items():
        for split, m in d.get("results", {}).items():
            print(f"{sk:<6} {split:<6} "
                  f"{m['hit_at_1']}/{m['n']} ({m['hit_at_1_rate']:.0%})  "
                  f"{m['hit_at_3']}/{m['n']} ({m['hit_at_3_rate']:.0%})  "
                  f"{d['chunk_count']:<8} {d['ingest_time_s']:.0f}s")


def _update_experiment_log(stats: dict) -> None:
    """Write artifacts/experiment_log.csv with real run results."""
    log_file = ARTIFACTS_DIR / "experiment_log.csv"
    rows = []
    for sk, d in stats.items():
        is_fixed = "fixed" in d["label"]
        dev = d["results"].get("dev", {})
        test = d["results"].get("test", {})
        rows.append({
            "strategy": sk,
            "chunking": "fixed_size" if is_fixed else "paragraph",
            "chunk_size": 500 if is_fixed else 800,
            "overlap": 50 if is_fixed else 0,
            "metadata": "document_id,source,page,section_title,article_title,topic_category",
            "filter": "topic_category" if "filt" in d["label"] else "none",
            "vectorizer": "fastembed/BAAI/bge-small-en-v1.5 (384d)",
            "hypothesis": {
                "A": "Fixed-size baseline; simplest chunking",
                "B": "Paragraph chunking preserves semantic coherence vs fixed-size",
                "C": "Metadata filter reduces noise for category-scoped questions",
            }[sk],
            "dev_hit1": f"{dev.get('hit_at_1', 0)}/{dev.get('n', 0)}",
            "dev_hit3": f"{dev.get('hit_at_3', 0)}/{dev.get('n', 0)}",
            "test_hit1": f"{test.get('hit_at_1', 0)}/{test.get('n', 0)}",
            "test_hit3": f"{test.get('hit_at_3', 0)}/{test.get('n', 0)}",
            "notes": f"ingest={d['ingest_time_s']}s embed={d['embed_time_s']}s chunks={d['chunk_count']} avg_len={d['avg_chunk_len']}",
        })

    with open(log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "strategy", "chunking", "chunk_size", "overlap", "metadata",
            "filter", "vectorizer", "hypothesis",
            "dev_hit1", "dev_hit3", "test_hit1", "test_hit3", "notes",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  experiment log -> {log_file}")


if __name__ == "__main__":
    main()
