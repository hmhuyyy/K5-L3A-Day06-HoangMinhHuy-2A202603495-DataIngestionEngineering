"""Evaluate retrieved evidence; compute Hit@1 and Evidence Hit@3.

A hit = chunk whose document_id matches a gold_source_id AND whose page matches
a gold_page (when specified).  Evidence content is assessed separately in the
manual grounding audit, not via exact-string matching — a chunk from the correct
section is a retrieval hit even if the specific evidence sentence landed in a
different chunk of the same section.
"""
from __future__ import annotations
from typing import Any


def evaluate_query(
    question: dict[str, Any],
    hits: list[dict[str, Any]],
) -> dict[str, Any]:
    gold_ids = question.get("gold_source_ids") or [question.get("gold_source_id", "")]
    gold_pages = question.get("gold_pages") or [question.get("gold_page")]
    gold_quotes = question.get("evidence_quotes") or [question.get("evidence_quote", "")]

    def _is_hit(hit: dict) -> bool:
        meta = hit.get("metadata", {})
        hit_doc = meta.get("document_id", "")
        hit_page = meta.get("page", "")

        for src, pg, _ev in zip(gold_ids, gold_pages, gold_quotes):
            doc_ok = hit_doc == src
            # page=None means any page in the correct document is accepted
            page_ok = (pg is None) or (str(pg) == str(hit_page))
            if doc_ok and page_ok:
                return True
        return False

    hit_flags = [_is_hit(h) for h in hits]

    return {
        "question_id": question["id"],
        "question": question["question"],
        "split": question.get("split", "dev"),
        "filter_used": question.get("filter"),
        "hit_at_1": hit_flags[0] if hit_flags else False,
        "hit_at_3": any(hit_flags[:3]),
        "hits": [
            {
                "rank": i + 1,
                "text": h["text"][:400] + ("…" if len(h["text"]) > 400 else ""),
                "score": h["score"],
                "metadata": h["metadata"],
                "is_hit": hit_flags[i],
            }
            for i, h in enumerate(hits)
        ],
        "gold_source_ids": gold_ids,
        "gold_pages": gold_pages,
        "evidence_quotes": gold_quotes,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 by strategy and dev/test split."""
    from collections import defaultdict

    groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for t in traces:
        groups[t.get("strategy", "?")][t.get("split", "dev")].append(t)

    summary: dict[str, Any] = {}
    for strategy, splits in groups.items():
        summary[strategy] = {}
        for split, records in splits.items():
            n = len(records)
            h1 = sum(1 for r in records if r["hit_at_1"])
            h3 = sum(1 for r in records if r["hit_at_3"])
            summary[strategy][split] = {
                "n": n,
                "hit_at_1": h1,
                "hit_at_3": h3,
                "hit_at_1_rate": round(h1 / n, 3) if n else 0.0,
                "hit_at_3_rate": round(h3 / n, 3) if n else 0.0,
            }
    return summary
