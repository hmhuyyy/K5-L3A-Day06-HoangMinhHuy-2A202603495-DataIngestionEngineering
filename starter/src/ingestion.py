"""Read and normalize source data from Wikipedia via REST API."""
import json
import re
import unicodedata
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime, timezone
from .models import DocumentPage


# 10 Wikipedia articles: 4 retrieval, 4 model, 2 language_processing
ARTICLES = [
    {"title": "Retrieval-augmented generation", "category": "retrieval"},
    {"title": "Vector database", "category": "retrieval"},
    {"title": "Semantic search", "category": "retrieval"},
    {"title": "Information retrieval", "category": "retrieval"},
    {"title": "Transformer (deep learning)", "category": "model"},
    {"title": "BERT (language model)", "category": "model"},
    {"title": "Attention (machine learning)", "category": "model"},
    {"title": "Large language model", "category": "model"},
    {"title": "Natural language processing", "category": "language_processing"},
    {"title": "Word embedding", "category": "language_processing"},
]

SKIP_SECTIONS = {
    "see also", "references", "external links", "further reading",
    "notes", "citations", "bibliography", "footnotes", "gallery",
}


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without discarding useful structure."""
    text = unicodedata.normalize("NFC", text)
    # Collapse repeated spaces (but keep newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _wiki_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))


def fetch_wikipedia_article(title: str, category: str) -> list[DocumentPage]:
    """Fetch a Wikipedia article via API and split into sections."""
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    api_url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=query&titles={encoded}&prop=extracts"
        "&explaintext=true&exsectionformat=wiki&format=json"
    )
    req = urllib.request.Request(api_url, headers={"User-Agent": "DataIngestionLab/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    pages_data = data["query"]["pages"]
    page_obj = next(iter(pages_data.values()))
    if "extract" not in page_obj:
        return []

    raw = page_obj["extract"]
    doc_id = (
        title.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("__", "_")
        .strip("_")
    )
    source_url = _wiki_url(title)
    return _split_sections(raw, doc_id, title, category, source_url)


def _split_sections(
    text: str, doc_id: str, title: str, category: str, source_url: str
) -> list[DocumentPage]:
    """Split raw Wikipedia text into section DocumentPage objects."""
    pattern = re.compile(r"^(={2,4})\s*(.+?)\s*\1\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))

    result: list[DocumentPage] = []
    idx = 0

    # Intro section (before first heading)
    intro_end = matches[0].start() if matches else len(text)
    intro_text = normalize_text(text[:intro_end])
    if len(intro_text) >= 150:
        result.append(DocumentPage(
            document_id=doc_id,
            source=source_url,
            page=idx,
            text=intro_text,
            metadata={
                "section_title": "Introduction",
                "article_title": title,
                "topic_category": category,
                "section_level": 0,
            },
        ))
        idx += 1

    for i, match in enumerate(matches):
        sec_title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sec_text = normalize_text(text[start:end])

        if len(sec_text) < 150:
            continue
        if sec_title.lower() in SKIP_SECTIONS:
            continue

        result.append(DocumentPage(
            document_id=doc_id,
            source=source_url,
            page=idx,
            text=sec_text,
            metadata={
                "section_title": sec_title,
                "article_title": title,
                "topic_category": category,
                "section_level": len(match.group(1)) - 2,
            },
        ))
        idx += 1

    return result


def load_all_articles(data_dir: Path) -> list[DocumentPage]:
    """Load from cache if available, otherwise fetch from Wikipedia."""
    cache_file = data_dir / "corpus_cache.json"

    if cache_file.exists():
        print(f"[ingestion] Loading corpus from cache: {cache_file}")
        with open(cache_file, encoding="utf-8") as f:
            raw = json.load(f)
        return [
            DocumentPage(
                document_id=item["document_id"],
                source=item["source"],
                page=item["page"],
                text=item["text"],
                metadata=item["metadata"],
            )
            for item in raw
        ]

    print("[ingestion] Fetching articles from Wikipedia ...")
    all_pages: list[DocumentPage] = []
    errors = []

    for art in ARTICLES:
        try:
            pages = fetch_wikipedia_article(art["title"], art["category"])
            print(f"  [{art['title']}] -> {len(pages)} sections")
            all_pages.extend(pages)
        except Exception as exc:
            errors.append({"title": art["title"], "error": str(exc)})
            print(f"  [{art['title']}] ERROR: {exc}")

    if errors:
        err_file = data_dir / "fetch_errors.json"
        with open(err_file, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"[ingestion] {len(errors)} errors -> {err_file}")

    cache = [
        {
            "document_id": p.document_id,
            "source": p.source,
            "page": p.page,
            "text": p.text,
            "metadata": p.metadata,
            "char_count": len(p.text),
        }
        for p in all_pages
    ]
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"[ingestion] Saved {len(all_pages)} sections to cache.")
    return all_pages


def build_manifest(pages: list[DocumentPage], data_dir: Path) -> dict:
    """Build and save the data manifest."""
    from collections import defaultdict

    by_doc: dict[str, list[DocumentPage]] = defaultdict(list)
    for p in pages:
        by_doc[p.document_id].append(p)

    sources = []
    total_chars = 0
    for doc_id, doc_pages in by_doc.items():
        chars = sum(len(p.text) for p in doc_pages)
        total_chars += chars
        sources.append({
            "source_id": doc_id,
            "title": doc_pages[0].metadata["article_title"],
            "source_path_or_url": doc_pages[0].source,
            "license_or_permission": "CC BY-SA 4.0 — https://creativecommons.org/licenses/by-sa/4.0/",
            "accessed_at": datetime.now(timezone.utc).isoformat(),
            "section_ids": list(range(len(doc_pages))),
            "section_count": len(doc_pages),
            "total_chars": chars,
            "topic_category": doc_pages[0].metadata["topic_category"],
        })

    manifest = {
        "corpus_description": (
            "10 Wikipedia articles on AI/ML topics: "
            "RAG, vector databases, semantic search, information retrieval, "
            "Transformer, BERT, attention, LLM, NLP, word embedding."
        ),
        "content_unit": "Wikipedia article section (each section = one DocumentPage)",
        "total_units": len(pages),
        "total_chars": total_chars,
        "note_on_pages": (
            "Wikipedia has no physical page numbers. "
            "Each 'page' is a sequential section index within an article (0 = intro). "
            "30-unit threshold: counted by sections with >=1500 chars."
        ),
        "filter_field": "topic_category",
        "filter_values": ["retrieval", "model", "language_processing"],
        "sources": sources,
    }

    manifest_file = data_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[ingestion] Manifest -> {manifest_file}")
    return manifest
