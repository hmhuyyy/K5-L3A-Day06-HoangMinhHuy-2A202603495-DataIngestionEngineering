"""Fixed-size and structure-aware chunking strategies."""
import re
from .models import Chunk, DocumentPage


def fixed_size(page: DocumentPage, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split text into overlapping character windows; retain page metadata."""
    text = page.text
    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                chunk_id=f"{page.document_id}_p{page.page}_fixed_{idx}",
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=chunk_text,
                metadata={**page.metadata, "chunk_strategy": "fixed", "chunk_index": idx},
            ))
            idx += 1
        if end >= len(text):
            break
        start = end - overlap

    return chunks


def by_paragraph(page: DocumentPage, max_size: int = 800) -> list[Chunk]:
    """Group paragraphs up to max_size; split oversized paragraphs at sentences."""
    raw_paras = re.split(r"\n\n+", page.text)
    paragraphs = [p.strip() for p in raw_paras if p.strip()]

    chunks: list[Chunk] = []
    idx = 0
    buf = ""

    def flush(text: str) -> None:
        nonlocal idx
        t = text.strip()
        if not t:
            return
        chunks.append(Chunk(
            chunk_id=f"{page.document_id}_p{page.page}_para_{idx}",
            document_id=page.document_id,
            source=page.source,
            page=page.page,
            text=t,
            metadata={**page.metadata, "chunk_strategy": "paragraph", "chunk_index": idx},
        ))
        idx += 1

    for para in paragraphs:
        if len(para) > max_size:
            # Flush current buffer first
            flush(buf)
            buf = ""
            # Split long paragraph at sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sent_buf = ""
            for sent in sentences:
                if sent_buf and len(sent_buf) + 1 + len(sent) > max_size:
                    flush(sent_buf)
                    sent_buf = sent
                else:
                    sent_buf = (sent_buf + " " + sent).strip() if sent_buf else sent
            buf = sent_buf
        else:
            # Try merging with current buffer
            if buf and len(buf) + 2 + len(para) <= max_size:
                buf += "\n\n" + para
            else:
                flush(buf)
                buf = para

    flush(buf)
    return chunks
