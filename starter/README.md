# Starter · PostgreSQL + pgvector → Qdrant local (Day 06 Lab)

Bộ khởi động **đã được hiện thực đầy đủ** cho lab cá nhân. Thay thế pgvector bằng **Qdrant local mode** (không cần Docker) + **fastembed** (ONNX, không cần PyTorch/API key).

## Cấu trúc

```text
starter/
├── docker-compose.yml     # pgvector original — không dùng (Docker không có)
├── requirements.txt       # psycopg, pypdf, python-dotenv + chromadb, fastembed
├── .env.example           # cấu hình mẫu (chỉ cần DATABASE_URL nếu dùng pgvector)
├── scripts/
│   ├── check_setup.py     # kiểm tra pgvector (bỏ qua nếu dùng Qdrant)
│   ├── read_sections.py   # in nội dung sections để viết benchmark
│   ├── dump_audit_data.py # in traces cho audit
│   └── dump_coherence.py  # in chunks theo position cho coherence audit
├── src/
│   ├── models.py          # DocumentPage, Chunk dataclasses
│   ├── ingestion.py       # fetch Wikipedia, normalize, split sections
│   ├── chunking.py        # fixed_size(500/50) và by_paragraph(800)
│   ├── embeddings.py      # FastEmbedEmbedder — BAAI/bge-small-en-v1.5, 384d
│   ├── store.py           # QdrantVectorStore — local persistence, cosine
│   ├── evaluation.py      # Hit@1 / Hit@3 by source+page match
│   └── run.py             # orchestrator — 3 strategies A/B/C
├── data/
│   ├── corpus_cache.json  # 176 sections fetched từ Wikipedia
│   └── manifest.json      # nguồn, giấy phép, số sections, filter fields
└── results/
    ├── chroma_db/         # Qdrant local persistence
    ├── traces.json        # top-3 traces cho 20 câu × 3 strategies
    ├── summary.json       # Hit@1/Hit@3 aggregated
    ├── coherence_audit.json
    ├── metadata_audit.json
    └── grounding_audit.json
```

## Setup nhanh

Python **3.12** trên Windows, không cần Docker:

```bash
cd starter
python -m pip install qdrant-client fastembed==0.4.2 pypdf python-dotenv
```

## Chạy pipeline

```bash
# Bước 1: fetch corpus (lần đầu, cần internet)
python -m src.run --fetch

# Bước 2: chạy pipeline đầy đủ (ingest → embed → store → evaluate)
python -m src.run
```

Lần đầu chạy, fastembed sẽ tải model BAAI/bge-small-en-v1.5 (~24MB ONNX). Mô hình được cache tại `%TEMP%\fastembed_cache\`.

## Kết quả thực tế

| Strategy | Chunking | Chunks | Hit@1 dev | Hit@3 dev | Hit@1 test | Hit@3 test |
|---|---|---|---|---|---|---|
| A | Fixed 500/50 | 518 | 7/15 (47%) | 13/15 (87%) | 3/5 (60%) | 3/5 (60%) |
| B | Paragraph 800 | 370 | 4/15 (27%) | 12/15 (80%) | 3/5 (60%) | 3/5 (60%) |
| C | Paragraph 800 + filter | 370 | 4/15 (27%) | 12/15 (80%) | 3/5 (60%) | 3/5 (60%) |

**Chiến lược chọn:** A (Hit@3 dev cao nhất = 87%).

## Corpus

10 bài Wikipedia về AI/ML (CC BY-SA 4.0), truy cập 2026-11-13:
- Retrieval-augmented generation, Vector database, Semantic search, Information retrieval *(topic: retrieval)*
- Transformer, BERT, Attention, Large language model *(topic: model)*
- Natural language processing, Word embedding *(topic: language_processing)*

176 sections, 42 sections ≥ 1500 chars, tổng 206.344 chars.

## Embedding & Vector Store

- **Model:** `BAAI/bge-small-en-v1.5` (fastembed, ONNX local), 384 dims, cosine metric
- **Vector store:** Qdrant local mode (qdrant-client 1.20.0), persistent tại `results/chroma_db/`
- **Chi phí:** 0 (toàn bộ local, không gọi API)
- **Thời gian embed/strategy:** ~33–35s trên CPU

## Lưu ý

- Không commit `.env` hoặc API keys.
- Qdrant local store được recreate mỗi lần chạy `python -m src.run` (delete + create collection).
- Nếu muốn giữ index cũ, comment out `self._client.delete_collection(name)` trong `store.py`.
