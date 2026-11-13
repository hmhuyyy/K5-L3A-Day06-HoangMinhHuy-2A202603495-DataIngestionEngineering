# Báo cáo cá nhân — Day 6

## Dữ liệu và pipeline

- **Chủ đề, nguồn/quyền, số trang và quy tắc tính trang:**
  Corpus gồm 10 bài Wikipedia về các chủ đề AI/ML: Retrieval-Augmented Generation, Vector database, Semantic search, Information retrieval, Transformer (deep learning), BERT (language model), Attention (machine learning), Large language model, Natural language processing, Word embedding. Tất cả đều thuộc giấy phép **CC BY-SA 4.0** (Wikipedia). Nguồn truy cập qua Wikipedia REST API ngày 2026-11-13. Không có số trang vật lý (Wikipedia là web). Mỗi "page" = một section của article (index bắt đầu từ 0 = intro). Đơn vị tối thiểu 1.500 ký tự: **42/176 sections đủ điều kiện**. Tổng: 206.344 ký tự, 176 sections từ 10 articles.

- **Chuẩn hóa, lỗi và giới hạn:**
  Text chuẩn hóa Unicode NFC, collapse khoảng trắng và newline thừa. Sections quá ngắn (<150 chars) và meta-sections (see also, references, external links) bị loại. Không có OCR (Wikipedia là plain text). Lỗi: 0 fetch errors. Giới hạn: Wikipedia content thay đổi theo thời gian — snapshot ngày 2026-11-13. Các section về mathematical formulas có ký tự Unicode đặc biệt (e.g., ∑, ∈, ℝ) được giữ nguyên.

- **Embedding model/provider/version/số chiều, vector store/version, metric, lệnh chạy và chi phí:**
  - **Embedding model:** `BAAI/bge-small-en-v1.5` qua thư viện `fastembed==0.4.2` (ONNX runtime, chạy local không cần API key). Số chiều: **384**. Provider: Qdrant/HuggingFace. Query prefix: `"Represent this sentence for searching relevant passages: "` (BGE asymmetric retrieval).
  - **Vector store:** `qdrant-client==1.20.0`, local mode (`QdrantClient(path=...)`), lưu bền vững trên đĩa tại `starter/results/chroma_db/`. Metric: **cosine similarity**. Approximate search (HNSW).
  - **Lệnh chạy:** `cd starter && python -m src.run`
  - **Chi phí:** 0 (toàn bộ local, không gọi API trả phí). Thời gian embed: ~33–35s/strategy trên CPU.

- **Metadata bắt buộc, trường lọc bổ sung, nguồn giá trị và lý do chọn:**
  Metadata bắt buộc mỗi chunk: `document_id`, `source` (URL Wikipedia), `page` (section index). Trường lọc bổ sung: **`topic_category`** với giá trị `"retrieval"`, `"model"`, `"language_processing"` — được gán trực tiếp từ danh sách `ARTICLES` trong `ingestion.py` (không tự đoán). Lý do chọn: 10 articles phân thành 3 nhóm chủ đề rõ ràng, cho phép filter câu hỏi theo domain. Các trường phụ: `section_title`, `article_title`, `section_level`, `chunk_strategy`, `chunk_index`.

---

## Cấu hình và benchmark

- **A / B / C (tham số cụ thể, khác biệt duy nhất B–C):**
  | Config | Chunking | chunk_size | overlap | Filter |
  |--------|----------|-----------|---------|--------|
  | **A** | Fixed-size | 500 chars | 50 chars | Không |
  | **B** | Paragraph (by_paragraph) | max 800 chars | 0 (paragraph boundary) | Không |
  | **C** | Paragraph (by_paragraph) | max 800 chars | 0 | **Có** (topic_category) |

  B và C giống hệt nhau về chunking và số chunks (370). Khác biệt duy nhất B→C: C áp dụng metadata filter `topic_category` cho các câu hỏi có trường `filter` trong benchmark. A–B khác về chiến lược chunking (fixed vs paragraph).

- **≥20 câu: phân chia ≥15 dev / ≥5 test, câu metadata filter và cách kiểm chứng evidence:**
  20 câu hỏi tự xây dựng từ corpus (file `benchmark/questions.json`), chia **15 dev / 5 test** trước khi chạy thử nghiệm. 4 câu có filter: q14 (dev, `retrieval`), q16 (test, `model`), q17 (test, `retrieval`), q18 (test, `language_processing`). Evidence kiểm chứng thủ công: mỗi câu có `evidence_quotes` là đoạn trích distinctive từ section nguồn, và `gold_source_ids` + `gold_pages` xác định article + section index. Evidence xác minh bằng cách đọc trực tiếp section tương ứng trong corpus cache (`starter/data/corpus_cache.json`) trước khi gán.

- **Tiêu chí chọn chiến lược đã chốt trước khi xem test:**
  Chọn chiến lược có **Hit@3 dev cao nhất**. Nếu bằng nhau, ưu tiên Hit@1 dev. Nếu vẫn bằng, ưu tiên số chunks nhỏ hơn (tốt hơn cho inference cost). Tiêu chí này được chốt trước khi xem kết quả test.

---

## Kết quả

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian | Chi phí |
|---|---|---|---|---|---|---|
| A (fixed 500/50) | dev | 7/15 (47%) | 13/15 (87%) | 518 | 39s | 0 |
| A | test | 3/5 (60%) | 3/5 (60%) | 518 | 39s | 0 |
| B (paragraph 800) | dev | 4/15 (27%) | 12/15 (80%) | 370 | 36s | 0 |
| B | test | 3/5 (60%) | 3/5 (60%) | 370 | 36s | 0 |
| C (paragraph 800+filter) | dev | 4/15 (27%) | 12/15 (80%) | 370 | 36s | 0 |
| C | test | 3/5 (60%) | 3/5 (60%) | 370 | 36s | 0 |

| Strategy | Coherence trung bình /2 (n=10) | Metadata đủ và đúng (n/N) | Filter tuân thủ (n/N hoặc N/A) | Grounding 0/1/2 (số câu) và trung bình /2 |
|---|---|---|---|---|
| A | **1.5/2** (scores: 2,1,2,1,1,2,2,2,0,2) | 15/15 (100%) | N/A (strategy A không filter) | 0:1 / 1:1 / 2:3 → **1.4/2** |
| B | **1.9/2** (scores: 2,1,2,2,2,2,2,2,2,2) | — (audited A only) | N/A | — |
| C | **1.9/2** (scores: 2,1,2,2,2,2,2,2,2,2) | — | 2/4 filter questions hit@3 (q14 dev hit, q16 miss, q17 miss, q18 hit) | — |

**Đường dẫn trace, nhãn thủ công và cách chọn mẫu:**
- Trace đầy đủ: `starter/results/traces.json` (60 records = 20 câu × 3 strategies)
- Tóm tắt: `starter/results/summary.json`
- Coherence audit: `starter/results/coherence_audit.json` — 10 positions chọn theo tiêu chí: mỗi article chọn tối đa 2 positions, ưu tiên intro section và section có evidence cho câu benchmark.
- Metadata audit: `starter/results/metadata_audit.json` — 5 test questions × top-3 = 15 records
- Grounding audit: `starter/results/grounding_audit.json` — 5 test questions, strategy A

---

## Phân tích và kết luận

**So sánh A–B và B–C; 2 ví dụ sai có ID câu và nguồn:**

*A vs B:* Chiến lược A (fixed-size 500/50) đạt Hit@1 dev cao hơn (47% vs 27%) vì chunk nhỏ hơn → xác suất cao hơn khi gold evidence nằm đúng rank 1 của một chunk cụ thể. Hit@3 A (87%) nhỉnh hơn B (80%). Tuy nhiên, A tạo 518 chunks so với B's 370 — nhiều hơn 40%, dẫn đến chi phí index và query cao hơn, và coherence thấp hơn (avg 1.5 vs 1.9).

*B vs C:* Kết quả retrieval giống hệt nhau (cùng 370 chunks, cùng vectors). Sự khác biệt chỉ ở filter questions: C loại bỏ chunks không thuộc `topic_category` được chỉ định. Với q14 (dev, filter=retrieval): B và C đều hit vì rank 1 đã là retrieval article. Với q17 (test, filter=retrieval): cả B và C miss vì gold section p9 (726 chars) không có chunk nào lọt top-3 — section ngắn, semantic tương đồng thấp với query.

**Ví dụ sai:**
1. **q04** (dev, miss A/B/C): "How does masked language modeling work in BERT pre-training?" — gold: `bert_language_model p5`. Retriever trả về `bert_language_model p4` (pre-training overview, rank 2 ở A) thay vì p5 (MLM detail). Nguyên nhân: p4 có `"Masked language modeling (MLM): In this task, BERT ingests..."` — overlap ngữ nghĩa cao với query vì đây là intro summary của MLM. Gold section p5 chứa chi tiết 15% random masking nhưng có semantic embedding gần giống p4. Chunk boundary của fixed-size cắt p5 thành các mảnh nhỏ làm giảm signal.

2. **q12** (dev, miss A/B/C): "What are higher-level NLP applications according to the Natural Language Processing article?" — gold: `natural_language_processing p14`. Retriever trả về `natural_language_processing p0` (intro, rank 1) vì intro section có câu tóm tắt đề cập "higher-level NLP applications" nhưng không có nội dung chi tiết. Section p14 (4082 chars) bị chia thành nhiều chunks nhỏ, mỗi chunk chỉ cover một ứng dụng (summarization, QA, etc.) → không có chunk nào đủ representative của toàn section → query "higher-level NLP applications" không khớp tốt với bất kỳ chunk cụ thể nào.

**Chiến lược chọn, đánh đổi, điều kiện áp dụng:**

**Chọn Strategy A** (fixed 500/50) vì đạt Hit@3 dev cao nhất (87%) — tiêu chí đã chốt trước khi xem test. Kết quả test (60%) bằng B và C.

Đánh đổi của A: (+) Hit@1 và Hit@3 cao nhất trên dev; (-) coherence thấp hơn (1.5/2 vs 1.9/2) — fixed-size cắt giữa câu, đặc biệt với list structures (BERT MLM bullet points) và sections bắt đầu bằng quote fragment (IR History). A tốt hơn khi câu hỏi map rõ ràng vào một đoạn văn cụ thể. B/C tốt hơn khi cần coherence (ứng dụng downstream cần chunk có thể đọc độc lập).

Điều kiện áp dụng: A phù hợp cho corpus có văn xuôi đều (không có list structure phức tạp); B/C phù hợp cho corpus có cấu trúc rõ ràng (Wikipedia sections, documentation). Filter (C) chỉ có giá trị khi corpus có nhiều domain rõ ràng và câu hỏi có filter hợp lệ từ người dùng.

**Hạn chế của dữ liệu, nhãn, test nhỏ, đo thời gian/chi phí:**
- Test set chỉ 5 câu → kết quả test không ổn định (1 câu đúng/sai = ±20% swing).
- Coherence và grounding do tác giả tự chấm → conflict of interest, nhãn có thể quá lạc quan.
- Thời gian embed (~33–35s/strategy) đo trên 1 lần chạy CPU, không có multiple runs để lấy trung bình.
- Chi phí đo = 0 (local) — không phản ánh chi phí thực tế nếu dùng API embedding.
- Wikipedia content thay đổi theo thời gian — benchmark có thể không còn valid sau vài tháng.
- Corpus English-only; model BGE-small-en-v1.5 không hỗ trợ tiếng Việt tốt.

**Nếu làm lại hoặc mở rộng, cải tiến gì?**
- Thêm **reranking** (cross-encoder) sau initial retrieval để cải thiện Hit@1 — đặc biệt với các câu bị retriever confuse section lân cận (q04, q16).
- Thêm **section-aware chunking**: split thành sections trước, sau đó chunk mỗi section độc lập, để tránh cross-section contamination.
- Thêm **hybrid search** (BM25 + dense) để xử lý câu hỏi đặc thù từ khóa (q17 "challenges" — sparse search sẽ tìm section "Challenges" tốt hơn dense).
- Mở rộng test set lên ≥20 câu để có kết quả đáng tin cậy hơn.
- Dùng multilingual embedding model (e.g., `paraphrase-multilingual-MiniLM-L12-v2`) nếu corpus có tiếng Việt.
