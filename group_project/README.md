# Bài Tập Nhóm — E-commerce Support RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1: Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về chính sách thương mại điện tử và hỗ trợ khách hàng liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

Xem code mẫu (DeepEval/RAGAS/TruLens) chi tiết trong `README.md` gốc mục "Yêu cầu 2".

### Deliverable Evaluation

- [x] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [x] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [x] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [x] So sánh A/B ít nhất 2 configs

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

```mermaid
flowchart TD
    subgraph Ingestion["📥 Data Ingestion (Task 1–3)"]
        T1["Task 1\nCollect Legal Docs\ntask1_collect_legal_docs.py"]
        T2["Task 2\nCrawl News\ntask2_crawl_news.py"]
        T3["Task 3\nConvert to Markdown\ntask3_convert_markdown.py"]
        T1 --> T3
        T2 --> T3
    end

    subgraph Indexing["🗂️ Chunking & Indexing (Task 4)"]
        T4["Task 4\nChunking + Vector Indexing\ntask4_chunking_indexing.py"]
        T3 --> T4
    end

    subgraph Storage["🗄️ Storage"]
        VDB[("Vector Store\nDense Embeddings")]
        BM25[("BM25 Index\nSparse")]
        T4 --> VDB
        T4 --> BM25
    end

    subgraph Retrieval["🔍 Hybrid Retrieval Pipeline (Task 5–9)"]
        Q["User Query"]
        T5["Task 5\nSemantic Search\nDense - task5_semantic_search.py"]
        T6["Task 6\nLexical Search\nBM25 - task6_lexical_search.py"]
        T7["Task 7\nRRF Fusion + Reranking\nCrossEncoder - task7_reranking.py"]
        T8["Task 8\nPageIndex Fallback\ntask8_pageindex_vectorless.py"]
        T9["Task 9\nRetrieval Orchestration\ntask9_retrieval_pipeline.py"]
        VDB --> T5
        BM25 --> T6
        Q --> T5
        Q --> T6
        T5 --> T7
        T6 --> T7
        T7 --> T9
        T8 -->|"low confidence fallback"| T9
    end

    subgraph Generation["💬 Generation (Task 10)"]
        T10["Task 10\nGenerate with Citation\ntask10_generation.py"]
        T9 --> T10
        T10 --> Answer["Answer + Source Citations"]
    end

    subgraph Evaluation["📊 Evaluation"]
        GD["Golden Dataset\n15+ Q&A pairs"]
        EP["eval_pipeline.py\nRAGAS Framework"]
        RES["results.md\nFaithfulness / Relevancy\nRecall / Precision"]
        GD --> EP
        Answer --> EP
        EP --> RES
    end

    subgraph UI["🖥️ Frontend"]
        APP["app.py\nStreamlit / Chainlit\nChat UI + Conversation Memory"]
    end

    Q --> APP
    Answer --> APP
```

---

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| Nguyễn Trí Trung | 2A202601594 | Role 1 (Team Leader & RAG Architect) | Hoàn thành |
| Trần Đặng Vương Quốc Long | 2A202601744 | Role 2 (Data Engineering & Scraping Dev) | Hoàn thành |
| Nguyễn Văn Qúy | 2A202601508 | Role 3 (Vector Database & Dense Search Dev): Task 4 + 5 | Hoàn thành |
| Nguyễn Nhật Minh | 2A202601414 | Role 4 (Sparse Retrieval & Fallback Dev) | Hoàn thành |
| Phạm Việt Bách | 2A202601410 | Role 5 (Frontend UI & App Integration Dev) | Hoàn thành |
| Trần Lê Quý Đăng | 2A202601408 | Role 6 (Evaluation & QA Engineer) | Hoàn thành |

---

## Hướng Dẫn Chạy

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy app
streamlit run app.py
# hoặc
chainlit run app.py
```

---

## Lưu ý

Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
