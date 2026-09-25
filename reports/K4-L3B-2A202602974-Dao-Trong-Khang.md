# Báo cáo đóng góp cá nhân

## Thông tin

- Họ và tên: Đào Trọng Khang
- Mã học viên: 2A202602974
- Nhóm: K4-L3B (nhóm một thành viên)
- Repository/branch: repository hiện tại, nhánh `main`

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/bằng chứng | Trạng thái |
| --- | --- | --- | --- |
| Thu thập và chuẩn hóa dữ liệu | Chọn chủ đề khả năng tiếp cận số EU; thu thập 3 tài liệu pháp lý và 5 trang hướng dẫn; chuẩn hóa Markdown có provenance | `src/task1_collect_legal_docs.py`, `src/task2_crawl_news.py`, `src/task3_convert_markdown.py`, `data/` | Done |
| Chunk và index | Recursive chunk 500/50, ID ổn định, embedding đa ngôn ngữ, Chroma cosine và upsert không trùng | `src/task4_chunking_indexing.py`; 8 documents, 1.157 chunks | Done |
| Retrieval | Dense search, BM25, RRF không mutate, fallback dựa trên dense score | `src/task5_semantic_search.py` đến `src/task9_retrieval_pipeline.py` | Done |
| Generation và UI | Context có nhãn `[S…]`, kiểm tra citation, Gemini dispatch, safe refusal và Streamlit lưu nguồn trong lịch sử | `src/task10_generation.py`, `app.py` | Done |
| Evaluation | Tạo 15 golden cases, runner A/B, lưu 30 raw records, phân tích metric và lỗi | `scripts/evaluate_ab.py`, `group_project/evaluation/` | Done |

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Dùng `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, vector chuẩn hóa, chunk 500 ký tự và overlap 50.  
   **Lý do/evidence:** Model chạy local và hỗ trợ câu hỏi tiếng Việt trên tài liệu tiếng Anh; cùng `embed_texts()` được dùng cho corpus và query.  
   **Trade-off:** Model nhỏ giúp chạy lại nhanh nhưng dense-only bỏ lỡ một số cụm chính xác như WCAG-EM; BM25/RRF được giữ để bù lại.

2. **Quyết định:** Chọn hybrid + RRF làm cấu hình ứng viên nhưng vẫn giữ safe refusal dựa trên cosine threshold 0,30.  
   **Lý do/evidence:** Trên 15 case với corpus hoàn chỉnh, hybrid đạt context recall 0,9667 so với 0,8333 của dense-only và average 0,9475 so với 0,8483.  
   **Trade-off:** Hybrid thêm BM25 và candidate depth; một case vẫn bị model từ chối dù evidence đã được retrieve, nên cần regression test generation.

## Kiểm thử và kết quả

- Contract test: `pytest tests/test_contracts.py -q`.
- Acceptance test: `pytest tests/test_acceptance.py -q`.
- Evaluation A/B: `python -u scripts/evaluate_ab.py` với 15 case × 2 cấu hình.
- Query đúng domain trả answer có citation ánh xạ tới `sources`; query làm bánh trả safe refusal, `sources=[]`, `retrieval_source="none"`.
- Lỗi đã phát hiện và xử lý: encoding trang EC, EUR-Lex WAF, model Gemini không còn khả dụng/quá tải, citation ngoài danh sách, Chroma stale IDs và PageIndex chưa cấu hình.

## Điều còn hạn chế

- LLM-as-judge dùng cùng họ model với generator và chỉ chạy một lần, chưa có confidence interval hoặc blind human review.
- Official Journal L 256 chứa nhiều nội dung ngoài hai quyết định, làm corpus legal có các chunk lân cận gây nhiễu.
- Nếu có thêm thời gian, ưu tiên tách đúng phạm vi từng act trong OJ, thêm adjacent-section retrieval và chạy evaluation lặp lại với evaluator độc lập.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của tôi và có thể chạy lại trong buổi demo.

- Ngày: 2026-09-26
- Tên thành viên: Đào Trọng Khang
