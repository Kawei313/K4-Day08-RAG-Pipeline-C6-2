# RAG Evaluation Results

## Cấu hình mặc định

| Metric | Score |
|--------|-------|
| faithfulness | 0.5500 |
| answer_relevancy | 0.3584 |
| context_recall | 0.4000 |
| context_precision | 0.9567 |

## So sánh A/B (Các cấu hình Retrieval)

| Config | faithfulness | answer_relevancy | context_recall | context_precision | 
|--------|-------|-------|-------|-------|
| hybrid_rerank | 0.6000 | 0.5377 | 0.4000 | 0.9467 | 
| dense_only | 0.4000 | 0.3562 | 0.4000 | 0.7867 | 


*Báo cáo được tạo tự động bởi RAGAS.*
