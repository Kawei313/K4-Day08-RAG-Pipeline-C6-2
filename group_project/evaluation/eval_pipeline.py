"""
RAG Evaluation Pipeline.

Sử dụng DeepEval / RAGAS / TruLens để đánh giá chất lượng RAG pipeline.
Chọn 1 framework và implement đầy đủ.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Lưu ý rate limit nếu dùng model OpenRouter ":free": RAGAS/DeepEval gọi LLM RẤT NHIỀU LẦN
(không phải 1 lần/câu hỏi mà nhiều lần/metric/câu hỏi). Model free của OpenRouter giới hạn
50 request/ngày CHO CẢ TÀI KHOẢN (không phải theo model hay theo API key — đổi model free
khác hay tạo key mới KHÔNG reset quota). Nếu chạy full 15+ câu hỏi mà bị rate limit giữa
chừng, thử giảm xuống subset 5 câu để chạy kịp trong buổi, hoặc nạp $10 credit để mở khóa
1000 request/ngày.
"""

import json
from pathlib import Path

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Option 1: DeepEval
# =============================================================================

def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng DeepEval.

    pip install deepeval
    """
    # TODO: Implement
    #
    # from deepeval import evaluate
    # from deepeval.metrics import (
    #     FaithfulnessMetric,
    #     AnswerRelevancyMetric,
    #     ContextualRecallMetric,
    #     ContextualPrecisionMetric,
    # )
    # from deepeval.test_case import LLMTestCase
    #
    # test_cases = []
    # for item in golden_dataset:
    #     result = rag_pipeline.generate_with_citation(item["question"])
    #     test_case = LLMTestCase(
    #         input=item["question"],
    #         actual_output=result["answer"],
    #         expected_output=item["expected_answer"],
    #         retrieval_context=[c["content"] for c in result["sources"]],
    #     )
    #     test_cases.append(test_case)
    #
    # metrics = [
    #     FaithfulnessMetric(threshold=0.7),
    #     AnswerRelevancyMetric(threshold=0.7),
    #     ContextualRecallMetric(threshold=0.7),
    #     ContextualPrecisionMetric(threshold=0.7),
    # ]
    #
    # results = evaluate(test_cases, metrics)
    # return results
    raise NotImplementedError("Implement evaluate_with_deepeval")


# =============================================================================
# Option 2: RAGAS
# =============================================================================

def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng RAGAS.

    pip install ragas
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from datasets import Dataset

    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    print("Đang chạy RAG pipeline để thu thập câu trả lời và context...")
    for i, item in enumerate(golden_dataset):
        print(f"  Đang xử lý câu hỏi {i+1}/{len(golden_dataset)}")
        result = rag_pipeline(item["question"])
        eval_data["question"].append(item["question"])
        eval_data["answer"].append(result["answer"])
        eval_data["contexts"].append([c["content"] for c in result["sources"]])
        eval_data["ground_truth"].append(item["expected_answer"])

    print("Bắt đầu đánh giá với RAGAS...")
    dataset = Dataset.from_dict(eval_data)
    
    import os
    # Cấu hình sử dụng Gemini nếu có key
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    kwargs = {}
    
    if gemini_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
            print("Phát hiện khóa API của Gemini! Cấu hình RAGAS sử dụng Gemini...")
            # Sử dụng flash để tiết kiệm chi phí và tốc độ nhanh, hoặc đổi sang pro nếu cần
            gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_key)
            gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_key)
            
            kwargs["llm"] = gemini_llm
            kwargs["embeddings"] = gemini_embeddings
        except ImportError:
            print("CẢNH BÁO: Tìm thấy GEMINI_API_KEY nhưng thư viện 'langchain-google-genai' chưa được cài đặt.")
            print("Vui lòng chạy lệnh: pip install langchain-google-genai")
            print("Sử dụng cấu hình mặc định của Ragas (OpenAI).")

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        **kwargs
    )
    return result.to_pandas()


# =============================================================================
# Option 3: TruLens
# =============================================================================

def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng TruLens.

    pip install trulens
    """
    # TODO: Implement
    #
    # from trulens.apps.custom import TruCustomApp
    # from trulens.core import Feedback
    # from trulens.providers.openai import OpenAI as TruOpenAI
    #
    # provider = TruOpenAI()
    #
    # f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
    # f_relevance = Feedback(provider.relevance).on_input_output()
    # f_context_relevance = Feedback(provider.context_relevance).on_input()
    #
    # tru_rag = TruCustomApp(
    #     rag_pipeline,
    #     app_name="EcommerceSupport_RAG",
    #     feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
    # )
    #
    # with tru_rag as recording:
    #     for item in golden_dataset:
    #         rag_pipeline.generate_with_citation(item["question"])
    #
    # # Dashboard: from trulens.dashboard import run_dashboard; run_dashboard()
    raise NotImplementedError("Implement evaluate_with_trulens")


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(rag_pipeline, golden_dataset: list[dict]):
    """
    So sánh A/B giữa ít nhất 2 configs.

    Gợi ý configs để so sánh:
    - Config A: hybrid search + reranking
    - Config B: dense-only (không reranking)
    - Config C: hybrid search + PageIndex fallback
    """
    from functools import partial
    
    # We will test two configurations:
    # 1. hybrid_rerank: use_reranking = True
    # 2. dense_only: use_reranking = False
    configs = {
        "hybrid_rerank": {"use_reranking": True},
        "dense_only": {"use_reranking": False},
    }
    
    results = {}
    for config_name, params in configs.items():
        print(f"\n--- Đang đánh giá cấu hình: {config_name} ---")
        # Create a pipeline wrapper with the given kwargs
        pipeline_wrapper = partial(rag_pipeline, **params)
        df = evaluate_with_ragas(pipeline_wrapper, golden_dataset)
        results[config_name] = df
    
    return results


# =============================================================================
# Export Results
# =============================================================================

def export_results(results: dict, comparison: dict):
    """Export evaluation results to results.md"""
    content = "# RAG Evaluation Results\n\n"
    content += "## Cấu hình mặc định\n\n"
    
    if results is not None and not results.empty:
        df_mean = results.mean(numeric_only=True)
        content += "| Metric | Score |\n|--------|-------|\n"
        for metric, score in df_mean.items():
            content += f"| {metric} | {score:.4f} |\n"
            
    content += "\n## So sánh A/B (Các cấu hình Retrieval)\n\n"
    if comparison:
        content += "| Config | "
        metrics = list(comparison.values())[0].select_dtypes(include='number').columns.tolist()
        for metric in metrics:
            content += f"{metric} | "
        content += "\n|--------|" + "-------|" * len(metrics) + "\n"
        
        for config_name, df in comparison.items():
            content += f"| {config_name} | "
            means = df.mean(numeric_only=True)
            for metric in metrics:
                content += f"{means.get(metric, 0):.4f} | "
            content += "\n"
            
    content += "\n*Báo cáo được tạo tự động bởi RAGAS.*\n"
    
    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"\nĐã xuất báo cáo ra {RESULTS_PATH}")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    # Import RAG pipeline
    import sys
    import os
    # Add root folder to sys.path to import src
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    from src.task10_generation import generate_with_citation
    
    # Run evaluation with RAGAS
    try:
        print("Đánh giá cấu hình mặc định:")
        results_df = evaluate_with_ragas(generate_with_citation, golden_dataset)
        print("\n=== ĐÁNH GIÁ THÀNH CÔNG (Mặc định) ===")
        print(results_df.mean(numeric_only=True))
        
        # Save raw results to CSV
        csv_path = Path(__file__).parent / "ragas_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\nĐã lưu chi tiết đánh giá vào {csv_path}")
        
        print("\nBắt đầu chạy so sánh A/B...")
        comparison = compare_configs(generate_with_citation, golden_dataset)
        
        export_results(results_df, comparison)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nLỗi khi chạy RAGAS: {e}")
        print("Lưu ý: RAGAS cần OPENAI_API_KEY (mặc định) hoặc GEMINI_API_KEY/GOOGLE_API_KEY để gọi mô hình Judge.")
