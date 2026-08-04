"""Task 9 - Complete hybrid retrieval pipeline."""

from concurrent.futures import ThreadPoolExecutor

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.6
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"


def _mark_source(results: list[dict], source: str) -> list[dict]:
    """Attach retrieval source without mutating caller-owned dictionaries."""
    return [{**item, "source": source} for item in results]


def _merge_dense_first(dense_results: list[dict], sparse_results: list[dict], top_k: int) -> list[dict]:
    """Simple merge used when reranking/fusion is disabled."""
    seen = set()
    merged = []

    for item in [*dense_results, *sparse_results]:
        key = item.get("content", "")
        if key in seen:
            continue
        seen.add(key)
        merged.append({**item, "source": "hybrid"})
        if len(merged) >= top_k:
            break

    return merged


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Run the complete retrieval pipeline.

    Steps:
        1. Run semantic_search and lexical_search
        2. Merge with RRF
        3. Rerank/finalize ordering
        4. If the original semantic score is below threshold, use PageIndex
        5. Return top_k results
    """
    query = query.strip()
    if not query or top_k <= 0:
        return []

    search_k = max(top_k * 2, top_k)
    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(semantic_search, query, search_k)
        sparse_future = executor.submit(lexical_search, query, search_k)
        dense_results = dense_future.result()
        sparse_results = sparse_future.result()

    best_dense_score = dense_results[0]["score"] if dense_results else 0.0

    if use_reranking:
        merged = rerank_rrf([dense_results, sparse_results], top_k=search_k)
        merged = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        hybrid_results = _mark_source(merged, "hybrid")
    else:
        hybrid_results = _merge_dense_first(dense_results, sparse_results, top_k)

    # Important: compare the threshold with the original semantic score, not the
    # RRF score. RRF is rank-only and is not calibrated as relevance.
    if best_dense_score < score_threshold:
        fallback_results = pageindex_search(query, top_k=top_k)
        if fallback_results:
            return fallback_results[:top_k]

    return hybrid_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "lich trinh du lich Ha Long 2 ngay 1 dem",
        "mon an dac san Ha Long",
        "xyzabc123nonsense",
    ]

    for query_text in test_queries:
        print(f"\nQuery: {query_text}")
        print("-" * 60)
        for index, result in enumerate(retrieve(query_text, top_k=3), start=1):
            print(f"{index}. [{result['score']:.3f}] [{result['source']}] {result['content'][:80]}...")
