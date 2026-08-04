"""Task 7 — result fusion and optional reranking helpers."""

from math import sqrt


def _cosine(left: list[float], right: list[float]) -> float:
    denominator = sqrt(sum(x * x for x in left)) * sqrt(sum(x * x for x in right))
    return sum(x * y for x, y in zip(left, right)) / denominator if denominator else 0.0


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Use Jina when configured; otherwise preserve the incoming ranking safely."""
    import os
    import requests

    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        return sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)[:top_k]
    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": [candidate["content"] for candidate in candidates],
            "top_n": min(top_k, len(candidates)),
        },
        timeout=30,
    )
    response.raise_for_status()
    return [
        {**candidates[item["index"]], "score": float(item["relevance_score"])}
        for item in response.json().get("results", [])
    ]


def rerank_mmr(query_embedding: list[float], candidates: list[dict], top_k: int = 5,
               lambda_param: float = 0.7) -> list[dict]:
    """Select relevant but non-duplicative candidates using MMR."""
    available = [item for item in candidates if item.get("embedding")]
    selected: list[dict] = []
    while available and len(selected) < top_k:
        def mmr_score(candidate: dict) -> float:
            relevance = _cosine(query_embedding, candidate["embedding"])
            redundancy = max((_cosine(candidate["embedding"], item["embedding"])
                              for item in selected), default=0.0)
            return lambda_param * relevance - (1 - lambda_param) * redundancy
        best = max(available, key=mmr_score)
        best = {**best, "score": mmr_score(best)}
        selected.append(best)
        available.remove(next(item for item in available if item["content"] == best["content"]))
    return selected


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """Fuse independent rankers with Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = item.get("id") or item["content"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items.setdefault(key, item)
    return [
        {**items[key], "score": score}
        for key, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]
    ]


def rerank(query: str, candidates: list[dict], top_k: int = 5, method: str = "rrf") -> list[dict]:
    """Public reranking interface; RRF accepts one already-merged list here."""
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "rrf":
        return sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)[:top_k]
    if method == "mmr":
        raise ValueError("MMR requires query_embedding; call rerank_mmr directly.")
    raise ValueError(f"Unknown reranking method: {method}")
