"""Task 7 - Reranking and rank-fusion helpers."""

import math
import os
import re
from collections import defaultdict


def _tokenize(text: str) -> set[str]:
    """Return normalized lexical tokens for lightweight relevance scoring."""
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _stable_key(item: dict) -> str:
    """Build a stable key so duplicate chunks from dense/sparse search merge."""
    metadata = item.get("metadata", {})
    source = metadata.get("source", "")
    chunk_index = metadata.get("chunk_index", "")
    return item.get("id") or f"{source}:{chunk_index}:{item.get('content', '')[:120]}"


def _cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right))
    if denominator == 0:
        return 0.0
    return sum(x * y for x, y in zip(left, right)) / denominator


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Optional Jina cross-encoder reranker.

    If JINA_API_KEY is not configured, the function falls back to the local
    lightweight reranker used by this lab.
    """
    if not candidates or top_k <= 0:
        return []

    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        return rerank(query, candidates, top_k=top_k, method="local")

    import requests

    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": [candidate.get("content", "") for candidate in candidates],
            "top_n": min(top_k, len(candidates)),
        },
        timeout=30,
    )
    response.raise_for_status()

    reranked = []
    for result in response.json().get("results", []):
        source_item = candidates[result["index"]]
        reranked.append({**source_item, "score": float(result["relevance_score"])})
    return reranked[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Select relevant but less-duplicative candidates with MMR."""
    available = [item for item in candidates if item.get("embedding")]
    selected: list[dict] = []

    while available and len(selected) < top_k:
        best_item = None
        best_score = float("-inf")

        for candidate in available:
            relevance = _cosine(query_embedding, candidate["embedding"])
            redundancy = max(
                (_cosine(candidate["embedding"], item["embedding"]) for item in selected),
                default=0.0,
            )
            score = lambda_param * relevance - (1.0 - lambda_param) * redundancy
            if score > best_score:
                best_item = candidate
                best_score = score

        selected.append({**best_item, "score": round(float(best_score), 4)})
        available.remove(best_item)

    return selected


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """
    Fuse ranked lists with Reciprocal Rank Fusion.

    RRF score depends on rank position, not raw relevance, so Task 9 should use
    the original semantic score for fallback decisions.
    """
    if top_k <= 0:
        return []

    scores: dict[str, float] = defaultdict(float)
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = _stable_key(item)
            scores[key] += 1.0 / (k + rank)
            if key not in items:
                items[key] = item

    fused = [
        {**items[key], "score": round(score, 6)}
        for key, score in scores.items()
    ]
    fused.sort(key=lambda item: item["score"], reverse=True)
    return fused[:top_k]


def _local_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Re-score candidates with keyword overlap plus their incoming score."""
    query_terms = _tokenize(query)
    reranked = []

    for candidate in candidates:
        content_terms = _tokenize(candidate.get("content", ""))
        overlap = len(query_terms.intersection(content_terms))
        coverage = overlap / max(1, len(query_terms))
        incoming = float(candidate.get("score", 0.0))
        score = 0.7 * coverage + 0.3 * incoming
        reranked.append({**candidate, "score": round(score, 4)})

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def rerank(query: str, candidates: list[dict], top_k: int = 5, method: str = "rrf") -> list[dict]:
    """
    Re-score and re-order candidates based on relevance to the query.

    Supported methods:
        - "rrf": preserve/finalize rank-fused candidates
        - "local": keyword-overlap reranking
        - "cross_encoder": Jina API when JINA_API_KEY exists
    """
    if not candidates or top_k <= 0:
        return []

    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "local":
        return _local_rerank(query, candidates, top_k)
    if method == "rrf":
        ranked = sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)
        return ranked[:top_k]
    if method == "mmr":
        raise ValueError("MMR requires query_embedding; call rerank_mmr directly.")

    raise ValueError(f"Unknown reranking method: {method}")
