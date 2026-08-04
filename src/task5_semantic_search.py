"""Task 5 - Dense semantic retrieval over the ChromaDB index from Task 4."""

from .task4_chunking_indexing import _fallback_embedding, get_collection, get_embedding_model


def _embed_query(query: str) -> list[float]:
    """Embed the query with the same model/configuration used in Task 4."""
    try:
        model = get_embedding_model()
        return model.encode(query, normalize_embeddings=True).tolist()
    except Exception:
        return _fallback_embedding(query)


def _distance_to_score(distance: float) -> float:
    """
    Convert Chroma cosine distance to a similarity-like score.

    The collection in Task 4 is configured with hnsw:space="cosine", where lower
    distance is better. The returned score uses higher-is-better semantics.
    """
    return round(max(0.0, 1.0 - float(distance)), 4)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search the vector store for chunks semantically similar to the query.

    Returns:
        List of {"content": str, "score": float, "metadata": dict}
    """
    query = query.strip()
    if not query or top_k <= 0:
        return []

    try:
        collection = get_collection()
        count = collection.count()
    except Exception:
        return []

    if count == 0:
        return []

    query_vector = _embed_query(query)
    raw_results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    documents = raw_results.get("documents", [[]])[0]
    metadatas = raw_results.get("metadatas", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]

    results = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        results.append(
            {
                "content": document,
                "score": _distance_to_score(distance),
                "metadata": metadata or {},
            }
        )

    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    sample_query = "return refund policy"
    for result in semantic_search(sample_query, top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
