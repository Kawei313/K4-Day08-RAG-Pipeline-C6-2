"""Task 5 — dense semantic retrieval over the Chroma collection from Task 4."""

from .task4_chunking_indexing import get_collection, get_embedding_model


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Return the most semantically similar chunks, sorted by cosine similarity."""
    if not query.strip() or top_k <= 0:
        return []

    try:
        collection = get_collection()
        count = collection.count()
        if count == 0:
            return []
        query_vector = get_embedding_model().encode(
            query, normalize_embeddings=True
        ).tolist()
        raw = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        # Before the Task 4 dependencies/index are ready, let hybrid retrieval use BM25.
        return []

    results = []
    for document, metadata, distance in zip(
        raw.get("documents", [[]])[0],
        raw.get("metadatas", [[]])[0],
        raw.get("distances", [[]])[0],
    ):
        results.append({
            "content": document,
            "score": round(max(0.0, 1.0 - float(distance)), 4),
            "metadata": metadata or {},
        })
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    for result in semantic_search("quy định trả hàng hoàn tiền shopee", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
