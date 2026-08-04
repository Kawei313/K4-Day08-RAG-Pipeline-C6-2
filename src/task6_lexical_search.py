"""Task 6 — BM25 lexical retrieval over the standardized Markdown corpus."""

import re
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CORPUS: list[dict] = []
_BM25 = None


def _tokenize(text: str) -> list[str]:
    """Keep Vietnamese letters/numbers while removing punctuation consistently."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _load_corpus() -> list[dict]:
    documents = []
    for path in STANDARDIZED_DIR.rglob("*.md"):
        content = path.read_text(encoding="utf-8").strip()
        if content:
            documents.append({
                "content": content,
                "metadata": {"source": path.name, "type": path.parent.name},
            })
    return documents


def build_bm25_index(corpus: list[dict]):
    """Build the rank-bm25 index for the supplied corpus."""
    from rank_bm25 import BM25Okapi

    return BM25Okapi([_tokenize(document["content"]) for document in corpus])


def _get_index_and_corpus():
    global CORPUS, _BM25
    if not CORPUS:
        CORPUS = _load_corpus()
        try:
            _BM25 = build_bm25_index(CORPUS) if CORPUS else None
        except ModuleNotFoundError:
            _BM25 = None
    return CORPUS, _BM25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Return BM25-ranked chunks/documents whose score is greater than zero."""
    if not query.strip() or top_k <= 0:
        return []
    corpus, bm25 = _get_index_and_corpus()
    if not corpus:
        return []

    tokens = _tokenize(query)
    if bm25 is not None:
        scores = [float(score) for score in bm25.get_scores(tokens)]
    else:
        # Keeps the module usable for diagnostics before rank-bm25 is installed.
        query_terms = set(tokens)
        scores = [float(len(query_terms.intersection(_tokenize(doc["content"])))) for doc in corpus]

    results = [
        {"content": doc["content"], "score": score, "metadata": doc["metadata"]}
        for doc, score in zip(corpus, scores)
        if score > 0
    ]
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    for result in lexical_search("Nhà hàng hải sản ngon", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
