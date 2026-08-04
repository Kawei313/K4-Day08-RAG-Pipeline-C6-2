"""Task 8 - PageIndex vectorless fallback.

The real PageIndex path is used when PAGEINDEX_API_KEY and document IDs are
configured. For classroom/demo use, this module also provides a local
vectorless fallback over Markdown chunks so the retrieval pipeline keeps
working without an external account.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from .task6_lexical_search import lexical_search

load_dotenv()

PROJECT_DIR = Path(__file__).parent.parent
LANDING_DIR = PROJECT_DIR / "data" / "landing"
MANIFEST_PATH = PROJECT_DIR / "data" / "pageindex_documents.json"
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")


def _client():
    """Create a PageIndex client when the SDK and API key are available."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY is not configured.")

    from pageindex import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def upload_documents() -> dict[str, str]:
    """
    Upload landing documents to PageIndex and persist returned document IDs.

    This is optional for the lab. Without PAGEINDEX_API_KEY, pageindex_search()
    uses the local vectorless fallback.
    """
    client = _client()
    document_ids: dict[str, str] = {}

    supported_extensions = {".pdf", ".docx", ".doc", ".md", ".txt"}
    for path in sorted(LANDING_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in supported_extensions:
            continue

        response = client.submit_document(str(path))
        doc_id = response.get("doc_id") or response.get("id")
        if not doc_id:
            raise RuntimeError(f"PageIndex did not return a document ID for {path.name}.")
        document_ids[str(path.relative_to(LANDING_DIR))] = doc_id

    MANIFEST_PATH.write_text(
        json.dumps(document_ids, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return document_ids


def _load_document_ids() -> list[str]:
    configured = os.getenv("PAGEINDEX_DOCUMENT_IDS", "")
    if configured:
        return [item.strip() for item in configured.split(",") if item.strip()]

    if MANIFEST_PATH.exists():
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return list(payload.values())

    return []


def _local_vectorless_search(query: str, top_k: int) -> list[dict]:
    """
    Local vectorless fallback using chunk text directly, not embeddings.

    It relies on lexical matching over document text and marks results as
    "pageindex" so Task 9 can treat them as fallback results.
    """
    results = lexical_search(query, top_k=top_k)
    for result in results:
        result["source"] = "pageindex"
        result["metadata"] = {
            **result.get("metadata", {}),
            "fallback": "local_vectorless",
        }
    return results


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval using PageIndex, with a local fallback.

    Returns:
        List of {
            "content": str,
            "score": float,
            "metadata": dict,
            "source": "pageindex"
        }
    """
    query = query.strip()
    if not query or top_k <= 0:
        return []

    document_ids = _load_document_ids()
    if not PAGEINDEX_API_KEY or not document_ids:
        return _local_vectorless_search(query, top_k)

    client = _client()
    results = []

    for doc_id in document_ids:
        try:
            retrieval = client.submit_query(doc_id, query)
            retrieval_id = retrieval.get("retrieval_id") or retrieval.get("id")

            payload = {}
            for _ in range(30):
                payload = client.get_retrieval(retrieval_id)
                if payload.get("status") == "completed":
                    break
                time.sleep(1)
            else:
                continue
        except Exception:
            continue

        for node in payload.get("retrieved_nodes", []):
            for group in node.get("relevant_contents", []):
                for item in group:
                    content = item.get("relevant_content", "").strip()
                    if not content:
                        continue

                    results.append(
                        {
                            "content": content,
                            "score": round(1.0 / (len(results) + 1), 4),
                            "metadata": {
                                "section": item.get("section_title", ""),
                                "doc_id": doc_id,
                            },
                            "source": "pageindex",
                        }
                    )
                    if len(results) >= top_k:
                        return results

    return results or _local_vectorless_search(query, top_k)


if __name__ == "__main__":
    for result in pageindex_search("lich trinh du lich ha long", top_k=3):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
