"""Task 8 — optional PageIndex vectorless fallback."""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_DIR = Path(__file__).parent.parent
LEGAL_DIR = PROJECT_DIR / "data" / "landing" / "legal"
MANIFEST_PATH = PROJECT_DIR / "data" / "pageindex_documents.json"
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")


def _client():
    from pageindex import PageIndexClient
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY is not configured.")
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def upload_documents() -> dict[str, str]:
    """Upload legal PDFs and persist their PageIndex document IDs locally."""
    client = _client()
    document_ids: dict[str, str] = {}
    for pdf_path in LEGAL_DIR.glob("*.pdf"):
        response = client.submit_document(str(pdf_path))
        doc_id = response.get("doc_id") or response.get("id")
        if not doc_id:
            raise RuntimeError(f"PageIndex did not return doc_id for {pdf_path.name}: {response}")
        document_ids[pdf_path.name] = doc_id

    MANIFEST_PATH.write_text(json.dumps(document_ids, indent=2), encoding="utf-8")
    return document_ids


def _load_document_ids() -> list[str]:
    configured = os.getenv("PAGEINDEX_DOCUMENT_IDS", "")
    if configured:
        return [value.strip() for value in configured.split(",") if value.strip()]
    if MANIFEST_PATH.exists():
        return list(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")).values())
    return []


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Use PageIndex legacy retrieval when configured; otherwise return no fallback."""
    if not query.strip() or top_k <= 0 or not PAGEINDEX_API_KEY:
        return []
    document_ids = _load_document_ids()
    if not document_ids:
        return []

    client = _client()
    results = []
    for doc_id in document_ids:
        try:
            retrieval = client.submit_query(doc_id, query)
            retrieval_id = retrieval.get("retrieval_id") or retrieval.get("id")
            for _ in range(30):
                payload = client.get_retrieval(retrieval_id)
                if payload.get("status") == "completed":
                    break
                time.sleep(1)
            else:
                continue
        except (AttributeError, RuntimeError, KeyError):
            # Current PageIndex SDK may expose Chat API only; fallback remains optional.
            continue

        for node in payload.get("retrieved_nodes", []):
            for group in node.get("relevant_contents", []):
                for item in group:
                    content = item.get("relevant_content", "").strip()
                    if content:
                        results.append({
                            "content": content,
                            "score": 1.0 / (len(results) + 1),
                            "metadata": {"section": item.get("section_title", ""), "doc_id": doc_id},
                            "source": "pageindex",
                        })
                        if len(results) >= top_k:
                            return results
    return results
