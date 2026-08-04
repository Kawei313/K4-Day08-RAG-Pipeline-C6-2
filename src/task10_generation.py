"""Task 10 - Generation with citations for the travel-guide RAG assistant."""

import os
from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve

load_dotenv()

# top_k=5 gives enough evidence for itinerary/food questions while keeping the
# prompt compact. top_p=0.9 allows natural wording, while temperature=0.3 keeps
# the answer factual and grounded in retrieved context.
TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

UNVERIFIED_MESSAGE = "I cannot verify this information"

SYSTEM_PROMPT = f"""You are a smart travel-guide assistant.
Answer only from the provided context.
For every factual claim, add a citation in brackets like [source, n.d.].
If the context does not contain enough evidence, say "{UNVERIFIED_MESSAGE}".
Do not invent places, prices, schedules, addresses, or opening hours."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Reorder chunks to reduce the lost-in-the-middle effect.

    Input sorted by score: [1, 2, 3, 4, 5]
    Output pattern:        [1, 3, 5, 4, 2]
    """
    if len(chunks) <= 2:
        return list(chunks)
    return list(chunks[::2]) + list(chunks[1::2])[::-1]


def _citation_label(chunk: dict, index: int) -> str:
    metadata = chunk.get("metadata", {})
    source = metadata.get("source") or metadata.get("section") or f"source-{index}"
    year = metadata.get("year") or metadata.get("date") or "n.d."
    return f"{source}, {year}"


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks as context with explicit citation labels.

    Returns a single string the LLM can cite from.
    """
    parts = []

    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        citation = _citation_label(chunk, index)
        doc_type = metadata.get("type", "unknown")
        score = float(chunk.get("score", 0.0))
        content = chunk.get("content", "").strip()

        parts.append(
            f"[Document {index}]\n"
            f"Citation: [{citation}]\n"
            f"Type: {doc_type}\n"
            f"Score: {score:.4f}\n"
            f"Content:\n{content}"
        )

    return "\n\n---\n\n".join(parts)


def _fallback_answer(query: str, chunks: list[dict]) -> str:
    """Return a grounded answer when no LLM API key is configured."""
    if not chunks:
        return UNVERIFIED_MESSAGE

    lines = [
        "Duoi day la thong tin lien quan nhat tim thay trong nguon hien co:",
        "",
    ]

    for index, chunk in enumerate(chunks[:3], start=1):
        citation = _citation_label(chunk, index)
        excerpt = " ".join(chunk.get("content", "").split())[:450]
        lines.append(f"{index}. {excerpt} [{citation}]")

    lines.append("")
    lines.append(
        "Neu ban can thong tin chi tiet hon ngoai cac nguon tren, "
        f"{UNVERIFIED_MESSAGE}."
    )
    return "\n".join(lines)


def _call_llm(query: str, context: str) -> str:
    """Call OpenAI/OpenRouter chat completion when an API key exists."""
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""

    from openai import OpenAI

    client_kwargs = {"api_key": api_key}
    if os.getenv("OPENROUTER_API_KEY"):
        client_kwargs["base_url"] = "https://openrouter.ai/api/v1"
        model = OPENROUTER_MODEL
    else:
        model = OPENAI_MODEL

    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{query}\n\n"
                    "Answer with citations using only the citation labels above."
                ),
            },
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return response.choices[0].message.content or ""


def generate_with_citation(query: str, top_k: int = TOP_K, **retrieve_kwargs) -> dict:
    """
    Generate an answer with citations.

    Returns:
        {
            "answer": str,
            "sources": list[dict],
            "retrieval_source": str
        }
    """
    query = query.strip()
    if not query:
        return {"answer": UNVERIFIED_MESSAGE, "sources": [], "retrieval_source": "none"}

    chunks = retrieve(query, top_k=top_k, **retrieve_kwargs)
    retrieval_source = chunks[0].get("source", "hybrid") if chunks else "none"

    if not chunks:
        return {
            "answer": UNVERIFIED_MESSAGE,
            "sources": [],
            "retrieval_source": retrieval_source,
        }

    ordered_chunks = reorder_for_llm(chunks)
    context = format_context(ordered_chunks)

    try:
        answer = _call_llm(query, context)
    except Exception:
        answer = ""

    if not answer:
        answer = _fallback_answer(query, ordered_chunks)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    examples = [
        "Goi y lich trinh du lich Ha Long 2 ngay 1 dem",
        "Nhung mon an nen thu khi den Ha Long",
    ]

    for example in examples:
        result = generate_with_citation(example)
        print(f"\nQ: {example}\nA: {result['answer']}\n")
