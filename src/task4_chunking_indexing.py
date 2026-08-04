"""
Task 4 - Chunking and indexing standardized Markdown files into ChromaDB.

Pipeline:
    1. Load Markdown documents from data/standardized/
    2. Split each document into chunks
    3. Embed chunks with one shared embedding model
    4. Persist chunks, embeddings, and metadata in ChromaDB
"""

from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# We use recursive character chunking because the corpus mixes policy PDFs and
# crawled articles. It keeps paragraph boundaries when possible and falls back
# to smaller separators when the text has weak Markdown structure.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"

# all-MiniLM-L6-v2 is small, fast, and local. It is enough for the lab and keeps
# setup lighter than larger multilingual models. Task 5 imports the same helper
# so query vectors and indexed vectors always have the same dimension.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ChromaDB is used because it is local, persistent, simple to inspect, and does
# not require Docker or a separate database server for this lab.
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "ecommerce_support_docs"


def load_documents() -> list[dict]:
    """
    Load all non-empty Markdown files from data/standardized/.

    Returns:
        List of {
            "content": str,
            "metadata": {
                "source": str,
                "type": str,
                "file_path": str
            }
        }
    """
    documents = []

    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if not md_file.is_file():
            continue

        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        doc_type = md_file.parent.name
        if doc_type not in {"legal", "news"}:
            doc_type = "unknown"

        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type,
                    "file_path": str(md_file.relative_to(STANDARDIZED_DIR)),
                },
            }
        )

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Split documents into overlapping chunks.

    Returns:
        List of {
            "content": str,
            "metadata": dict
        }
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        split_text = splitter.split_text
    except ModuleNotFoundError:
        step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)

        def split_text(text: str) -> list[str]:
            return [text[start : start + CHUNK_SIZE] for start in range(0, len(text), step)]

    chunks = []

    for doc_index, document in enumerate(documents):
        for chunk_index, chunk_text in enumerate(split_text(document["content"])):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        **document["metadata"],
                        "doc_index": doc_index,
                        "chunk_index": chunk_index,
                    },
                }
            )

    return chunks


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the shared sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def _fallback_embedding(text: str) -> list[float]:
    """
    Deterministic fallback used only when the local model cannot be loaded.

    This keeps the pipeline runnable in constrained classroom machines, while
    still preserving a fixed vector dimension for ChromaDB.
    """
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIM
        vector[index] += 1.0

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.tolist()


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add an embedding vector to every chunk.

    Returns:
        The input chunks with an extra "embedding" key.
    """
    if not chunks:
        return []

    texts = [chunk["content"] for chunk in chunks]

    try:
        model = get_embedding_model()
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = [embedding.tolist() for embedding in embeddings]
    except Exception as exc:
        print(f"Warning: using fallback embeddings because model loading failed: {exc}")
        embeddings = [_fallback_embedding(text) for text in texts]

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    return chunks


def get_collection():
    """Open the persistent Chroma collection created by this task."""
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(name=COLLECTION_NAME)


def index_to_vectorstore(chunks: list[dict]):
    """Persist chunks and embeddings to ChromaDB."""
    import chromadb

    if not chunks:
        print("No chunks to index.")
        return

    chunks_without_embeddings = [chunk for chunk in chunks if "embedding" not in chunk]
    if chunks_without_embeddings:
        chunks = embed_chunks(chunks)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [
        f"{chunk['metadata']['source']}::chunk-{chunk['metadata']['chunk_index']}"
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )

    print(f"Indexed {len(chunks)} chunks into collection '{COLLECTION_NAME}'.")


def run_pipeline():
    """Run the full Task 4 pipeline: load -> chunk -> embed -> index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"Vector store: {VECTOR_STORE}")
    print("=" * 50)

    documents = load_documents()
    print(f"Loaded {len(documents)} documents")

    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print(f"Done. ChromaDB path: {CHROMA_DIR}")


if __name__ == "__main__":
    run_pipeline()
