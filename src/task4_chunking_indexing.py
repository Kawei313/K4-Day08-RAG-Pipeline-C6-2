"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options (chọn 1, cân nhắc đánh đổi cài đặt nặng vs cần API key):
    - sentence-transformers/all-MiniLM-L6-v2 hoặc BAAI/bge-m3 — chạy local, không
      cần API key, nhưng cài nặng (~1-2GB vì kéo theo torch)
    - Google models/text-embedding-004 (768 dim) — nhẹ, cần GEMINI_API_KEY
    - OpenAI text-embedding-3-small (1536 dim) — nhẹ, cần OPENAI_API_KEY
    Gợi ý: đọc EMBEDDING_PROVIDER từ .env (os.getenv("EMBEDDING_PROVIDER", "sentence_transformers"))
    để cả nhóm có thể đổi provider mà không sửa code — nhớ đổi provider phải xoá
    chroma_db/ cũ và reindex vì dimension khác nhau (1024/768/1536) không tương thích ngược.

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# TODO: Chọn chunking strategy và giải thích vì sao

# Kích thước tối đa của mỗi chunk là 500 ký tự.
# 500 là mức cân bằng giữa:
# - Đủ ngữ cảnh để AI hiểu một ý hoàn chỉnh (ví dụ: chính sách đổi trả).
# - Không quá dài khiến embedding bị “loãng” vì chứa nhiều ý khác nhau.
# - Khi tìm kiếm, hệ thống trả về đoạn ngắn và đúng trọng tâm hơn.
CHUNK_SIZE = 500        # Vì sao chọn 500? ...

# Hai chunk liên tiếp sẽ lặp lại 50 ký tự ở phần giao nhau.
# Overlap giúp tránh mất ngữ cảnh khi một câu, điều kiện hoặc ý quan trọng
# nằm đúng tại ranh giới giữa hai chunk.
# 50/500 = 10%, thường đủ để giữ tính liên tục mà không tạo quá nhiều dữ liệu trùng lặp.
CHUNK_OVERLAP = 50      # Vì sao chọn 50? ...  

# Dùng Recursive Chunking:
# Hệ thống ưu tiên tách văn bản theo cấu trúc tự nhiên:
# đoạn văn -> dòng -> câu -> từ -> ký tự.
# Cách này phù hợp với tài liệu hỗn hợp như FAQ, chính sách hoàn tiền,
# hướng dẫn vận chuyển và mô tả sản phẩm.
# Nó ổn định, đơn giản, không cần gọi LLM nên chi phí và thời gian xử lý thấp.
#
# Các lựa chọn khác:
# - markdown_header: phù hợp khi tài liệu Markdown có cấu trúc tiêu đề rõ ràng.
# - semantic: chia theo ngữ nghĩa, có thể chính xác hơn nhưng chậm và tốn tài nguyên hơn
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# TODO: Chọn embedding model và giải thích

# Model embedding BAAI/bge-m3 tạo vector biểu diễn ý nghĩa của văn bản.
# Chọn bge-m3 vì:
# - Hỗ trợ đa ngôn ngữ tốt, đặc biệt phù hợp khi chatbot nhận cả tiếng Việt và tiếng Anh.
# - Có chất lượng retrieval tốt cho RAG/search.
# - Có thể chạy local, không phụ thuộc API và không phát sinh chi phí theo token.
# - Phù hợp cho hệ thống e-commerce vì người dùng có thể tìm bằng nhiều cách diễn đạt:
#   "đổi hàng", "trả sản phẩm", "refund", "return policy", ...
EMBEDDING_MODEL = "BAAI/bge-m3"  # Vì sao? Multilingual, tốt cho tiếng Việt lẫn tiếng Anh

# Số chiều của vector được bge-m3 tạo ra là 1024.
# Vector có nhiều chiều hơn thường giữ được nhiều thông tin ngữ nghĩa hơn,
# nhưng cũng tốn thêm RAM/dung lượng lưu trữ và thời gian tìm kiếm.
EMBEDDING_DIM = 1024

# TODO: Chọn vector store

# Dùng ChromaDB để lưu trữ và tìm kiếm các embedding vector.
# ChromaDB phù hợp cho giai đoạn học tập, prototype hoặc MVP vì:
# - Dễ cài đặt và tích hợp với Python/LangChain.
# - Có thể chạy local, không cần triển khai server riêng.
# - Lưu được vector, nội dung chunk và metadata (nguồn tài liệu, tiêu đề, loại chính sách...).
# - Hỗ trợ similarity search để lấy các chunk liên quan nhất với câu hỏi người dùng.
#
# Các lựa chọn khác:
# - FAISS: rất nhanh, nhẹ, tốt khi chỉ cần tìm vector local; metadata/persistence cần tự xử lý thêm.
# - Weaviate: mạnh hơn cho production quy mô lớn, nhưng phức tạp hơn vì thường cần chạy server/Docker.
VECTOR_STORE = "chromadb"  # "chromadb" | "weaviate" | "faiss"

# Tên collection trong ChromaDB.
# Collection có thể hiểu như một bảng/nhóm dữ liệu vector.
# Collection này chỉ chứa các chunk từ tài liệu hỗ trợ khách hàng e-commerce:
# FAQ, chính sách giao hàng, hoàn tiền, đổi trả, thanh toán, bảo hành, ...
COLLECTION_NAME = "ecommerce_support_docs"

# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    # TODO: Iterate qua STANDARDIZED_DIR, đọc .md files
    # documents = []
    # for md_file in STANDARDIZED_DIR.rglob("*.md"):
    #     content = md_file.read_text(encoding="utf-8")
    #     doc_type = "legal" if "legal" in str(md_file) else "news"
    #     documents.append({
    #         "content": content,
    #         "metadata": {"source": md_file.name, "type": doc_type}
    #     })
    # return documents
    # raise NotImplementedError("Implement load_documents")

    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8").strip()

        if not content:
            continue

        doc_type = "legal" if md_file.parent.name == "legal" else "news"

        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type,
                },
            }
        )

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    # TODO: Implement chunking
    #
    # Ví dụ với RecursiveCharacterTextSplitter:
    # from langchain_text_splitters import RecursiveCharacterTextSplitter
    #
    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=CHUNK_SIZE,
    #     chunk_overlap=CHUNK_OVERLAP,
    #     separators=["\n\n", "\n", ". ", " ", ""]
    # )
    # chunks = []
    # for doc in documents:
    #     splits = splitter.split_text(doc["content"])
    #     for i, chunk_text in enumerate(splits):
    #         chunks.append({
    #             "content": chunk_text,
    #             "metadata": {**doc["metadata"], "chunk_index": i}
    #         })
    # return chunks
    # raise NotImplementedError("Implement chunk_documents")

    """Chia từng document thành chunks có metadata."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for doc in documents:
        splits = splitter.split_text(doc["content"])

        for chunk_index, chunk_text in enumerate(splits):
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": chunk_index,
                    },
                }
            )

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    # TODO: Implement embedding
    #
    # Ví dụ với sentence-transformers (local, mặc định):
    # from sentence_transformers import SentenceTransformer
    #
    # model = SentenceTransformer(EMBEDDING_MODEL)
    # texts = [c["content"] for c in chunks]
    # embeddings = model.encode(texts, show_progress_bar=True)
    # for chunk, emb in zip(chunks, embeddings):
    #     chunk["embedding"] = emb.tolist()
    # return chunks
    # Nâng cao (optional): nếu muốn cho cả nhóm chọn được provider qua .env, viết
    # 1 hàm embed_texts(texts) dispatch theo os.getenv("EMBEDDING_PROVIDER") sang
    # sentence-transformers | Google (genai.embed_content) | OpenAI (client.embeddings.create)
    # rồi gọi lại hàm đó ở đây và ở Task 5 — tránh viết logic embed lặp lại 2 nơi.

    """Tạo embedding cho toàn bộ chunks."""
    from sentence_transformers import SentenceTransformer

    if not chunks:
        return []

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [chunk["content"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()

    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    # TODO: Implement indexing
    #
    # Ví dụ với ChromaDB:
    # import chromadb
    #
    # CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    # client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # collection = client.get_or_create_collection(
    #     name=COLLECTION_NAME,
    #     metadata={"hnsw:space": "cosine"},
    # )
    #
    # ids = [f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}" for c in chunks]
    # collection.upsert(
    #     ids=ids,
    #     documents=[c["content"] for c in chunks],
    #     embeddings=[c["embedding"] for c in chunks],
    #     metadatas=[c["metadata"] for c in chunks],
    # )
    # raise NotImplementedError("Implement index_to_vectorstore")

    """Lưu chunks và embeddings vào persistent ChromaDB."""
    import chromadb

    if not chunks:
        print("No chunks to index.")
        return

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [
        f"{chunk['metadata']['source']}_chunk_{chunk['metadata']['chunk_index']}"
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
