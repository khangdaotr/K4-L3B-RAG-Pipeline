"""Load, chunk, embed and idempotently index the normalized corpus."""

import re
from pathlib import Path

from src.contracts import validate_document

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# Multilingual local model: useful for Vietnamese questions over English sources.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
COLLECTION_NAME = "rag_documents"

_EMBEDDING_MODEL_INSTANCE = None


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed corpus and queries through one shared, normalized local model."""
    if not texts:
        return []
    global _EMBEDDING_MODEL_INSTANCE
    if _EMBEDDING_MODEL_INSTANCE is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDING_MODEL_INSTANCE = SentenceTransformer(
            EMBEDDING_MODEL,
            local_files_only=True,
        )
    vectors = _EMBEDDING_MODEL_INSTANCE.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 32,
    )
    output = vectors.tolist()
    if any(len(vector) != EMBEDDING_DIM for vector in output):
        raise ValueError(f"Embedding dimension does not match {EMBEDDING_DIM}")
    return output


def get_collection():
    """Open the persistent cosine-distance Chroma collection."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _header_value(content: str, label: str) -> str | None:
    match = re.search(rf"^\*\*{re.escape(label)}:\*\*\s*(.+?)\s*$", content, re.MULTILINE)
    return match.group(1).strip() if match else None


def load_documents() -> list[dict]:
    """Read normalized Markdown into stable, contract-valid Documents."""
    documents: list[dict] = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        relative = path.relative_to(STANDARDIZED_DIR)
        if not relative.parts or relative.parts[0] not in {"legal", "news"}:
            continue
        content = path.read_text(encoding="utf-8").strip()
        title_match = re.search(r"^#\s+(.+?)\s*$", content, re.MULTILINE)
        document = {
            "id": relative.with_suffix("").as_posix(),
            "content": content,
            "metadata": {
                "source": relative.as_posix(),
                "title": title_match.group(1).strip() if title_match else path.stem.replace("_", " "),
                "doc_type": relative.parts[0],
                "url": _header_value(content, "Source"),
            },
        }
        validate_document(document)
        documents.append(document)
    ids = [document["id"] for document in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("Document IDs are not unique")
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Recursively split Documents while preserving source metadata."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks: list[dict] = []
    for document in documents:
        validate_document(document)
        for index, text in enumerate(splitter.split_text(document["content"])):
            chunk = {
                "id": f"{document['id']}::chunk-{index:05d}",
                "content": text.strip(),
                "metadata": {**document["metadata"], "chunk_index": index},
            }
            validate_document(chunk, require_chunk=True)
            chunks.append(chunk)
    ids = [chunk["id"] for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Chunk IDs are not unique")
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Return new chunk dictionaries with one embedding per chunk."""
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    if len(vectors) != len(chunks):
        raise ValueError("Embedding provider returned an unexpected vector count")
    return [{**chunk, "embedding": vector} for chunk, vector in zip(chunks, vectors)]


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert current chunks and remove stale IDs from earlier corpus versions."""
    collection = get_collection()
    current_ids = {chunk["id"] for chunk in chunks}
    existing_ids = set(collection.get(include=[])["ids"])
    stale_ids = sorted(existing_ids - current_ids)
    if stale_ids:
        collection.delete(ids=stale_ids)
    if not chunks:
        return
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[
            {key: ("" if value is None else value) for key, value in chunk["metadata"].items()}
            for chunk in chunks
        ],
    )


def run_pipeline() -> None:
    documents = load_documents()
    if not documents:
        raise RuntimeError("No Markdown corpus found under data/standardized")
    chunks = chunk_documents(documents)
    index_to_vectorstore(embed_chunks(chunks))
    print(f"Indexed {len(chunks)} chunks from {len(documents)} documents")


if __name__ == "__main__":
    run_pipeline()
