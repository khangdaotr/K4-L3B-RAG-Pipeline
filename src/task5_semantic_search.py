"""Dense retrieval over the Task 4 Chroma collection."""

from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Return unique dense SearchResults ordered by cosine similarity."""
    if top_k <= 0 or not query.strip():
        return []
    collection = get_collection()
    response = collection.query(
        query_embeddings=[embed_texts([query])[0]],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    by_id: dict[str, dict] = {}
    for item_id, content, metadata, distance in zip(
        response["ids"][0],
        response["documents"][0],
        response["metadatas"][0],
        response["distances"][0],
    ):
        result = {
            "id": item_id,
            "content": content,
            "score": float(1.0 - distance),
            "metadata": metadata,
            "retrieval_method": "dense",
        }
        previous = by_id.get(item_id)
        if previous is None or result["score"] > previous["score"]:
            by_id[item_id] = result
    return sorted(by_id.values(), key=lambda item: (-item["score"], item["id"]))[:top_k]


if __name__ == "__main__":
    for result in semantic_search("accessibility statement feedback mechanism", top_k=3):
        print(result["score"], result["id"], result["metadata"])
