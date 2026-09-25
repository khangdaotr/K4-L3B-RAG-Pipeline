"""Reciprocal Rank Fusion for dense and lexical ranked lists."""


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse rankings by ID without mutating any upstream SearchResult."""
    if top_k <= 0:
        return []
    if k < 0:
        raise ValueError("k must be non-negative")

    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    first_seen: dict[str, int] = {}
    sequence = 0
    for ranked_list in ranked_lists:
        ids = [item["id"] for item in ranked_list]
        if len(ids) != len(set(ids)):
            raise ValueError("Each upstream ranked list must contain unique IDs")
        for rank, item in enumerate(ranked_list, start=1):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            if item_id not in items:
                items[item_id] = item
                first_seen[item_id] = sequence
                sequence += 1

    ranked_ids = sorted(scores, key=lambda item_id: (-scores[item_id], first_seen[item_id]))
    results: list[dict] = []
    for item_id in ranked_ids[:top_k]:
        result = items[item_id].copy()
        result["score"] = scores[item_id]
        result["retrieval_method"] = "hybrid"
        results.append(result)
    return results


if __name__ == "__main__":
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search

    query = "Directive 2016/2102 accessibility statement"
    dense = semantic_search(query, top_k=6)
    sparse = lexical_search(query, top_k=6)
    for result in rerank_rrf([dense, sparse], top_k=3):
        print(result["score"], result["id"], result["metadata"]["source"])
