"""Hybrid retrieval with dense-score-based vectorless fallback."""

import os

from dotenv import load_dotenv

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

load_dotenv()


def _configured_threshold(default: float = 0.3) -> float:
    raw = os.getenv("SCORE_THRESHOLD", "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as error:
        raise ValueError("SCORE_THRESHOLD must be numeric") from error


SCORE_THRESHOLD = _configured_threshold()
DEFAULT_TOP_K = 5


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Retrieve hybrid results, falling back based only on dense cosine score."""
    if top_k <= 0 or not query.strip():
        return []

    candidate_k = top_k * 2
    dense = semantic_search(query, top_k=candidate_k)
    sparse = lexical_search(query, top_k=candidate_k)
    hybrid = (
        rerank_rrf([dense, sparse], top_k=top_k)
        if use_reranking
        else [item.copy() for item in dense[:top_k]]
    )

    best_dense_score = max((float(item["score"]) for item in dense), default=0.0)
    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback[:top_k]
        except Exception:
            # PageIndex is optional and external; preserve local retrieval.
            pass
    return hybrid[:top_k]


if __name__ == "__main__":
    query = "How must public bodies publish an accessibility statement?"
    for result in retrieve(query, top_k=3):
        print(result["score"], result["retrieval_method"], result["id"])
