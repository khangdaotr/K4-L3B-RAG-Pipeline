"""BM25 retrieval over the same Task 4 chunk corpus."""

import re

from src.task4_chunking_indexing import chunk_documents, load_documents

CORPUS: list[dict] = []
_BM25 = None
_BM25_CORPUS_ID: int | None = None


def _tokenize(text: str) -> list[str]:
    # Preserve legal identifiers such as 2016/2102 and EN 301 549.
    return re.findall(r"(?u)\b[\w]+(?:[/.-][\w]+)*\b", text.casefold())


def build_bm25_index(corpus: list[dict]):
    """Build a BM25 index from Task 4 chunks."""
    from rank_bm25 import BM25Okapi

    return BM25Okapi([_tokenize(item["content"]) for item in corpus])


def _active_corpus() -> list[dict]:
    global CORPUS
    if not CORPUS:
        CORPUS = chunk_documents(load_documents())
    return CORPUS


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Return unique BM25 SearchResults ordered by decreasing score."""
    if top_k <= 0 or not query.strip():
        return []
    corpus = _active_corpus()
    if not corpus:
        return []

    global _BM25, _BM25_CORPUS_ID
    if _BM25 is None or _BM25_CORPUS_ID != id(corpus):
        _BM25 = build_bm25_index(corpus)
        _BM25_CORPUS_ID = id(corpus)

    query_tokens = _tokenize(query)
    query_set = set(query_tokens)
    scores = _BM25.get_scores(query_tokens)
    ranked: list[tuple[float, int, str, dict]] = []
    for item, raw_score in zip(corpus, scores):
        overlap = len(query_set.intersection(_tokenize(item["content"])))
        if overlap:
            ranked.append((float(raw_score), overlap, item["id"], item))
    ranked.sort(key=lambda row: (-row[0], -row[1], row[2]))

    results: list[dict] = []
    seen: set[str] = set()
    for score, _, _, item in ranked:
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        results.append(
            {
                "id": item["id"],
                "content": item["content"],
                "score": score,
                "metadata": item["metadata"],
                "retrieval_method": "bm25",
            }
        )
        if len(results) == top_k:
            break
    return results


if __name__ == "__main__":
    for result in lexical_search("Directive 2016/2102 accessibility", top_k=3):
        print(result["score"], result["id"], result["metadata"])
