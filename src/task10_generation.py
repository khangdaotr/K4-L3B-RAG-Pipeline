"""Grounded answer generation with traceable source citations."""

import os
import re

from dotenv import load_dotenv

from .task5_semantic_search import semantic_search
from .task9_retrieval_pipeline import SCORE_THRESHOLD, retrieve

load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()

SAFE_REFUSAL = "Tôi không thể xác minh thông tin này từ các nguồn hiện có."
SYSTEM_PROMPT = """Bạn là trợ lý nghiên cứu về khả năng tiếp cận số tại EU.
Chỉ trả lời bằng thông tin có trong context. Mỗi khẳng định thực tế phải có
trích dẫn dạng [S1], [S2] tương ứng với nhãn nguồn trong context. Không tạo URL,
nguồn hoặc citation mới. Nếu context không đủ bằng chứng, hãy trả lời chính xác:
"Tôi không thể xác minh thông tin này từ các nguồn hiện có."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Move higher-ranked chunks to context edges without mutating input."""
    if len(chunks) <= 2:
        return list(chunks)
    front = list(chunks[::2])
    back = list(chunks[1::2])
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Format chunks with stable citation, title, source, URL and ID labels."""
    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]
        url = metadata.get("url") or "N/A"
        parts.append(
            f"[S{index} | Title: {metadata['title']} | Source: {metadata['source']} | "
            f"URL: {url} | Chunk ID: {chunk['id']}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Dispatch to the configured provider and return plain response text."""
    if not LLM_MODEL:
        raise RuntimeError("LLM_MODEL is not configured")

    if LLM_PROVIDER == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        response = OpenAI(api_key=api_key).chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        text = response.choices[0].message.content or ""

    elif LLM_PROVIDER == "gemini":
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            ),
        )
        text = response.text or ""

    elif LLM_PROVIDER == "anthropic":
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured")
        response = Anthropic(api_key=api_key).messages.create(
            model=LLM_MODEL,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=1200,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        text = "".join(block.text for block in response.content if block.type == "text")
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")

    text = text.strip()
    if not text:
        raise RuntimeError("LLM provider returned an empty response")
    return text


def _has_dense_evidence(query: str) -> bool:
    dense = semantic_search(query, top_k=1)
    return bool(dense and float(dense[0]["score"]) >= SCORE_THRESHOLD)


def _valid_citations(answer: str, source_count: int) -> bool:
    citations = [int(value) for value in re.findall(r"\[S(\d+)\]", answer)]
    return bool(citations) and all(1 <= value <= source_count for value in citations)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Return a synchronized GenerationResult or a safe refusal."""
    if top_k <= 0 or not query.strip():
        return {"answer": SAFE_REFUSAL, "sources": [], "retrieval_source": "none"}

    chunks = retrieve(query, top_k=top_k)
    if not chunks or not _has_dense_evidence(query):
        return {"answer": SAFE_REFUSAL, "sources": [], "retrieval_source": "none"}

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"
    try:
        answer = call_llm(SYSTEM_PROMPT, user_message)
    except Exception:
        return {"answer": SAFE_REFUSAL, "sources": [], "retrieval_source": "none"}

    if SAFE_REFUSAL.casefold() in answer.casefold() or not _valid_citations(answer, len(reordered)):
        return {"answer": SAFE_REFUSAL, "sources": [], "retrieval_source": "none"}

    method = reordered[0]["retrieval_method"]
    retrieval_source = "pageindex" if method == "pageindex" else "hybrid"
    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    print(generate_with_citation("How must public bodies publish an accessibility statement?"))
