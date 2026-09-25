"""Reproducible dense-only versus hybrid+RRF evaluation.

The generator, evaluator, prompt, golden data and top_k are identical between
configurations. Raw per-case outputs are checkpointed for audit and resume.
"""

import json
import re
import time
from pathlib import Path

from src.task10_generation import LLM_MODEL, SYSTEM_PROMPT, call_llm, format_context, reorder_for_llm
from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank_rrf

ROOT = Path(__file__).parent.parent
GOLDEN_PATH = ROOT / "group_project" / "evaluation" / "golden_dataset.json"
OUTPUT_PATH = ROOT / "group_project" / "evaluation" / "evaluation_results.json"
TOP_K = 5

EVALUATOR_PROMPT = """You are a strict RAG evaluator. Score each record from 0 to 1.
faithfulness: generated answer claims are supported by retrieved_contexts.
answer_relevance: generated answer directly resolves the question.
context_recall: retrieved_contexts contain the facts in expected_answer/context.
context_precision: retrieved chunks are focused and relevant rather than noise.
Use only increments of 0.05. Return only a JSON array with objects containing
case_id, faithfulness, answer_relevance, context_recall, context_precision,
and a concise failure_reason. Do not add Markdown."""


def retrieve_for_config(question: str, config: str) -> list[dict]:
    if config == "A":
        return semantic_search(question, top_k=TOP_K)
    dense = semantic_search(question, top_k=TOP_K * 2)
    sparse = lexical_search(question, top_k=TOP_K * 2)
    return rerank_rrf([dense, sparse], top_k=TOP_K)


def call_with_retry(system_prompt: str, message: str, attempts: int = 5) -> str:
    for attempt in range(1, attempts + 1):
        try:
            return call_llm(system_prompt, message)
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(attempt * 5)
    raise RuntimeError("unreachable")


def generate(question: str, chunks: list[dict]) -> tuple[str, list[dict]]:
    sources = reorder_for_llm(chunks)
    message = f"Context:\n{format_context(sources)}\n\nQuestion: {question}"
    return call_with_retry(SYSTEM_PROMPT, message), sources


def parse_json(text: str):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    return json.loads(cleaned)


def save(payload: dict) -> None:
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    payload = {
        "run": {
            "generator_model": LLM_MODEL,
            "evaluator_model": LLM_MODEL,
            "top_k": TOP_K,
            "config_a": "dense-only",
            "config_b": "dense + BM25 + one RRF fusion",
        },
        "cases": [],
    }

    for case_id, case in enumerate(golden, start=1):
        for config in ("A", "B"):
            started = time.perf_counter()
            chunks = retrieve_for_config(case["question"], config)
            answer, sources = generate(case["question"], chunks)
            payload["cases"].append(
                {
                    "case_id": case_id,
                    "config": config,
                    **case,
                    "answer": answer,
                    "contexts": [source["content"] for source in sources],
                    "source_ids": [source["id"] for source in sources],
                    "latency_seconds": round(time.perf_counter() - started, 3),
                }
            )
            save(payload)
            print(f"generated case={case_id:02d} config={config}", flush=True)

    for start in range(0, len(payload["cases"]), 5):
        batch = payload["cases"][start : start + 5]
        judge_input = [
            {
                "case_id": f"{item['case_id']}{item['config']}",
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "expected_context": item["expected_context"],
                "generated_answer": item["answer"],
                "retrieved_contexts": item["contexts"],
            }
            for item in batch
        ]
        judged = parse_json(
            call_with_retry(EVALUATOR_PROMPT, json.dumps(judge_input, ensure_ascii=False))
        )
        by_id = {str(item["case_id"]): item for item in judged}
        for item in batch:
            scores = by_id[f"{item['case_id']}{item['config']}"]
            for metric in ("faithfulness", "answer_relevance", "context_recall", "context_precision"):
                item[metric] = float(scores[metric])
            item["failure_reason"] = str(scores.get("failure_reason", ""))
        save(payload)
        print(f"judged records {start + 1}-{start + len(batch)}", flush=True)

    metrics = ("faithfulness", "answer_relevance", "context_recall", "context_precision")
    payload["summary"] = {}
    for config in ("A", "B"):
        rows = [item for item in payload["cases"] if item["config"] == config]
        payload["summary"][config] = {
            metric: round(sum(row[metric] for row in rows) / len(rows), 4) for metric in metrics
        }
        payload["summary"][config]["average"] = round(
            sum(payload["summary"][config][metric] for metric in metrics) / len(metrics), 4
        )
        payload["summary"][config]["mean_latency_seconds"] = round(
            sum(row["latency_seconds"] for row in rows) / len(rows), 3
        )
    save(payload)
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
