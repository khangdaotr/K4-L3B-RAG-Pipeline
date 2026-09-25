# RAG evaluation results

## Run information

| Field | Value |
| --- | --- |
| Evaluation date | 2026-09-26 |
| Framework and version | Project A/B runner (`scripts/evaluate_ab.py`), `google-genai 2.23.0`; fixed LLM-as-judge rubric |
| Evaluator model | `gemini-3.1-flash-lite-preview` |
| Generator model | `gemini-3.1-flash-lite-preview` |
| Embedding model | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, normalized 384-dimensional vectors |
| Corpus version/commit | Git `24d61ff` plus working-tree corpus: 3 legal PDFs + 5 public guidance pages, 1,157 chunks |
| Golden dataset size | 15 grounded cases |
| `top_k` | 5 for both configurations |
| Fallback threshold and calibration | Product threshold `0.30`; fallback disabled in this A/B run so the isolated variable is retrieval strategy |

Raw answers, retrieved contexts, source IDs, per-case scores and latency are in
[`evaluation_results.json`](evaluation_results.json). The runner uses the same
golden data, generator, evaluator, system prompt and `top_k` for both arms.
The judge scores each metric from 0 to 1 in increments of 0.05. This is one
LLM-judged run without confidence intervals; scores are diagnostic rather than
proof of general superiority.

## Configurations

- **Config A — dense-only:** top 5 cosine results from the shared multilingual embedding model and Chroma cosine collection.
- **Config B — hybrid + RRF:** dense top 10 and BM25 top 10, fused exactly once with RRF (`k=60`), truncated to top 5.

No PageIndex fallback was used during A/B evaluation. Generation used the same
prompt and model after retrieval, so the intended experimental difference is
dense-only versus dense + BM25 + RRF.

## Overall scores

| Metric | Config A | Config B | Delta B−A |
| --- | ---: | ---: | ---: |
| Faithfulness | 0.8667 | 0.9333 | +0.0666 |
| Answer relevance | 0.8533 | 0.9233 | +0.0700 |
| Context recall | 0.8333 | 0.9667 | +0.1334 |
| Context precision | 0.8400 | 0.9667 | +0.1267 |
| **Average** | **0.8483** | **0.9475** | **+0.0992** |

## A/B comparison

- **Better configuration in this run:** Config B, hybrid + RRF.
- **Evidence:** Config B recovered full expected context for case 3 (three-year monitoring cycle) and case 15 (purpose of WCAG-EM), whereas dense-only missed or failed to use that evidence. Across the corpus-complete run, recall improved by 0.1334 and precision by 0.1267.
- **Trade-off:** mean end-to-end latency was 16.059 s for A and 5.607 s for B, but this counter-intuitive difference is dominated by variable Gemini response time, retries and sequential run order. It is not evidence that RRF is faster. Config B performs BM25 plus a larger dense candidate query, so it has slightly more local retrieval work; both arms make one generation call and pass five chunks.
- **Important-case check:** hybrid retrieved complete evidence for case 3 but the generator still refused, so retrieval gains do not automatically become answer gains. Case 1 remained partially complete under both arms because the requested statement fields were spread across legal chunks.

## Worst performers

| # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | How often must Member States report monitoring results? | A | 0.00 | 0.00 | 0.00 | 0.00 | Retrieval/generation | Dense top-5 mixed a generic monitoring chunk with distant Official Journal chunks; the interval was not presented clearly enough and the generator safely refused. |
| 2 | What is WCAG-EM used for? | A | 0.00 | 0.00 | 0.00 | 0.00 | Retrieval | Dense retrieval favored the WCAG overview and standards pages instead of the evaluation page containing the WCAG-EM definition. |
| 3 | How often must Member States report monitoring results? | B | 0.00 | 0.00 | 1.00 | 1.00 | Generation | Hybrid retrieved the expected reporting evidence with focused context, but the generator still returned a refusal; this is no longer a retrieval-recall failure. |

The refusals contain no unsupported claims, but case 3-B shows over-refusal:
the judge and manual source inspection found sufficient evidence in context.
Case 15-A remains a clear retrieval miss, while case 3 needs both retrieval
presentation and generation-policy investigation.

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| ---: | --- | --- | --- | --- |
| 1 | Keep hybrid + RRF and add regression checks for cases 3 and 15. | BM25/RRF raised context recall from 0.8333 to 0.9667 and recovered the WCAG-EM definition missed by dense-only. | Preserve exact-term and identifier coverage as the legal corpus grows. | Re-run `scripts/evaluate_ab.py`; require case 15 recall and answer relevance to remain 1.0 and inspect case 3 source IDs. |
| 2 | Add a generation regression test for evidence-present refusal and make the prompt distinguish “insufficient evidence” from “answer is stated in one source.” | Case 3-B had recall and precision 1.0 but faithfulness/relevance 0 because Gemini refused. | Convert retrieved evidence into an answer without weakening the safe-refusal behavior for genuine out-of-domain queries. | Replay case 3-B and the cake query; accept only if case 3 answers with valid citations while the cake query still returns `sources=[]` and `none`. |
| 3 | Improve context assembly for multi-part legal questions, using adjacent-chunk expansion or parent-section grouping with a strict token cap. | Both configs answered case 1 only partially because required statement fields were distributed across legal chunks. | Improve completeness for detailed obligations without broadly increasing `top_k` noise. | Re-run case 1 and require answer relevance/recall 1.0, then confirm overall context precision does not fall below 0.9667 for Config B. |

## Limitations and reproducibility

- The evaluator and generator use the same model family, which can introduce correlated judgment bias. A later run should add a second judge or human blind review.
- The legal branch includes the Directive, Decision 2018/1523, and Official Journal L 256 containing Decisions 2018/1523 and 2018/1524; the whole OJ creates many navigation/adjacent-act chunks that can affect precision.
- API latency varied substantially, so the single-run latency values should not guide architecture decisions. Measure retrieval and generation separately over repeated runs.
- Reproduce with `python -u scripts/evaluate_ab.py`; the script overwrites the raw result file using the fixed dataset and settings documented above.

## Bonus experiments

No bonus reranker experiment was claimed. The measured comparison is the required dense-only baseline against hybrid + RRF; no Jina or cross-encoder result is presented without a controlled measurement.
