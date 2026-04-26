"""LangChain LLM judges for the three contextual retrieval metrics.

Phase 3 Option A — replaces the DeepEval wrapper. Each metric is a single
`ChatOpenAI` call at temperature 0 with structured output:

  - contextual_relevancy:  do retrieved chunks address the query
  - contextual_precision:  are relevant chunks ranked above irrelevant ones
  - contextual_recall:     do retrieved chunks cover the claims in the
                           expected answer

One LangSmith evaluator runs all three (three LLM calls per example) and
emits a multi-result feedback envelope so `experiment.py`'s wiring is
unchanged from the DeepEval era. `langchain_openai` is imported lazily
so the IR-only path doesn't pay for it.

Rubric design — and why scores run higher than DeepEval

Each metric is judged in a single shot over the *whole* retrieved set,
not per-statement or per-chunk. That makes the judges cheaper (3 LLM
calls per example vs. ~3 * num_chunks in DeepEval) but inflates
absolute scores compared to the DeepEval baseline:

  - relevancy: DeepEval splits chunks into atomic statements and scores
    `relevant_statements / total_statements`; a wiki paragraph with one
    on-topic sentence in ten scores ~0.10. Our judge rates each chunk
    holistically, so a chunk with any on-topic content reads as
    "relevant" and the score skews up (~0.33 -> ~0.88 on era1).
  - precision: DeepEval position-weights per-chunk verdicts, so even a
    perfect top-1 takes a hit when chunks at the bottom are judged
    irrelevant. Our judge collapses ranking quality into one 0-1 score
    and tends to round toward 1.0 when the top results are clearly good
    (~0.89 -> ~1.00 on era1).
  - recall: closest match to DeepEval (both ask roughly "do the chunks
    support the expected answer's claims") so scores stay comparable.

Treat the new numbers as a regression signal in their own right — useful
for "did retrieval get worse," not for absolute parity with the
DeepEval-era reports. If parity matters later, switch relevancy and
precision to per-chunk loops and compute the weighted-precision formula
in Python.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from cosmere_rag.eval.config import DEFAULT_JUDGE_MODEL

DEFAULT_LLM_METRICS = (
    "contextual_relevancy",
    "contextual_precision",
    "contextual_recall",
)


class _Verdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reason: str


_RUBRICS: dict[str, str] = {
    "contextual_relevancy": (
        "You are scoring CONTEXTUAL RELEVANCY for a retrieval system. "
        "Given a user query and a set of retrieved chunks, score the "
        "fraction of chunks that are directly relevant to answering the "
        "query. 1.0 means every chunk is on-topic; 0.0 means none are. "
        "Ignore the expected answer for this metric — judge each chunk "
        "against the query alone. Return the score and a one-sentence "
        "reason."
    ),
    "contextual_precision": (
        "You are scoring CONTEXTUAL PRECISION for a retrieval system. "
        "Given a user query, an expected answer, and an ordered list of "
        "retrieved chunks, score how well the most relevant chunks "
        "(those needed to produce the expected answer) are ranked above "
        "the irrelevant ones. 1.0 means all relevant chunks come first; "
        "0.0 means relevant chunks are buried at the bottom or absent. "
        "Return the score and a one-sentence reason."
    ),
    "contextual_recall": (
        "You are scoring CONTEXTUAL RECALL for a retrieval system. "
        "Given a user query, an expected answer, and the retrieved "
        "chunks, score the fraction of distinct claims in the expected "
        "answer that are supported by at least one retrieved chunk. "
        "1.0 means every claim in the expected answer can be grounded "
        "in the chunks; 0.0 means none can. Return the score and a "
        "one-sentence reason."
    ),
}


def _format_chunks(chunks: Sequence[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))


def _render_messages(
    rubric: str, query: str, retrieved: Sequence[str], expected: str
) -> list[tuple[str, str]]:
    user = (
        f"Query:\n{query}\n\n"
        f"Expected answer:\n{expected}\n\n"
        f"Retrieved chunks (ranked):\n{_format_chunks(retrieved)}"
    )
    return [("system", rubric), ("user", user)]


def llm_contextual_evaluator(
    metrics: Sequence[str] = DEFAULT_LLM_METRICS,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> Callable[[Any, Any], dict[str, Any] | None]:
    """LangSmith evaluator returning the three contextual judge scores.

    Returns `None` when the example lacks a `query`, `expected_answer`,
    or any retrieved chunks — these metrics are reference-grounded and
    have nothing to score against without all three.
    """
    unknown = set(metrics) - set(DEFAULT_LLM_METRICS)
    if unknown:
        raise ValueError(f"unknown LLM metrics: {sorted(unknown)}")

    def _evaluator(run: Any, example: Any) -> dict[str, Any] | None:
        query = (example.inputs or {}).get("query")
        expected_answer = (example.outputs or {}).get("expected_answer")
        retrieved_texts = (run.outputs or {}).get("retrieved_texts", [])
        if not query or not expected_answer or not retrieved_texts:
            return None

        from langchain_openai import ChatOpenAI

        judge = ChatOpenAI(
            model=judge_model, temperature=0
        ).with_structured_output(_Verdict)

        results: list[dict[str, Any]] = []
        for name in metrics:
            verdict = judge.invoke(
                _render_messages(
                    _RUBRICS[name], query, retrieved_texts, expected_answer
                )
            )
            results.append(
                {
                    "key": name,
                    "score": float(verdict.score),
                    "comment": verdict.reason,
                }
            )
        return {"results": results}

    return _evaluator
