"""LangSmith evaluator that wraps DeepEval's contextual metrics.

Phase 3 Option B: keep DeepEval's prompts under the hood and surface
its three retrieval-context metrics through LangSmith so experiment
results land in the same UI as the IR metrics. A follow-up PR
(Option A) replaces this with native LangChain judges and drops the
DeepEval dependency.

One evaluator builds the `LLMTestCase` once per example and emits
three feedback rows (`contextual_relevancy`, `contextual_precision`,
`contextual_recall`) so we don't pay the test-case construction cost
three times. DeepEval is imported lazily so importing this module
stays cheap for the IR-only path.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from cosmere_rag.eval.config import DEFAULT_JUDGE_MODEL

DEFAULT_DEEPEVAL_METRICS = (
    "contextual_relevancy",
    "contextual_precision",
    "contextual_recall",
)


def deepeval_contextual_evaluator(
    metrics: Sequence[str] = DEFAULT_DEEPEVAL_METRICS,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    threshold: float = 0.5,
) -> Callable[[Any, Any], dict[str, Any] | None]:
    """LangSmith evaluator returning the three DeepEval contextual scores.

    Returns `None` when the example lacks an `expected_answer` (DeepEval's
    precision/recall metrics are reference-grounded and have nothing to
    score against without it).
    """
    unknown = set(metrics) - set(DEFAULT_DEEPEVAL_METRICS)
    if unknown:
        raise ValueError(f"unknown DeepEval metrics: {sorted(unknown)}")

    def _evaluator(run: Any, example: Any) -> dict[str, Any] | None:
        query = (example.inputs or {}).get("query")
        expected_answer = (example.outputs or {}).get("expected_answer")
        retrieved_texts = (run.outputs or {}).get("retrieved_texts", [])
        if not query or not expected_answer or not retrieved_texts:
            return None

        from deepeval.metrics import (
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            ContextualRelevancyMetric,
        )
        from deepeval.test_case import LLMTestCase

        factories = {
            "contextual_relevancy": lambda: ContextualRelevancyMetric(
                threshold=threshold, model=judge_model, include_reason=False
            ),
            "contextual_precision": lambda: ContextualPrecisionMetric(
                threshold=threshold, model=judge_model, include_reason=False
            ),
            "contextual_recall": lambda: ContextualRecallMetric(
                threshold=threshold, model=judge_model, include_reason=False
            ),
        }

        test_case = LLMTestCase(
            input=query,
            actual_output=expected_answer,
            expected_output=expected_answer,
            retrieval_context=list(retrieved_texts),
        )

        results: list[dict[str, Any]] = []
        for name in metrics:
            metric = factories[name]()
            metric.measure(test_case)
            score = metric.score
            if score is None:
                continue
            results.append({"key": name, "score": float(score)})

        if not results:
            return None
        return {"results": results}

    return _evaluator
