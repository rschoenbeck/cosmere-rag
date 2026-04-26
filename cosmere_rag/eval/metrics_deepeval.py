"""DeepEval LLM-judge metrics over RetrievalResults.

DeepEval is imported lazily so importing the rest of the eval package
(e.g. for IR-only runs or unit tests) doesn't pay deepeval's import
cost. The wrapper builds one `LLMTestCase` per query and runs the
selected reference-grounded retrieval metrics:

  - ContextualRelevancyMetric — does each retrieved chunk address the query?
  - ContextualPrecisionMetric — are relevant chunks ranked above irrelevant ones?
  - ContextualRecallMetric    — do retrieved chunks cover the expected answer?

We do not evaluate generation here — `actual_output` is set to the
expected answer as a stub since these metrics score `retrieval_context`.
"""
from __future__ import annotations

from collections.abc import Sequence

from cosmere_rag.eval.runner import RetrievalResult

DEFAULT_DEEPEVAL_METRICS = ("contextual_relevancy", "contextual_precision", "contextual_recall")


def aggregate_deepeval_metrics(
    results: Sequence[RetrievalResult],
    metrics: Sequence[str] = DEFAULT_DEEPEVAL_METRICS,
    judge_model: str = "gpt-5.4-mini",
    threshold: float = 0.5,
) -> dict[str, float]:
    if not results:
        return {}

    from deepeval.metrics import (
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
    )
    from deepeval.test_case import LLMTestCase

    metric_factories = {
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

    unknown = set(metrics) - set(metric_factories)
    if unknown:
        raise ValueError(f"unknown DeepEval metrics: {sorted(unknown)}")

    totals: dict[str, float] = {name: 0.0 for name in metrics}
    counts: dict[str, int] = {name: 0 for name in metrics}

    for r in results:
        test_case = LLMTestCase(
            input=r.query,
            actual_output=r.expected_answer,
            expected_output=r.expected_answer,
            retrieval_context=list(r.retrieved_texts),
        )
        for name in metrics:
            metric = metric_factories[name]()
            metric.measure(test_case)
            score = metric.score
            if score is None:
                continue
            totals[name] += float(score)
            counts[name] += 1

    return {
        name: (totals[name] / counts[name]) if counts[name] > 0 else 0.0
        for name in metrics
    }
