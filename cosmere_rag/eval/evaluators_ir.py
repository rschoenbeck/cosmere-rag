"""LangSmith evaluators for the IR metrics.

Each factory returns a `(run, example) -> {key, score}` callable that
LangSmith's `evaluate()` invokes once per example. The factories close
over `k` so the metric key embeds the cutoff (`precision@8` etc.) and
matches what the JSON/markdown reports used to print.

The retrieval contract this assumes:
  - `run.outputs["retrieved_chunk_ids"]` — ordered list (rank 1 first)
  - `example.outputs["relevant_chunk_ids"]` — unordered ground-truth ids

Examples without `relevant_chunk_ids` are skipped (return `None`),
matching the `aggregate_ir_metrics` behaviour of dropping unlabeled
queries from the mean rather than scoring them as zero.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cosmere_rag.eval.metrics_ir import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

Evaluator = Callable[[Any, Any], dict[str, Any] | None]


def precision_at_k_evaluator(k: int) -> Evaluator:
    def _evaluator(run: Any, example: Any) -> dict[str, Any] | None:
        retrieved, relevant = _extract(run, example)
        if not relevant:
            return None
        return {
            "key": f"precision@{k}",
            "score": precision_at_k(retrieved, relevant, k),
        }

    return _evaluator


def recall_at_k_evaluator(k: int) -> Evaluator:
    def _evaluator(run: Any, example: Any) -> dict[str, Any] | None:
        retrieved, relevant = _extract(run, example)
        if not relevant:
            return None
        return {
            "key": f"recall@{k}",
            "score": recall_at_k(retrieved, relevant, k),
        }

    return _evaluator


def mrr_evaluator() -> Evaluator:
    def _evaluator(run: Any, example: Any) -> dict[str, Any] | None:
        retrieved, relevant = _extract(run, example)
        if not relevant:
            return None
        return {"key": "mrr", "score": reciprocal_rank(retrieved, relevant)}

    return _evaluator


def ndcg_at_k_evaluator(k: int) -> Evaluator:
    def _evaluator(run: Any, example: Any) -> dict[str, Any] | None:
        retrieved, relevant = _extract(run, example)
        if not relevant:
            return None
        return {
            "key": f"ndcg@{k}",
            "score": ndcg_at_k(retrieved, relevant, k),
        }

    return _evaluator


def all_ir_evaluators(k: int) -> list[Evaluator]:
    """Convenience: the four IR evaluators wired up for a given k."""
    return [
        precision_at_k_evaluator(k),
        recall_at_k_evaluator(k),
        mrr_evaluator(),
        ndcg_at_k_evaluator(k),
    ]


def _extract(run: Any, example: Any) -> tuple[list[str], list[str]]:
    retrieved = (run.outputs or {}).get("retrieved_chunk_ids", [])
    relevant = (example.outputs or {}).get("relevant_chunk_ids", [])
    return list(retrieved), list(relevant)
