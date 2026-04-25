"""Classic IR metrics over (retrieved_ids, relevant_ids) pairs.

Each metric is a pure function on a single query's results. Aggregation
across a run is the arithmetic mean — queries are weighted equally.

Conventions:
  - `relevant_ids` are an unordered set of ground-truth chunk ids.
  - `retrieved_ids` are ordered; rank 1 is the top hit.
  - `k` truncates `retrieved_ids` before scoring.
  - Queries with `relevant_ids` empty raise — they shouldn't be in the
    golden set for IR scoring; use the LLM-judge track instead.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from cosmere_rag.eval.runner import RetrievalResult


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    relevant_set = set(relevant)
    _require_relevant(relevant_set)
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if cid in relevant_set)
    return hits / k


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    relevant_set = set(relevant)
    _require_relevant(relevant_set)
    hits = sum(1 for cid in retrieved[:k] if cid in relevant_set)
    return hits / len(relevant_set)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    relevant_set = set(relevant)
    _require_relevant(relevant_set)
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Binary-relevance NDCG@k.

    DCG = sum_{i=1..k} rel_i / log2(i + 1) where rel_i ∈ {0, 1}.
    Ideal DCG assumes min(k, |relevant|) relevant hits at the top ranks.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    relevant_set = set(relevant)
    _require_relevant(relevant_set)
    dcg = 0.0
    for i, cid in enumerate(retrieved[:k], start=1):
        if cid in relevant_set:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(k, len(relevant_set))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate_ir_metrics(
    results: Sequence[RetrievalResult],
    k: int,
) -> dict[str, float]:
    """Mean of each metric over results that have labeled relevant ids."""
    scored = [r for r in results if r.relevant_chunk_ids]
    if not scored:
        return {}
    n = len(scored)
    totals = {
        f"precision@{k}": 0.0,
        f"recall@{k}": 0.0,
        "mrr": 0.0,
        f"ndcg@{k}": 0.0,
    }
    for r in scored:
        totals[f"precision@{k}"] += precision_at_k(
            r.retrieved_chunk_ids, r.relevant_chunk_ids, k
        )
        totals[f"recall@{k}"] += recall_at_k(
            r.retrieved_chunk_ids, r.relevant_chunk_ids, k
        )
        totals["mrr"] += reciprocal_rank(r.retrieved_chunk_ids, r.relevant_chunk_ids)
        totals[f"ndcg@{k}"] += ndcg_at_k(
            r.retrieved_chunk_ids, r.relevant_chunk_ids, k
        )
    return {name: value / n for name, value in totals.items()}


def per_query_ir_metrics(
    result: RetrievalResult,
    k: int,
) -> dict[str, float]:
    if not result.relevant_chunk_ids:
        return {}
    return {
        f"precision@{k}": precision_at_k(
            result.retrieved_chunk_ids, result.relevant_chunk_ids, k
        ),
        f"recall@{k}": recall_at_k(
            result.retrieved_chunk_ids, result.relevant_chunk_ids, k
        ),
        "mrr": reciprocal_rank(
            result.retrieved_chunk_ids, result.relevant_chunk_ids
        ),
        f"ndcg@{k}": ndcg_at_k(
            result.retrieved_chunk_ids, result.relevant_chunk_ids, k
        ),
    }


def _require_relevant(relevant_set: set[str]) -> None:
    if not relevant_set:
        raise ValueError(
            "IR metrics require at least one relevant chunk id; "
            "queries without labels should be skipped or scored via the LLM-judge track"
        )
