"""Hand-computed checks for the IR metrics.

The cases here are small enough to verify with pencil and paper:
catches off-by-one errors in the ranking math (especially log2 indexing
in NDCG) and confirms aggregation averages over only labeled queries.
"""
from __future__ import annotations

import math

import pytest

from cosmere_rag.eval.metrics_ir import (
    aggregate_ir_metrics,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from cosmere_rag.eval.runner import RetrievalResult


def _result(retrieved: list[str], relevant: list[str]) -> RetrievalResult:
    return RetrievalResult(
        query_id="q",
        query="?",
        expected_answer="",
        relevant_chunk_ids=relevant,
        retrieved_chunk_ids=retrieved,
        scores=[0.0] * len(retrieved),
        retrieved_texts=[""] * len(retrieved),
    )


def test_precision_at_k_counts_hits_in_top_k():
    assert precision_at_k(["a", "b", "c", "d"], {"a", "c"}, k=4) == 0.5
    assert precision_at_k(["a", "b", "c", "d"], {"a", "c"}, k=2) == 0.5
    assert precision_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0


def test_recall_at_k_uses_total_relevant_count():
    assert recall_at_k(["a", "b", "c"], {"a", "b", "x"}, k=3) == pytest.approx(2 / 3)
    assert recall_at_k(["a"], {"a", "b"}, k=1) == 0.5


def test_reciprocal_rank_uses_first_hit():
    assert reciprocal_rank(["x", "a", "b"], {"a"}) == 0.5
    assert reciprocal_rank(["a", "x"], {"a"}) == 1.0
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_ndcg_at_k_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_handcomputed():
    # retrieved: [a, x, b]  relevant: {a, b}
    # DCG = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3) = 1 + 1/log2(3)
    dcg = 1.0 + 1.0 / math.log2(4)
    idcg = 1.0 + 1.0 / math.log2(3)
    assert ndcg_at_k(["a", "x", "b"], {"a", "b"}, k=3) == pytest.approx(dcg / idcg)


def test_ndcg_at_k_no_hits_is_zero():
    assert ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_metrics_reject_empty_relevant_set():
    with pytest.raises(ValueError):
        precision_at_k(["a"], set(), k=1)
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), k=1)
    with pytest.raises(ValueError):
        reciprocal_rank(["a"], set())
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], set(), k=1)


def test_metrics_reject_non_positive_k():
    with pytest.raises(ValueError):
        precision_at_k(["a"], {"a"}, k=0)
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a"}, k=-1)
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], {"a"}, k=0)


def test_aggregate_skips_results_without_labels():
    labeled = _result(["a", "b"], ["a"])
    unlabeled = _result(["a", "b"], [])
    summary = aggregate_ir_metrics([labeled, unlabeled], k=2)
    assert summary["precision@2"] == 0.5
    assert summary["recall@2"] == 1.0
    assert summary["mrr"] == 1.0
    assert summary["ndcg@2"] == pytest.approx(1.0)


def test_aggregate_returns_empty_when_no_labels():
    assert aggregate_ir_metrics([_result(["a"], [])], k=1) == {}
