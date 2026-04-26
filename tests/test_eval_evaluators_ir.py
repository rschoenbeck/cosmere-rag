"""LangSmith adapter checks for the IR evaluators.

The metric math itself is covered by `test_eval_metrics_ir.py`. These
tests focus on the wrapper contract: pulling `retrieved_chunk_ids` from
`run.outputs`, `relevant_chunk_ids` from `example.outputs`, returning
the right `key`, and skipping (returning `None`) when the example has
no labels.
"""
from __future__ import annotations

from types import SimpleNamespace

from cosmere_rag.eval.evaluators_ir import (
    all_ir_evaluators,
    mrr_evaluator,
    ndcg_at_k_evaluator,
    precision_at_k_evaluator,
    recall_at_k_evaluator,
)


def _run(retrieved: list[str]) -> SimpleNamespace:
    return SimpleNamespace(outputs={"retrieved_chunk_ids": retrieved})


def _example(relevant: list[str]) -> SimpleNamespace:
    return SimpleNamespace(outputs={"relevant_chunk_ids": relevant})


def test_precision_evaluator_returns_keyed_score():
    out = precision_at_k_evaluator(k=2)(_run(["a", "b", "c"]), _example(["a"]))
    assert out == {"key": "precision@2", "score": 0.5}


def test_recall_evaluator_uses_total_relevant():
    out = recall_at_k_evaluator(k=2)(_run(["a", "x"]), _example(["a", "b"]))
    assert out == {"key": "recall@2", "score": 0.5}


def test_mrr_evaluator_uses_first_hit():
    out = mrr_evaluator()(_run(["x", "a"]), _example(["a"]))
    assert out == {"key": "mrr", "score": 0.5}


def test_ndcg_evaluator_perfect_ranking():
    out = ndcg_at_k_evaluator(k=2)(_run(["a", "b"]), _example(["a", "b"]))
    assert out == {"key": "ndcg@2", "score": 1.0}


def test_evaluators_skip_unlabeled_example():
    run = _run(["a", "b"])
    unlabeled = _example([])
    assert precision_at_k_evaluator(k=2)(run, unlabeled) is None
    assert recall_at_k_evaluator(k=2)(run, unlabeled) is None
    assert mrr_evaluator()(run, unlabeled) is None
    assert ndcg_at_k_evaluator(k=2)(run, unlabeled) is None


def test_evaluators_handle_missing_outputs_dict():
    run = SimpleNamespace(outputs=None)
    example = SimpleNamespace(outputs=None)
    assert precision_at_k_evaluator(k=2)(run, example) is None


def test_all_ir_evaluators_emits_four_keys():
    run = _run(["a", "b", "c"])
    example = _example(["a", "c"])
    keys = [ev(run, example)["key"] for ev in all_ir_evaluators(k=3)]
    assert keys == ["precision@3", "recall@3", "mrr", "ndcg@3"]
