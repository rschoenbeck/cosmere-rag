"""LangSmith experiment runner.

Wraps `langsmith.evaluate()` so a single call runs the retrieval chain
over a dataset, scores it with the IR evaluators (and three LLM-judge
contextual metrics when requested), and records everything as one
experiment in the LangSmith UI. Per-query drill-down replaces the
JSON/markdown reports the old `runner.py` + `report.py` pair produced.

The dataset side of the contract is set by
`cosmere_rag/eval/langsmith_dataset.py`:

    inputs  = {"query", "spoiler_scope", "series_filter"}
    outputs = {"expected_answer", "relevant_chunk_ids"}
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langsmith import evaluate
from langsmith.evaluation._runner import ExperimentResults

from cosmere_rag.core.retriever import Retriever
from cosmere_rag.embed.embedder import Embedder
from cosmere_rag.eval.config import DEFAULT_JUDGE_MODEL
from cosmere_rag.eval.evaluators_ir import all_ir_evaluators
from cosmere_rag.eval.evaluators_llm import llm_contextual_evaluator
from cosmere_rag.eval.runner import build_where
from cosmere_rag.retrieval.chain import run_retrieval_chain

Target = Callable[[dict[str, Any]], dict[str, Any]]


def build_target(
    *,
    retriever: Retriever,
    embedder: Embedder,
    k: int,
    collection: str | None = None,
) -> Target:
    """Closure that adapts a dataset example's inputs to `run_retrieval_chain`.

    `evaluate()` calls this once per example with the example's `inputs`
    dict and stores the returned dict as `run.outputs`.
    """

    def _target(inputs: dict[str, Any]) -> dict[str, Any]:
        return run_retrieval_chain(
            query=inputs["query"],
            retriever=retriever,
            embedder=embedder,
            k=k,
            where=build_where(
                spoiler_scope=inputs.get("spoiler_scope"),
                series_filter=inputs.get("series_filter"),
            ),
            collection=collection,
        )

    return _target


def run_experiment(
    *,
    retriever: Retriever,
    embedder: Embedder,
    dataset_name: str,
    k: int,
    collection: str | None = None,
    metrics: Sequence[str] = ("ir",),
    judge_model: str = DEFAULT_JUDGE_MODEL,
    experiment_prefix: str | None = None,
    max_concurrency: int = 0,
) -> ExperimentResults:
    """Run a LangSmith experiment over `dataset_name`.

    `metrics` selects which evaluator tracks to attach: `"ir"` (always
    cheap, no LLM calls) and/or `"llm"` (three contextual judge metrics,
    ~3 LLM calls per example). The IR track scores examples that have
    `relevant_chunk_ids`; the LLM track scores examples that have
    `expected_answer` and at least one retrieved chunk.
    """
    unknown = set(metrics) - {"ir", "llm"}
    if unknown:
        raise ValueError(f"unknown metric track(s): {sorted(unknown)}")

    evaluators: list[Callable[..., Any]] = []
    if "ir" in metrics:
        evaluators.extend(all_ir_evaluators(k))
    if "llm" in metrics:
        evaluators.append(llm_contextual_evaluator(judge_model=judge_model))

    target = build_target(
        retriever=retriever,
        embedder=embedder,
        k=k,
        collection=collection,
    )

    return evaluate(
        target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        metadata={
            "collection": collection,
            "embedding_model": embedder.model,
            "k": k,
            "metrics": list(metrics),
        },
        max_concurrency=max_concurrency,
    )
