"""Retrieval evaluation harness.

Public entry points:
  - `EvalQuery`, `load_golden_set`, `save_golden_set` — curated query set I/O.
  - `RetrievalResult`, `run_retrieval` — execute a Retriever over the golden set.
  - `aggregate_ir_metrics` — classic IR metrics (precision@k, recall@k, MRR, NDCG@k).
  - `aggregate_deepeval_metrics` — DeepEval LLM-judge metrics (lazy-imported).
  - `EvalReport` — combined per-run report, writable as JSON + markdown.
"""
from cosmere_rag.eval.dataset import (
    EvalQuery,
    load_golden_set,
    save_golden_set,
)
from cosmere_rag.eval.report import EvalReport
from cosmere_rag.eval.runner import RetrievalResult, run_retrieval

__all__ = [
    "EvalQuery",
    "EvalReport",
    "RetrievalResult",
    "load_golden_set",
    "run_retrieval",
    "save_golden_set",
]
