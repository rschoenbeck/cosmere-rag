"""Retrieval evaluation harness.

Public entry points:
  - `EvalQuery`, `load_golden_set`, `save_golden_set` — curated query set I/O.
  - `RetrievalResult`, `run_retrieval`, `build_where` — offline retrieval pass
    (still used by `--offline` mode and the IR metric module).
  - `aggregate_ir_metrics` — classic IR metrics (precision@k, recall@k, MRR, NDCG@k).
  - `run_experiment` — LangSmith `evaluate()` wrapper for the online path
    (attaches IR + LLM-judge contextual evaluators).
"""
from cosmere_rag.eval.dataset import (
    EvalQuery,
    load_golden_set,
    save_golden_set,
)
from cosmere_rag.eval.runner import RetrievalResult, build_where, run_retrieval

__all__ = [
    "EvalQuery",
    "RetrievalResult",
    "build_where",
    "load_golden_set",
    "run_retrieval",
    "save_golden_set",
]
