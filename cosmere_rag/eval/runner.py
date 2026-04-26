"""Run a Retriever over a golden set and capture per-query results.

Used by the offline path (`cosmere-eval run --offline`) and by the
metric unit tests, which still consume `RetrievalResult`. The online
LangSmith path (`cosmere_rag/eval/experiment.py`) calls
`run_retrieval_chain` per example instead and does not go through here.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cosmere_rag.core.retrieved_chunk import RetrievedChunk
from cosmere_rag.core.retriever import Retriever
from cosmere_rag.embed.embedder import Embedder
from cosmere_rag.eval.dataset import EvalQuery


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    query_id: str
    query: str
    expected_answer: str
    relevant_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    scores: list[float]
    retrieved_texts: list[str]
    retrieved: list[RetrievedChunk] = Field(default_factory=list)


def build_where(
    *,
    spoiler_scope: str | None = None,
    series_filter: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    where: dict[str, Any] = {}
    if spoiler_scope:
        where["spoiler_scope"] = spoiler_scope
    if series_filter:
        values = list(series_filter)
        where["series_mentioned"] = values[0] if len(values) == 1 else {"$in": values}
    return where or None


def run_retrieval(
    retriever: Retriever,
    embedder: Embedder,
    queries: Sequence[EvalQuery],
    k: int = 8,
) -> list[RetrievalResult]:
    results: list[RetrievalResult] = []
    for q in queries:
        vec = embedder.embed_query(q.query)
        where: Mapping[str, Any] | None = build_where(
            spoiler_scope=q.spoiler_scope,
            series_filter=q.series_filter,
        )
        retrieved = retriever.query(vec, k=k, where=where)
        results.append(
            RetrievalResult(
                query_id=q.query_id,
                query=q.query,
                expected_answer=q.expected_answer,
                relevant_chunk_ids=list(q.relevant_chunk_ids),
                retrieved_chunk_ids=[r.chunk.chunk_id for r in retrieved],
                scores=[r.score for r in retrieved],
                retrieved_texts=[r.chunk.text for r in retrieved],
                retrieved=list(retrieved),
            )
        )
    return results
