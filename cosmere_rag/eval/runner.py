"""Run a Retriever over a golden set and capture per-query results.

Sits between the golden set and the metric modules: every metric (IR or
LLM-judge) reads from `RetrievalResult` rather than calling the
retriever itself, so a single retrieval pass feeds both tracks.
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


def _build_where(query: EvalQuery) -> Mapping[str, Any] | None:
    where: dict[str, Any] = {}
    if query.spoiler_scope:
        where["spoiler_scope"] = query.spoiler_scope
    if query.series_filter:
        values = list(query.series_filter)
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
        retrieved = retriever.query(vec, k=k, where=_build_where(q))
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
