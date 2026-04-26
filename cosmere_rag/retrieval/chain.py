"""Top-level retrieval chain.

One traceable parent run per user query — embed and retriever spans
nest underneath it. The output shape (`retrieved_chunk_ids`,
`retrieved_texts`, `scores`) is the contract Phase 4's evaluators
will consume.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from cosmere_rag.core.retriever import Retriever
from cosmere_rag.embed.embedder import Embedder


@traceable(run_type="chain", name="retrieval_chain")
def run_retrieval_chain(
    query: str,
    *,
    retriever: Retriever,
    embedder: Embedder,
    k: int = 8,
    where: Mapping[str, Any] | None = None,
    collection: str | None = None,
) -> dict[str, Any]:
    run = get_current_run_tree()
    if run is not None:
        run.add_metadata(
            {
                "embedding_model": embedder.model,
                "collection": collection,
                "k": k,
                "where": dict(where) if where else None,
            }
        )

    embedding = embedder.embed_query(query)
    results = retriever.query(embedding, k=k, where=where)

    return {
        "retrieved_chunk_ids": [r.chunk.chunk_id for r in results],
        "retrieved_texts": [r.chunk.text for r in results],
        "scores": [r.score for r in results],
        "results": results,
    }
