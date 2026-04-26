from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from cosmere_rag.core.chunk import Chunk
from cosmere_rag.core.retrieved_chunk import RetrievedChunk
from cosmere_rag.embed.embedder import Embedder
from cosmere_rag.retrieval.chain import run_retrieval_chain


class _StubEmbeddingProvider:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class _StubRetriever:
    def __init__(self) -> None:
        self.last_args: dict[str, Any] = {}

    def add(self, chunks, embeddings) -> None:  # pragma: no cover
        raise NotImplementedError

    def count(self) -> int:  # pragma: no cover
        return 1

    def query(
        self,
        embedding: list[float],
        k: int = 8,
        where: Mapping[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        self.last_args = {"embedding": embedding, "k": k, "where": where}
        chunk = Chunk(
            chunk_id="c1",
            text="vin burns steel",
            article_title="Vin",
            heading_path=[],
            spoiler_scope="MB-Era1",
            series_mentioned=["Mistborn"],
            source_url="https://example/coppermind/Vin",
            content_provenance="coppermind",
            corpus_snapshot="2026-04-01",
            ingested_at=datetime.now(tz=timezone.utc),
            token_count=4,
        )
        return [RetrievedChunk(chunk=chunk, score=0.91)]


def test_run_retrieval_chain_shape_and_pass_through():
    retriever = _StubRetriever()
    embedder = Embedder(model="stub-model", provider=_StubEmbeddingProvider())

    out = run_retrieval_chain(
        "what does vin burn?",
        retriever=retriever,
        embedder=embedder,
        k=3,
        where={"spoiler_scope": "MB-Era1"},
        collection="stub-collection",
    )

    assert out["retrieved_chunk_ids"] == ["c1"]
    assert out["retrieved_texts"] == ["vin burns steel"]
    assert out["scores"] == [0.91]
    assert len(out["results"]) == 1

    assert retriever.last_args["k"] == 3
    assert retriever.last_args["where"] == {"spoiler_scope": "MB-Era1"}
    assert len(retriever.last_args["embedding"]) == 3
