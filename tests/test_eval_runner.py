"""Runner integration test against a fake retriever and embedder.

Verifies the runner threads filters, k, and embeddings through the
Retriever protocol correctly, and that RetrievalResult preserves the
ordering returned by the store.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from cosmere_rag.core.chunk import Chunk
from cosmere_rag.core.retrieved_chunk import RetrievedChunk
from cosmere_rag.embed.embedder import Embedder
from cosmere_rag.eval.baselines import NoiseEmbedder
from cosmere_rag.eval.dataset import EvalQuery
from cosmere_rag.eval.runner import run_retrieval


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        article_title="Vin",
        heading_path=["Allomancy"],
        spoiler_scope="MB-Era1",
        series_mentioned=["MB-Era1"],
        source_url="https://coppermind.net/wiki/Vin",
        content_provenance="coppermind-mirror",
        corpus_snapshot="abc",
        ingested_at=datetime(2026, 4, 22, tzinfo=timezone.utc),
        token_count=10,
    )


class FakeRetriever:
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        self.calls: list[dict[str, Any]] = []

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[list[float]]) -> None:
        raise NotImplementedError

    def query(
        self,
        embedding: list[float],
        k: int = 8,
        where: Mapping[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append({"embedding": embedding, "k": k, "where": where})
        return [
            RetrievedChunk(chunk=c, score=1.0 - 0.1 * i)
            for i, c in enumerate(self._chunks[:k])
        ]

    def count(self) -> int:
        return len(self._chunks)


def test_runner_returns_one_result_per_query():
    retriever = FakeRetriever([_chunk("a", "alpha"), _chunk("b", "beta")])
    embedder = Embedder(model="noise", provider=NoiseEmbedder(dim=8))
    queries = [
        EvalQuery(query_id="q1", query="alpha?", expected_answer="x"),
        EvalQuery(query_id="q2", query="beta?", expected_answer="y"),
    ]
    results = run_retrieval(retriever, embedder, queries, k=2)
    assert len(results) == 2
    assert results[0].query_id == "q1"
    assert results[0].retrieved_chunk_ids == ["a", "b"]
    assert results[0].retrieved_texts == ["alpha", "beta"]
    assert len(retriever.calls) == 2


def test_runner_passes_k_and_filters():
    retriever = FakeRetriever([_chunk("a", "alpha")])
    embedder = Embedder(model="noise", provider=NoiseEmbedder(dim=8))
    queries = [
        EvalQuery(
            query_id="q1",
            query="?",
            expected_answer="",
            spoiler_scope="MB-Era1",
            series_filter=["MB-Era1", "MB-Era2"],
        )
    ]
    run_retrieval(retriever, embedder, queries, k=3)
    call = retriever.calls[0]
    assert call["k"] == 3
    assert call["where"] == {
        "spoiler_scope": "MB-Era1",
        "series_mentioned": {"$in": ["MB-Era1", "MB-Era2"]},
    }


def test_runner_omits_where_when_no_filters():
    retriever = FakeRetriever([_chunk("a", "alpha")])
    embedder = Embedder(model="noise", provider=NoiseEmbedder(dim=8))
    queries = [EvalQuery(query_id="q1", query="?", expected_answer="")]
    run_retrieval(retriever, embedder, queries, k=1)
    assert retriever.calls[0]["where"] is None
