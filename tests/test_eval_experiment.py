"""Tests for the LangSmith experiment target + filter construction.

These don't hit LangSmith — they exercise the closure shape and the
`build_where` mapping that both the offline runner and the online
target rely on. The end-to-end `evaluate()` path is verified manually
in the LangSmith UI.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from cosmere_rag.core.chunk import Chunk
from cosmere_rag.core.retrieved_chunk import RetrievedChunk
from cosmere_rag.embed.embedder import Embedder
from cosmere_rag.eval.experiment import build_target
from cosmere_rag.eval.runner import build_where


class _StubEmbeddingProvider:
    def embed_query(self, text: str) -> list[float]:
        return [0.5, 0.5, 0.0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.5, 0.5, 0.0] for _ in texts]


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
            text="kelsier crew",
            article_title="Kelsier",
            heading_path=[],
            spoiler_scope="MB-Era1",
            series_mentioned=["MB-Era1"],
            source_url="https://example/coppermind/Kelsier",
            content_provenance="coppermind",
            corpus_snapshot="2026-04-01",
            ingested_at=datetime.now(tz=timezone.utc),
            token_count=4,
        )
        return [RetrievedChunk(chunk=chunk, score=0.77)]


def test_build_where_collapses_single_series_to_scalar():
    assert build_where(spoiler_scope="MB-Era1", series_filter=["MB-Era1"]) == {
        "spoiler_scope": "MB-Era1",
        "series_mentioned": "MB-Era1",
    }


def test_build_where_uses_in_for_multiple_series():
    where = build_where(spoiler_scope=None, series_filter=["MB-Era1", "MB-Era2"])
    assert where == {"series_mentioned": {"$in": ["MB-Era1", "MB-Era2"]}}


def test_build_where_returns_none_when_no_filters():
    assert build_where() is None
    assert build_where(spoiler_scope=None, series_filter=None) is None
    assert build_where(spoiler_scope=None, series_filter=[]) is None


def test_build_target_returns_chain_output_shape():
    retriever = _StubRetriever()
    embedder = Embedder(model="stub-model", provider=_StubEmbeddingProvider())
    target = build_target(
        retriever=retriever, embedder=embedder, k=4, collection="stub"
    )

    out = target(
        {
            "query": "who founded the survivor's crew?",
            "spoiler_scope": "MB-Era1",
            "series_filter": ["MB-Era1"],
        }
    )

    assert out["retrieved_chunk_ids"] == ["c1"]
    assert out["retrieved_texts"] == ["kelsier crew"]
    assert out["scores"] == [0.77]
    assert retriever.last_args["k"] == 4
    assert retriever.last_args["where"] == {
        "spoiler_scope": "MB-Era1",
        "series_mentioned": "MB-Era1",
    }


def test_build_target_handles_missing_filter_keys():
    retriever = _StubRetriever()
    embedder = Embedder(model="stub-model", provider=_StubEmbeddingProvider())
    target = build_target(retriever=retriever, embedder=embedder, k=2)

    target({"query": "anything?"})

    assert retriever.last_args["where"] is None
    assert retriever.last_args["k"] == 2
