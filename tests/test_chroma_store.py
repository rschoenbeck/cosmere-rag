"""Unit tests for ChromaStore.

These tests drive a tmp-dir-backed Chroma client with hand-built chunks
and fake (but L2-normalized) embeddings. They exist to prove that:
  - metadata flattening + unflattening round-trip cleanly,
  - the `series_mentioned` filter translates to the bool columns,
  - cosine distance is normalized to similarity before being returned,
  - `upsert` semantics make re-adds idempotent.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from cosmere_rag.core.chunk import Chunk
from cosmere_rag.retrieval.chroma_store import ChromaStore


def _normalize(vec: list[float]) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).tolist()


def _chunk(chunk_id: str, text: str, **overrides) -> Chunk:
    defaults = dict(
        chunk_id=chunk_id,
        text=text,
        article_title="Vin",
        heading_path=["Attributes", "Allomancy"],
        spoiler_scope="MB-Era1",
        series_mentioned=["MB-Era1"],
        source_url="https://coppermind.net/wiki/Vin",
        content_provenance="coppermind-mirror",
        corpus_snapshot="abc123",
        ingested_at=datetime(2026, 4, 22, tzinfo=timezone.utc),
        token_count=42,
    )
    defaults.update(overrides)
    return Chunk(**defaults)


@pytest.fixture
def store(tmp_path: Path) -> ChromaStore:
    return ChromaStore(path=tmp_path / "chroma", collection_name="test__v1")


def test_add_and_count(store: ChromaStore):
    chunks = [_chunk(f"id-{i}", f"text {i}") for i in range(3)]
    embeddings = [_normalize([1.0, 0.0, 0.0]) for _ in chunks]
    store.add(chunks, embeddings)
    assert store.count() == 3


def test_query_returns_similarity_not_distance(store: ChromaStore):
    chunks = [_chunk("id-a", "alpha"), _chunk("id-b", "beta")]
    # Make id-a exactly parallel to the query, id-b orthogonal.
    embeddings = [_normalize([1.0, 0.0, 0.0]), _normalize([0.0, 1.0, 0.0])]
    store.add(chunks, embeddings)

    results = store.query(_normalize([1.0, 0.0, 0.0]), k=2)
    assert [r.chunk.chunk_id for r in results] == ["id-a", "id-b"]
    # Parallel vectors: similarity ~1.0. Orthogonal: ~0.0. Never negative
    # on normalized vectors with cosine distance in [0, 2].
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
    assert results[1].score == pytest.approx(0.0, abs=1e-5)


def test_roundtrip_preserves_list_metadata(store: ChromaStore):
    original = _chunk(
        "id-x",
        "some text",
        heading_path=["A", "B", "C"],
        series_mentioned=["MB-Era1", "Stormlight"],
    )
    store.add([original], [_normalize([1.0, 0.0, 0.0])])

    got = store.query(_normalize([1.0, 0.0, 0.0]), k=1)[0].chunk
    assert got.heading_path == ["A", "B", "C"]
    assert got.series_mentioned == ["MB-Era1", "Stormlight"]
    assert got.ingested_at == original.ingested_at


def test_series_mentioned_filter_selects_only_matching(store: ChromaStore):
    mistborn = _chunk("id-m", "mistborn text", series_mentioned=["MB-Era1"])
    stormlight = _chunk(
        "id-s",
        "stormlight text",
        article_title="Kaladin",
        series_mentioned=["Stormlight"],
        spoiler_scope="Stormlight",
    )
    store.add(
        [mistborn, stormlight],
        [_normalize([1.0, 0.0, 0.0]), _normalize([1.0, 0.0, 0.0])],
    )

    results = store.query(
        _normalize([1.0, 0.0, 0.0]),
        k=5,
        where={"series_mentioned": "Stormlight"},
    )
    assert [r.chunk.chunk_id for r in results] == ["id-s"]


def test_series_mentioned_in_filter_matches_either(store: ChromaStore):
    a = _chunk("id-a", "a", series_mentioned=["MB-Era1"])
    b = _chunk("id-b", "b", series_mentioned=["Stormlight"])
    c = _chunk("id-c", "c", series_mentioned=["Warbreaker"])
    emb = _normalize([1.0, 0.0, 0.0])
    store.add([a, b, c], [emb, emb, emb])

    results = store.query(
        emb,
        k=5,
        where={"series_mentioned": {"$in": ["MB-Era1", "Stormlight"]}},
    )
    assert {r.chunk.chunk_id for r in results} == {"id-a", "id-b"}


def test_spoiler_scope_equality_filter(store: ChromaStore):
    a = _chunk("id-a", "a", spoiler_scope="MB-Era1")
    b = _chunk("id-b", "b", spoiler_scope="Stormlight")
    emb = _normalize([1.0, 0.0, 0.0])
    store.add([a, b], [emb, emb])

    results = store.query(emb, k=5, where={"spoiler_scope": "MB-Era1"})
    assert [r.chunk.chunk_id for r in results] == ["id-a"]


def test_upsert_is_idempotent(store: ChromaStore):
    c = _chunk("id-1", "first version")
    emb = _normalize([1.0, 0.0, 0.0])
    store.add([c], [emb])
    store.add([c], [emb])  # same id, same text — should not duplicate
    assert store.count() == 1

    updated = _chunk("id-1", "second version")
    store.add([updated], [emb])
    assert store.count() == 1
    got = store.query(emb, k=1)[0].chunk
    assert got.text == "second version"


def test_add_rejects_mismatched_lengths(store: ChromaStore):
    with pytest.raises(ValueError):
        store.add([_chunk("id-1", "t")], [])