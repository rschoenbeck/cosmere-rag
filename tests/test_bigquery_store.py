"""Unit tests for BigQueryStore.

We don't stand up a live BigQuery instance here — the point of these
tests is to nail down the pure-logic bits that are easy to get wrong:

  - the `where` vocabulary translates to the right SQL + parameters,
  - ARRAY fields round-trip through `_chunk_to_row` / `_row_to_chunk`,
  - parameter types are inferred correctly for scalar vs. `$in`.

Live query behavior is covered by a separate (unlandled) integration
test that hits a real dataset; that's a deployment-time concern, not a
development-time one.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from google.cloud import bigquery

from cosmere_rag.core.chunk import Chunk
from cosmere_rag.retrieval.bigquery_store import (
    _chunk_to_row,
    _row_to_chunk,
    translate_where,
)


def _chunk(**overrides) -> Chunk:
    defaults = dict(
        chunk_id="id-1",
        text="some text",
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


def test_translate_where_empty_returns_no_clause():
    sql, params = translate_where(None)
    assert sql == ""
    assert params == []

    sql, params = translate_where({})
    assert sql == ""
    assert params == []


def test_translate_where_scalar_equality():
    sql, params = translate_where({"spoiler_scope": "MB-Era1"})
    assert sql == "WHERE base.spoiler_scope = @w0"
    assert len(params) == 1
    assert isinstance(params[0], bigquery.ScalarQueryParameter)
    assert params[0].name == "w0"
    assert params[0].type_ == "STRING"
    assert params[0].value == "MB-Era1"


def test_translate_where_scalar_in():
    sql, params = translate_where(
        {"spoiler_scope": {"$in": ["MB-Era1", "Stormlight"]}}
    )
    assert sql == "WHERE base.spoiler_scope IN UNNEST(@w0)"
    assert isinstance(params[0], bigquery.ArrayQueryParameter)
    assert params[0].array_type == "STRING"
    assert params[0].values == ["MB-Era1", "Stormlight"]


def test_translate_where_series_mentioned_scalar_unnests():
    sql, params = translate_where({"series_mentioned": "Stormlight"})
    assert sql == "WHERE @w0 IN UNNEST(base.series_mentioned)"
    assert isinstance(params[0], bigquery.ScalarQueryParameter)
    assert params[0].value == "Stormlight"


def test_translate_where_series_mentioned_in_uses_exists():
    sql, params = translate_where(
        {"series_mentioned": {"$in": ["MB-Era1", "Stormlight"]}}
    )
    assert (
        sql
        == "WHERE EXISTS (SELECT 1 FROM UNNEST(base.series_mentioned) AS s "
        "WHERE s IN UNNEST(@w0))"
    )
    assert isinstance(params[0], bigquery.ArrayQueryParameter)
    assert params[0].values == ["MB-Era1", "Stormlight"]


def test_translate_where_multiple_conditions_anded():
    sql, params = translate_where(
        {
            "spoiler_scope": "MB-Era1",
            "series_mentioned": "MB-Era1",
        }
    )
    assert sql == (
        "WHERE base.spoiler_scope = @w0 "
        "AND @w1 IN UNNEST(base.series_mentioned)"
    )
    assert [p.name for p in params] == ["w0", "w1"]


def test_translate_where_infers_int_type():
    sql, params = translate_where({"token_count": 42})
    assert sql == "WHERE base.token_count = @w0"
    assert params[0].type_ == "INT64"
    assert params[0].value == 42


def test_chunk_row_round_trip_preserves_arrays_and_timestamp():
    original = _chunk(
        heading_path=["A", "B", "C"],
        series_mentioned=["MB-Era1", "Stormlight"],
    )
    embedding = [0.1, 0.2, 0.3]
    row = _chunk_to_row(original, embedding)

    assert row["heading_path"] == ["A", "B", "C"]
    assert row["series_mentioned"] == ["MB-Era1", "Stormlight"]
    assert row["embedding"] == [0.1, 0.2, 0.3]
    # ingested_at is serialized to ISO for JSON load; round-trip goes
    # back through Pydantic, which parses ISO strings.
    assert isinstance(row["ingested_at"], str)

    restored = _row_to_chunk(row)
    assert restored.heading_path == ["A", "B", "C"]
    assert restored.series_mentioned == ["MB-Era1", "Stormlight"]
    assert restored.ingested_at == original.ingested_at
    assert restored.token_count == original.token_count


def test_row_to_chunk_handles_null_arrays():
    # BigQuery returns None for empty repeated fields in some client paths;
    # the un-mapper must treat that as [].
    row = {
        "chunk_id": "id-x",
        "text": "t",
        "article_title": "Vin",
        "heading_path": None,
        "spoiler_scope": "MB-Era1",
        "series_mentioned": None,
        "source_url": "https://example",
        "content_provenance": "coppermind-mirror",
        "corpus_snapshot": "abc",
        "ingested_at": datetime(2026, 4, 22, tzinfo=timezone.utc),
        "token_count": 1,
    }
    chunk = _row_to_chunk(row)
    assert chunk.heading_path == []
    assert chunk.series_mentioned == []


def test_chunk_to_row_rejects_no_embedding():
    # No explicit guard here, but a None embedding would fail type
    # coercion — document the expectation.
    with pytest.raises(TypeError):
        _chunk_to_row(_chunk(), None)  # type: ignore[arg-type]
