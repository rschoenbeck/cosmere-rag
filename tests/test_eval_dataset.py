"""Round-trip and validation checks for the golden-set JSONL format."""
from __future__ import annotations

from pathlib import Path

import pytest

from cosmere_rag.eval.dataset import EvalQuery, load_golden_set, save_golden_set


def test_round_trip_preserves_fields(tmp_path: Path):
    queries = [
        EvalQuery(
            query_id="q001",
            query="What metals can Vin burn?",
            expected_answer="Steel, iron, pewter, tin, zinc, brass, copper, bronze, atium.",
            relevant_chunk_ids=["abc", "def"],
            spoiler_scope="MB-Era1",
            series_filter=["MB-Era1"],
            notes="from Vin article",
        ),
        EvalQuery(
            query_id="q002",
            query="Who is the Lord Ruler?",
            expected_answer="Rashek, the Sliver of Infinity.",
            relevant_chunk_ids=["xyz"],
        ),
    ]
    path = tmp_path / "golden.jsonl"
    save_golden_set(path, queries)
    loaded = load_golden_set(path)
    assert loaded == queries


def test_load_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "golden.jsonl"
    path.write_text(
        '{"query_id":"q1","query":"x","expected_answer":"y","relevant_chunk_ids":["a"]}\n'
        "\n"
        '{"query_id":"q2","query":"x","expected_answer":"y","relevant_chunk_ids":["b"]}\n',
        encoding="utf-8",
    )
    assert len(load_golden_set(path)) == 2


def test_duplicate_query_ids_rejected_on_save(tmp_path: Path):
    queries = [
        EvalQuery(query_id="dup", query="x", expected_answer="y"),
        EvalQuery(query_id="dup", query="x2", expected_answer="y2"),
    ]
    with pytest.raises(ValueError, match="duplicate query_id"):
        save_golden_set(tmp_path / "g.jsonl", queries)


def test_duplicate_query_ids_rejected_on_load(tmp_path: Path):
    path = tmp_path / "g.jsonl"
    path.write_text(
        '{"query_id":"dup","query":"a","expected_answer":"x"}\n'
        '{"query_id":"dup","query":"b","expected_answer":"y"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate query_id"):
        load_golden_set(path)
