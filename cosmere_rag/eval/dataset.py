"""Golden query set: data model and JSONL persistence.

A golden set is a small, hand-curated list of `EvalQuery` records. Each
record carries everything both metric tracks need:

  - `expected_answer` for DeepEval's reference-grounded LLM-judge metrics
    (Contextual Precision / Contextual Recall),
  - `relevant_chunk_ids` for classic IR metrics (precision@k, recall@k,
    MRR, NDCG@k). These ids must be stable hashes from
    `cosmere_rag.embed.ids.compute_chunk_id`.

Stored as JSONL so individual queries can be reviewed in a diff and
appended without re-serializing the whole set.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class EvalQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str
    query: str
    expected_answer: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    spoiler_scope: str | None = None
    series_filter: list[str] | None = None
    notes: str | None = None


def load_golden_set(path: Path) -> list[EvalQuery]:
    queries: list[EvalQuery] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(EvalQuery.model_validate_json(line))
    _check_unique_ids(queries)
    return queries


def save_golden_set(path: Path, queries: Iterable[EvalQuery]) -> None:
    queries = list(queries)
    _check_unique_ids(queries)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q.model_dump(), ensure_ascii=False) + "\n")


def _check_unique_ids(queries: list[EvalQuery]) -> None:
    seen: set[str] = set()
    for q in queries:
        if q.query_id in seen:
            raise ValueError(f"duplicate query_id {q.query_id!r} in golden set")
        seen.add(q.query_id)
