"""Parquet-backed cache of chunk embeddings.

Schema is pinned — readers fail fast on drift rather than silently
interpreting the wrong columns. Writes are atomic (tmp-then-rename) so
an interrupted run can't leave a half-written parquet on disk.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA = pa.schema(
    [
        pa.field("chunk_id", pa.string(), nullable=False),
        pa.field("embedding", pa.list_(pa.float32()), nullable=False),
        pa.field("model", pa.string(), nullable=False),
        pa.field("model_version", pa.string(), nullable=False),
        pa.field("chunk_text_hash", pa.string(), nullable=False),
        pa.field("embedded_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


def read_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    table = pq.read_table(path)
    table = table.cast(SCHEMA)
    return {row["chunk_id"]: row for row in table.to_pylist()}


def write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows), schema=SCHEMA)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)