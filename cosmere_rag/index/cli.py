"""CLI: index embedded chunks into a retrieval backend.

Joins the JSONL chunks corpus with the embeddings parquet produced by
`cosmere-embed`, and upserts the resulting (chunk, vector) pairs into the
selected backend. Only `chroma` is wired up today; the flag exists so the
BigQuery backend can slot in without changing callers.

Integrity checks before writing:
  - the parquet must not mix embedding models (a single collection is
    tied to one model),
  - every JSONL chunk must have a matching embedding row.
Either failure means `cosmere-embed` needs to run again; we'd rather
surface that than silently index a partial corpus.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from cosmere_rag.embed.ids import compute_chunk_id
from cosmere_rag.embed.store import SCHEMA as EMBEDDINGS_SCHEMA
from cosmere_rag.core.chunk import Chunk
from cosmere_rag.retrieval.chroma_store import ChromaStore


def _load_chunks(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    for r in rows:
        r["chunk_id"] = compute_chunk_id(
            r["article_title"], r["heading_path"], r["text"]
        )
    return rows


def _load_embeddings(path: Path) -> dict[str, dict[str, Any]]:
    table = pq.read_table(path).cast(EMBEDDINGS_SCHEMA)
    return {row["chunk_id"]: row for row in table.to_pylist()}


def _sole_model(embeddings: Iterable[dict[str, Any]]) -> str:
    models = {row["model"] for row in embeddings}
    if len(models) != 1:
        raise ValueError(
            f"embeddings parquet mixes models {sorted(models)}; "
            "one collection must map to exactly one model"
        )
    return next(iter(models))


def run(
    chunks_path: Path,
    embeddings_path: Path,
    chroma_path: Path,
    collection_name: str,
) -> int:
    chunk_rows = _load_chunks(chunks_path)
    if not chunk_rows:
        print(f"no chunks in {chunks_path}; nothing to index", file=sys.stderr)
        return 0

    embeddings = _load_embeddings(embeddings_path)
    model = _sole_model(embeddings.values())

    missing = [r["chunk_id"] for r in chunk_rows if r["chunk_id"] not in embeddings]
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(chunk_rows)} chunks have no embedding "
            f"(example chunk_id={missing[0]}); re-run cosmere-embed"
        )

    chunks = [Chunk.model_validate(r) for r in chunk_rows]
    vectors = [embeddings[c.chunk_id]["embedding"] for c in chunks]

    store = ChromaStore(path=chroma_path, collection_name=collection_name)
    store.add(chunks, vectors)
    print(
        f"indexed {len(chunks)} chunks ({model}) into "
        f"{collection_name!r} at {chroma_path} [total={store.count()}]",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cosmere-index")
    p.add_argument("--backend", choices=["chroma"], default="chroma")
    p.add_argument("--chunks", type=Path, required=True)
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--chroma-path", type=Path, default=Path("data/chroma"))
    p.add_argument(
        "--collection",
        default=None,
        help="Collection name (default: the embeddings parquet file stem).",
    )
    args = p.parse_args(argv)

    collection = args.collection or args.embeddings.stem
    return run(args.chunks, args.embeddings, args.chroma_path, collection)


if __name__ == "__main__":
    raise SystemExit(main())
