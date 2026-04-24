"""CLI: embed chunks from JSONL into a parquet cache, idempotently.

A chunk needs (re-)embedding iff any of these hold:
  - it has no cached row, or
  - the cached row's `model` differs, or
  - the chunk's text hash differs from the cached `chunk_text_hash`.

Rows whose `chunk_id` no longer appears in the JSONL are dropped from
the cache on write, so re-ingests that retire a chunk don't accumulate
dead embeddings.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cosmere_rag.embed import store
from cosmere_rag.embed.embedder import Embedder
from cosmere_rag.embed.ids import compute_chunk_id, compute_text_hash


def _load_chunks(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    for r in rows:
        r["chunk_id"] = compute_chunk_id(r["article_title"], r["heading_path"], r["text"])
        r["chunk_text_hash"] = compute_text_hash(r["text"])
    return rows


def _needs_embed(chunk: dict[str, Any], cached: dict[str, Any] | None, model: str) -> bool:
    if cached is None:
        return True
    if cached["model"] != model:
        return True
    return cached["chunk_text_hash"] != chunk["chunk_text_hash"]


def run(chunks_path: Path, out_path: Path, model: str) -> int:
    chunks = _load_chunks(chunks_path)
    cache = store.read_existing(out_path)

    pending = [c for c in chunks if _needs_embed(c, cache.get(c["chunk_id"]), model)]
    print(
        f"{len(chunks)} chunks, {len(pending)} to (re-)embed with {model}",
        file=sys.stderr,
    )

    if pending:
        result = Embedder(model=model).embed_documents([c["text"] for c in pending])
        now = datetime.now(timezone.utc)
        for chunk, vec in zip(pending, result.embeddings, strict=True):
            cache[chunk["chunk_id"]] = {
                "chunk_id": chunk["chunk_id"],
                "embedding": vec,
                "model": result.model,
                "model_version": result.model_version,
                "chunk_text_hash": chunk["chunk_text_hash"],
                "embedded_at": now,
            }

    live_ids = {c["chunk_id"] for c in chunks}
    rows = [row for cid, row in cache.items() if cid in live_ids]

    store.write(out_path, rows)
    print(f"wrote {len(rows)} embeddings to {out_path}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cosmere-embed")
    p.add_argument("--chunks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="text-embedding-3-small")
    args = p.parse_args(argv)
    return run(args.chunks, args.out, args.model)


if __name__ == "__main__":
    raise SystemExit(main())