"""Embed the corpus with NoiseEmbedder and index it into a separate Chroma collection.

This produces the noise-baseline collection used by `cosmere-eval` to
sanity-check that the eval can distinguish signal from random vectors.

Usage:
    uv run python scripts/build_noise_index.py \\
        --chunks data/coppermind/mistborn_era1.jsonl \\
        --collection cosmere-noise-era1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cosmere_rag.core.chunk import Chunk
from cosmere_rag.embed.embedder import Embedder
from cosmere_rag.embed.ids import compute_chunk_id
from cosmere_rag.eval.baselines import NoiseEmbedder
from cosmere_rag.retrieval.chroma_store import ChromaStore


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunks", type=Path, required=True)
    p.add_argument("--collection", required=True)
    p.add_argument("--chroma-path", type=Path, default=Path("data/chroma"))
    p.add_argument("--dim", type=int, default=1536)
    args = p.parse_args(argv)

    chunk_dicts: list[dict] = []
    with args.chunks.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk_dicts.append(json.loads(line))

    if not chunk_dicts:
        print(f"no chunks in {args.chunks}", file=sys.stderr)
        return 1

    for r in chunk_dicts:
        r["chunk_id"] = compute_chunk_id(r["article_title"], r["heading_path"], r["text"])
    chunks = [Chunk.model_validate(r) for r in chunk_dicts]

    embedder = Embedder(model="noise", provider=NoiseEmbedder(dim=args.dim))
    result = embedder.embed_documents([c.text for c in chunks])

    store = ChromaStore(path=args.chroma_path, collection_name=args.collection)
    store.add(chunks, result.embeddings)
    print(
        f"indexed {len(chunks)} noise embeddings into "
        f"{args.collection!r} at {args.chroma_path} [total={store.count()}]",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
