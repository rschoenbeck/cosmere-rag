"""Smoke-test script: embed a query, run it against the Chroma collection, print top-K.

Usage:
    uv run python scripts/query_chroma.py "What metals can Vin burn?"
    uv run python scripts/query_chroma.py --k 5 --spoiler-scope MB-Era1 "Who is the Lord Ruler?"

This exists for eyeballing retrieval quality during iteration — it is not
part of the production query path. The same embedding model must be used
at query time as at index time, or scores are garbage.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cosmere_rag.embed.embedder import Embedder
from cosmere_rag.retrieval.chroma_store import ChromaStore

DEFAULT_CHROMA_PATH = Path("data/chroma")
DEFAULT_COLLECTION = "mistborn_era1__text-embedding-3-small__v1"
DEFAULT_MODEL = "text-embedding-3-small"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("query", help="Natural-language query string.")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA_PATH)
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--spoiler-scope", default=None)
    p.add_argument(
        "--series",
        default=None,
        help="Filter on series_mentioned (single value or comma-separated list for $in).",
    )
    args = p.parse_args(argv)

    store = ChromaStore(path=args.chroma_path, collection_name=args.collection)
    total = store.count()
    if total == 0:
        print(
            f"collection {args.collection!r} at {args.chroma_path} is empty; "
            "run cosmere-index first",
            file=sys.stderr,
        )
        return 1

    where: dict[str, object] = {}
    if args.spoiler_scope:
        where["spoiler_scope"] = args.spoiler_scope
    if args.series:
        values = [v.strip() for v in args.series.split(",") if v.strip()]
        where["series_mentioned"] = values[0] if len(values) == 1 else {"$in": values}

    query_vec = Embedder(model=args.model).embed_query(args.query)
    results = store.query(query_vec, k=args.k, where=where or None)

    print(
        f"query: {args.query!r}  [k={args.k}, total={total}, where={where or None}]\n",
        file=sys.stderr,
    )
    for i, r in enumerate(results, start=1):
        c = r.chunk
        heading = " › ".join(c.heading_path) if c.heading_path else "(root)"
        print(f"#{i}  score={r.score:.4f}  {c.article_title} — {heading}")
        print(f"    {c.source_url}")
        snippet = c.text.strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:240] + "…"
        print(f"    {snippet}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())