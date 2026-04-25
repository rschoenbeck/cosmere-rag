"""CLI: run a retrieval eval against an existing collection.

Usage:
    cosmere-eval \\
        --golden-set eval/golden_set/mistborn_era1.jsonl \\
        --collection cosmere-3small-era1 \\
        --embedding-model text-embedding-3-small \\
        --k 8 \\
        --metrics ir,deepeval \\
        --report reports/era1_3small.json

The same embedding model must be used at query time as at index time.
DeepEval metrics use OpenAI by default and incur LLM cost; pass
`--metrics ir` for a free run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cosmere_rag.embed.embedder import Embedder
from cosmere_rag.eval.baselines import NoiseEmbedder
from cosmere_rag.eval.dataset import load_golden_set
from cosmere_rag.eval.metrics_ir import (
    aggregate_ir_metrics,
    per_query_ir_metrics,
)
from cosmere_rag.eval.report import EvalReport, per_query_record
from cosmere_rag.eval.runner import run_retrieval
from cosmere_rag.retrieval.chroma_store import ChromaStore


def _parse_metrics(value: str) -> list[str]:
    parts = [p.strip().lower() for p in value.split(",") if p.strip()]
    allowed = {"ir", "deepeval"}
    bad = [p for p in parts if p not in allowed]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown metric track(s) {bad}; choose from {sorted(allowed)}"
        )
    return parts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cosmere-eval", description=__doc__)
    p.add_argument("--golden-set", type=Path, required=True)
    p.add_argument("--collection", required=True)
    p.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI model id, or 'noise' for the deterministic random baseline.",
    )
    p.add_argument("--chroma-path", type=Path, default=Path("data/chroma"))
    p.add_argument("--k", type=int, default=8)
    p.add_argument(
        "--metrics",
        type=_parse_metrics,
        default=["ir"],
        help="Comma-separated tracks: ir,deepeval (default: ir).",
    )
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--report", type=Path, required=True)
    p.add_argument(
        "--run-name",
        default=None,
        help="Label used in the report header (default: derived from collection).",
    )
    args = p.parse_args(argv)

    queries = load_golden_set(args.golden_set)
    if not queries:
        print(f"golden set {args.golden_set} is empty", file=sys.stderr)
        return 1

    embedder = _build_embedder(args.embedding_model)
    store = ChromaStore(path=args.chroma_path, collection_name=args.collection)
    if store.count() == 0:
        print(
            f"collection {args.collection!r} at {args.chroma_path} is empty; "
            "run cosmere-index first",
            file=sys.stderr,
        )
        return 1

    print(
        f"evaluating {len(queries)} queries against {args.collection!r} "
        f"(model={args.embedding_model}, k={args.k}, metrics={args.metrics})",
        file=sys.stderr,
    )
    results = run_retrieval(store, embedder, queries, k=args.k)

    ir_summary = aggregate_ir_metrics(results, k=args.k) if "ir" in args.metrics else {}
    deepeval_summary: dict[str, float] = {}
    if "deepeval" in args.metrics:
        from cosmere_rag.eval.metrics_deepeval import aggregate_deepeval_metrics

        deepeval_summary = aggregate_deepeval_metrics(
            results, judge_model=args.judge_model
        )

    per_query = [
        per_query_record(
            r,
            ir_scores=per_query_ir_metrics(r, k=args.k) if "ir" in args.metrics else None,
        )
        for r in results
    ]

    report = EvalReport(
        run_name=args.run_name or args.collection,
        embedding_model=args.embedding_model,
        collection=args.collection,
        backend="chroma",
        k=args.k,
        num_queries=len(queries),
        ir_metrics=ir_summary,
        deepeval_metrics=deepeval_summary,
        per_query=per_query,
    )
    report.write_json(args.report)
    md_path = args.report.with_suffix(".md")
    report.write_markdown(md_path)
    print(f"wrote report to {args.report} and {md_path}", file=sys.stderr)
    return 0


def _build_embedder(model: str) -> Embedder:
    if model == "noise":
        return Embedder(model="noise", provider=NoiseEmbedder())
    return Embedder(model=model)


if __name__ == "__main__":
    raise SystemExit(main())
