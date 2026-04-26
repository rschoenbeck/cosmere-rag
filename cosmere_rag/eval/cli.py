"""CLI: retrieval eval + LangSmith dataset management.

Subcommands:

    cosmere-eval run \\
        --golden-set eval/golden_set/mistborn_era1.jsonl \\
        --collection cosmere-3small-era1 \\
        --embedding-model text-embedding-3-small \\
        --k 8 \\
        --metrics ir,deepeval \\
        [--dataset-name mistborn_era1] \\
        [--experiment-prefix era1-3small-k8]

    cosmere-eval run --offline \\
        --golden-set eval/golden_set/mistborn_era1.jsonl \\
        --collection cosmere-3small-era1 \\
        --embedding-model text-embedding-3-small \\
        --k 8

    cosmere-eval upload-dataset \\
        --golden-set eval/golden_set/mistborn_era1.jsonl \\
        [--dataset-name mistborn_era1]

The same embedding model must be used at query time as at index time.
The default `run` mode uploads results to LangSmith (the JSONL stem is
used as the dataset name unless `--dataset-name` is given). `--offline`
skips LangSmith entirely and prints the IR aggregate to stdout — useful
for hermetic checks and quick local sanity runs. DeepEval metrics use
OpenAI by default and incur LLM cost; pass `--metrics ir` for a free
run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from cosmere_rag.embed.embedder import Embedder
from cosmere_rag.eval.baselines import NoiseEmbedder
from cosmere_rag.eval.config import DEFAULT_JUDGE_MODEL
from cosmere_rag.eval.dataset import load_golden_set
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
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser(
        "run",
        help="Run a retrieval eval as a LangSmith experiment (or --offline).",
    )
    _add_run_args(run_p)

    upload_p = sub.add_parser(
        "upload-dataset",
        help="Upsert a JSONL golden set into a LangSmith dataset.",
    )
    _add_upload_args(upload_p)

    args = p.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "upload-dataset":
        return _cmd_upload_dataset(args)
    p.error(f"unknown command {args.command!r}")
    return 2


def _add_run_args(p: argparse.ArgumentParser) -> None:
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
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument(
        "--dataset-name",
        default=None,
        help="LangSmith dataset name (default: golden-set file stem).",
    )
    p.add_argument(
        "--experiment-prefix",
        default=None,
        help="LangSmith experiment-name prefix (default: collection-model-k{k}).",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=0,
        help="Parallel evaluator workers (0 = serial).",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Skip LangSmith; run IR metrics locally and print the aggregate.",
    )


def _add_upload_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--golden-set", type=Path, required=True)
    p.add_argument(
        "--dataset-name",
        default=None,
        help="LangSmith dataset name (default: golden-set file stem).",
    )
    p.add_argument(
        "--description",
        default=None,
        help="Description set on dataset creation (ignored if it already exists).",
    )


def _cmd_run(args: argparse.Namespace) -> int:
    store = ChromaStore(path=args.chroma_path, collection_name=args.collection)
    if store.count() == 0:
        print(
            f"collection {args.collection!r} at {args.chroma_path} is empty; "
            "run cosmere-index first",
            file=sys.stderr,
        )
        return 1

    embedder = _build_embedder(args.embedding_model)

    if args.offline:
        return _run_offline(args, store, embedder)
    return _run_langsmith(args, store, embedder)


def _run_langsmith(
    args: argparse.Namespace, store: ChromaStore, embedder: Embedder
) -> int:
    from cosmere_rag.eval.experiment import run_experiment

    dataset_name = args.dataset_name or args.golden_set.stem
    prefix = args.experiment_prefix or (
        f"{args.collection}-{args.embedding_model}-k{args.k}"
    )

    print(
        f"running experiment {prefix!r} on dataset {dataset_name!r} "
        f"(model={args.embedding_model}, k={args.k}, metrics={args.metrics})",
        file=sys.stderr,
    )
    results = run_experiment(
        retriever=store,
        embedder=embedder,
        dataset_name=dataset_name,
        k=args.k,
        collection=args.collection,
        metrics=args.metrics,
        judge_model=args.judge_model,
        experiment_prefix=prefix,
        max_concurrency=args.max_concurrency,
    )
    url = getattr(results, "url", None)
    if url:
        print(f"experiment: {url}", file=sys.stderr)
    return 0


def _run_offline(
    args: argparse.Namespace, store: ChromaStore, embedder: Embedder
) -> int:
    from cosmere_rag.eval.metrics_ir import aggregate_ir_metrics

    queries = load_golden_set(args.golden_set)
    if not queries:
        print(f"golden set {args.golden_set} is empty", file=sys.stderr)
        return 1
    if "deepeval" in args.metrics:
        print(
            "warning: --offline ignores the deepeval track; only IR metrics "
            "are reported",
            file=sys.stderr,
        )

    print(
        f"offline run: {len(queries)} queries against {args.collection!r} "
        f"(model={args.embedding_model}, k={args.k})",
        file=sys.stderr,
    )
    results = run_retrieval(store, embedder, queries, k=args.k)
    summary = aggregate_ir_metrics(results, k=args.k)
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _cmd_upload_dataset(args: argparse.Namespace) -> int:
    from cosmere_rag.eval.langsmith_dataset import upload_golden_set

    dataset = upload_golden_set(
        args.golden_set,
        dataset_name=args.dataset_name,
        description=args.description,
    )
    print(
        f"upserted dataset {dataset.name!r} (id={dataset.id}) from {args.golden_set}",
        file=sys.stderr,
    )
    return 0


def _build_embedder(model: str) -> Embedder:
    if model == "noise":
        return Embedder(model="noise", provider=NoiseEmbedder())
    return Embedder(model=model)


if __name__ == "__main__":
    raise SystemExit(main())
