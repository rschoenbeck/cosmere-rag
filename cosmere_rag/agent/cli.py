"""CLI: ask the Cosmere RAG agent a question.

    cosmere-ask --collection cosmere-3small-era1 "Who is Kelsier?"
    cosmere-ask --collection cosmere-3small-era1            # REPL

The same embedding model must be used at query time as at index time, so
`--embedding-model` defaults to the same value `cosmere-eval` does.
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from cosmere_rag.agent.agent import answer, build_agent
from cosmere_rag.agent.config import DEFAULT_AGENT_K, DEFAULT_AGENT_MODEL
from cosmere_rag.agent.types import AgentResponse
from cosmere_rag.embed.embedder import Embedder
from cosmere_rag.retrieval.chroma_store import ChromaStore


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cosmere-ask", description=__doc__)
    p.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Question to ask. Omit for an interactive REPL.",
    )
    p.add_argument("--collection", required=True)
    p.add_argument("--chroma-path", type=Path, default=Path("data/chroma"))
    p.add_argument("--embedding-model", default="text-embedding-3-small")
    p.add_argument("--k", type=int, default=DEFAULT_AGENT_K)
    p.add_argument(
        "--model",
        default=DEFAULT_AGENT_MODEL,
        help="Generator model (passed through to LangChain's init_chat_model).",
    )
    args = p.parse_args(argv)

    store = ChromaStore(path=args.chroma_path, collection_name=args.collection)
    if store.count() == 0:
        print(
            f"collection {args.collection!r} at {args.chroma_path} is empty; "
            "run cosmere-index first",
            file=sys.stderr,
        )
        return 1

    embedder = Embedder(model=args.embedding_model)
    graph = build_agent(store, embedder, model=args.model, k=args.k)

    if args.question:
        return _ask_once(graph, store, embedder, args.question, args.model, args.k)
    return _repl(graph, store, embedder, args.model, args.k)


def _ask_once(
    graph,
    store: ChromaStore,
    embedder: Embedder,
    question: str,
    model: str,
    k: int,
    thread_id: str | None = None,
) -> int:
    response = answer(
        question,
        retriever=store,
        embedder=embedder,
        thread_id=thread_id,
        model=model,
        k=k,
        agent=graph,
    )
    _print_response(response)
    return 0


def _repl(graph, store: ChromaStore, embedder: Embedder, model: str, k: int) -> int:
    thread_id = str(uuid.uuid4())
    print(
        f"cosmere-ask REPL (thread={thread_id[:8]}) — :q or Ctrl-D to exit",
        file=sys.stderr,
    )
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return 0
        if not question:
            continue
        if question in {":q", ":quit", ":exit"}:
            return 0
        _ask_once(graph, store, embedder, question, model, k, thread_id=thread_id)
    return 0


def _print_response(response: AgentResponse) -> None:
    print(response.answer)
    if response.citations:
        print()
        print("Sources:")
        for c in response.citations:
            print(f"  - {c.title} — {c.url}")
    if response.trace_url:
        print(f"\nTrace: {response.trace_url}")


if __name__ == "__main__":
    raise SystemExit(main())
