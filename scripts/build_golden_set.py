"""Draft golden-set candidates from indexed chunks via an LLM.

Samples N chunks from a Chroma collection, prompts an LLM to produce
`(query, expected_answer)` pairs grounded in each chunk, and writes the
candidates as JSONL. The author then hand-curates the candidates —
fixing the queries, verifying answers, and adding any additional
`relevant_chunk_ids` — into the final golden set.

Each candidate's `relevant_chunk_ids` defaults to the source chunk;
curation should expand this to all chunks a human considers relevant.

Usage:
    uv run python scripts/build_golden_set.py \\
        --collection cosmere-3small-era1 \\
        --num 30 \\
        --out eval/golden_set/mistborn_era1.candidates.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from openai import OpenAI

from cosmere_rag.eval.config import DEFAULT_JUDGE_MODEL
from cosmere_rag.eval.dataset import EvalQuery, save_golden_set
from cosmere_rag.retrieval.chroma_store import ChromaStore

DRAFT_PROMPT = """You are drafting a question-and-answer pair for a retrieval evaluation set.

Below is one chunk of source text from the Coppermind wiki article {article!r}.
Write a single specific question that this chunk uniquely answers, and the ideal short answer (1-3 sentences). 
The question should sound like something a Cosmere reader might genuinely ask. Avoid yes/no questions.

Source text:
\"\"\"{text}\"\"\"

Respond as JSON with exactly two keys: "query" and "expected_answer". No prose outside the JSON."""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection", required=True)
    p.add_argument("--chroma-path", type=Path, default=Path("data/chroma"))
    p.add_argument("--num", type=int, default=30)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--llm-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set", file=sys.stderr)
        return 1

    chunks = _sample_chunks(args.chroma_path, args.collection, args.num, args.seed)
    if not chunks:
        print(f"collection {args.collection!r} has no chunks", file=sys.stderr)
        return 1

    client = OpenAI()
    candidates: list[EvalQuery] = []
    for i, ch in enumerate(chunks, start=1):
        try:
            draft = _draft_one(client, args.llm_model, ch)
        except Exception as exc:
            print(f"  [{i}/{len(chunks)}] skipped chunk {ch['chunk_id']}: {exc}", file=sys.stderr)
            continue
        candidates.append(
            EvalQuery(
                query_id=f"q{i:03d}",
                query=draft["query"],
                expected_answer=draft["expected_answer"],
                relevant_chunk_ids=[ch["chunk_id"]],
                spoiler_scope=ch.get("spoiler_scope"),
                notes=f"Drafted from {ch.get('article_title')!r}",
            )
        )
        print(f"  [{i}/{len(chunks)}] {draft['query']}", file=sys.stderr)

    save_golden_set(args.out, candidates)
    print(f"wrote {len(candidates)} candidates to {args.out}", file=sys.stderr)
    return 0


def _sample_chunks(
    chroma_path: Path, collection: str, num: int, seed: int
) -> list[dict]:
    store = ChromaStore(path=chroma_path, collection_name=collection)
    total = store.count()
    if total == 0:
        return []
    raw = store._collection.get(include=["documents", "metadatas"])
    items = [
        {
            "chunk_id": cid,
            "text": doc,
            "article_title": meta.get("article_title"),
            "spoiler_scope": meta.get("spoiler_scope"),
        }
        for cid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"], strict=True)
    ]
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:num]


def _draft_one(client: OpenAI, model: str, chunk: dict) -> dict[str, str]:
    prompt = DRAFT_PROMPT.format(article=chunk["article_title"], text=chunk["text"])
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.4,
    )
    payload = json.loads(response.choices[0].message.content or "{}")
    if "query" not in payload or "expected_answer" not in payload:
        raise ValueError(f"draft missing fields: {payload!r}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
