# cosmere-rag

A learning project: retrieval-augmented question answering over Brandon Sanderson's Cosmere.

Pulls source material from public APIs (Coppermind wiki, Arcanum/Palanaeum, blog), embeds and indexes it, and answers questions via a LangGraph agent — exposed through a CLI and a Slack app.

## Layout

- `cosmere_rag/ingest` — fetch source documents
- `cosmere_rag/embed` / `index` — embeddings + vector index
- `cosmere_rag/retrieval` — retriever
- `cosmere_rag/agent` — LangGraph QA agent + CLI (`cosmere-ask`)
- `cosmere_rag/slack` — Slack Bolt app (`cosmere-slack`)
- `cosmere_rag/eval` — eval harness
- `eval/`, `tests/`, `scripts/` — datasets, tests, ops scripts

## Status

Work in progress. A proper README will follow.