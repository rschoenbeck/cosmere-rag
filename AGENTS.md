# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Retrieval-augmented question answering over Brandon Sanderson's Cosmere. Source material is fetched from public APIs (Coppermind wiki, Arcanum/Palanaeum, Sanderson's blog) — never scraped, never books — embedded with OpenAI, indexed in Chroma (local) or BigQuery (deployed), and answered by a LangGraph agent surfaced via a CLI (`cosmere-ask`) and a Slack Bolt app (`cosmere-slack`). LangSmith handles tracing and the eval harness.

Status: learning project, work in progress. Polite rate limits and legitimate APIs are non-negotiable — see `memory/project_source_access.md`.

## Toolchain

- Python `>=3.13,<3.14`, managed by `uv`. The repo pins via `uv.lock`.
- Install for local dev: `uv sync --all-extras` (pulls every optional group + the `dev` group).
- Project install groups (`pyproject.toml`):
  - `build` — pipeline deps (`tiktoken`, `pyarrow`, `langchain-openai`, BigQuery client). Used by the ingest/embed/index CLIs and `Dockerfile.job`.
  - `serve` — runtime deps (`langgraph`, `langchain`, `slack-bolt`, BigQuery client). Used by `cosmere-ask` / `cosmere-slack` and `Dockerfile.serve`.
  - `eval` — extends `serve` with LangSmith for the eval harness.
  - `dev` (dependency group) — `pytest`, `chromadb` (local-only retrieval backend).

## Commands

Console entrypoints (declared in `[project.scripts]`):

- `cosmere-ingest-coppermind --corpus-dir data/coppermind-mirror/Cosmere --out data/processed/era1.jsonl` — selects Era 1 articles, parses, chunks, writes JSONL + a sidecar titles file.
- `cosmere-embed --chunks data/processed/era1.jsonl --out data/embeddings/era1.parquet [--model text-embedding-3-small]` — idempotent embed; re-embeds only on chunk-id miss, model change, or text-hash change.
- `cosmere-index --backend {chroma|bigquery} --chunks ... --embeddings ... [--collection ...]` — joins JSONL+parquet and upserts into the chosen backend.
- `cosmere-ask --collection <name> [question]` — one-shot or REPL; uses the local Chroma store at `data/chroma`.
- `cosmere-slack` — Socket-Mode listener. Reads BigQuery + Slack tokens from env (see `.env.example`).
- `cosmere-eval run|upload-dataset` — LangSmith experiments or `--offline` IR-only runs against a JSONL golden set.

Other:

- `bash scripts/fetch_corpus.sh` — clones the pinned Coppermind markdown mirror to `data/coppermind-mirror/` and writes `COMMIT.txt`. The mirror SHA is the corpus-snapshot stamp on every chunk.
- `pytest` (from repo root) — full suite. `pytest tests/test_agent.py::test_x` for a single test.
- `docker build -f Dockerfile.job .` / `docker build -f Dockerfile.serve .` — separate images for build-time pipeline tools vs serve-time Slack app.

## Architecture

The single most important boundary is **build-time vs serve-time** (see `plans/build-time-vs-serve-time-split.md`). The package is structured so the deployed serve image only pulls in what it needs:

- `cosmere_rag/core/` — domain types (`Chunk`, `RetrievedChunk`, `Retriever` Protocol). Imported everywhere; depends on nothing else.
- `cosmere_rag/ingest/` — Coppermind parsing + chunking from the vendored mirror. Build-time only.
- `cosmere_rag/embed/` — `Embedder` (LangChain wrapper, L2-normalizes vectors so `1 - cosine_distance` is cosine similarity in [0,1]) plus the parquet cache (`embed/store.py`) and the `cosmere-embed` CLI. Build-time except `Embedder` itself, which is reused at query time.
- `cosmere_rag/index/cli.py` — joins chunks JSONL + embeddings parquet and upserts into a backend. Build-time. Refuses to mix embedding models in one collection.
- `cosmere_rag/retrieval/` — pure query-time: `ChromaStore` (local) and `BigQueryStore` (deployed) implementing the `Retriever` Protocol, plus a tiny `chain.py` that does query-embed → retrieve and emits a LangSmith span.
- `cosmere_rag/agent/` — LangGraph `create_agent` wired to a single search tool that calls `run_retrieval_chain`. `answer()` is the single seam every UI calls; it returns `AgentResponse(answer, citations, trace_url)` and is the surface tests target. `InMemorySaver` checkpointer keys threads by `thread_id`.
- `cosmere_rag/slack/` — Bolt app. `app.py` builds deps once and reuses the compiled graph across requests so Slack threads share conversation state. Handlers DM-filter `message` events and route `app_mention`. Optional `PORT` env starts a tiny health server for Cloud Run.
- `cosmere_rag/eval/` — LangSmith-backed retrieval eval (IR + LLM-judge tracks) plus an `--offline` mode that prints aggregated IR metrics without touching LangSmith.

Invariants worth knowing before changing things:

- `retrieval/` must not import from `embed/` or `ingest/`. That's the build/serve seam — keep it clean.
- The same embedding model must be used at query time as at index time. The CLIs default to `text-embedding-3-small`; one collection is tied to one model and `cosmere-index` enforces this.
- `chunk_id = compute_chunk_id(article_title, heading_path, text)` is the join key between JSONL chunks and the parquet embedding cache. Don't recompute it ad-hoc; use `cosmere_rag.embed.ids`.
- Embeddings are L2-normalized in `Embedder` before they ever hit a store. Don't normalize again downstream.
- LangSmith tracing wraps `Embedder`, `run_retrieval_chain`, and `answer()` with `@traceable`. Don't add another wrapper layer — let the library decorators do it.

## Configuration

Env vars (see `.env.example` for the full list with comments):

- `OPENAI_API_KEY` — required for embeddings and the agent generator.
- `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `LANGSMITH_TRACING` — tracing / eval. Set `LANGSMITH_TRACING=false` for hermetic runs.
- Slack: `SLACK_APP_TOKEN` (xapp- with `connections:write`), `SLACK_BOT_TOKEN` (xoxb-), `SLACK_INCLUDE_TRACE_URL` (default false; only flip on for local debugging).
- BigQuery (used by `cosmere-slack` and `cosmere-index --backend bigquery`): `GOOGLE_CLOUD_PROJECT`, `BIGQUERY_DATASET`, `BIGQUERY_TABLE`, `BIGQUERY_LOCATION`, `EMBEDDING_MODEL`.

`.env` is loaded via `python-dotenv` at the top of every CLI entrypoint that needs it.

## Plans directory

`plans/` contains design notes for in-flight work (build/serve split, Chroma→BigQuery migration, LangSmith integration, GCP deploy, Era 1 corpus gaps, Slack agent). Read the relevant plan before making structural changes — they capture motivation that isn't recoverable from the code.