# cosmere-rag

A learning project: retrieval-augmented question answering over Brandon Sanderson's Mistborn Era 1 novels.

Source material comes exclusively from public APIs — the [Coppermind wiki](https://coppermind.net), as [index by a scraping repo](https://github.com/Malthemester/CoppermindScraper).
Text is embedded with OpenAI, indexed in Chroma (locally) or BigQuery (deployed), and answered by a LangGraph agent exposed through a CLI and a Slack bot.

## Architecture

```
                    ┌─ build-time pipeline ────────────────────────────────┐
                    │                                                      │
  Coppermind mirror │  cosmere-ingest-coppermind                           │
  (pinned git SHA)  │    → era1.jsonl (chunks + metadata)                  │
                    │  cosmere-embed                                       │
                    │    → era1.parquet (L2-normalized embeddings cache)   │
                    │  cosmere-index --backend {chroma|bigquery}           │
                    │    → upsert into vector store                        │
                    └──────────────────────────────────────────────────────┘
                                           │
                    ┌─ serve-time ─────────▼───────────────────────────────┐
                    │                                                      │
                    │  cosmere-ask (CLI / REPL)    cosmere-slack           │
                    │       └──────────────────────────┘                   │
                    │                   │                                  │
                    │          LangGraph agent (answer())                  │
                    │                   │                                  │
                    │        run_retrieval_chain                           │
                    │          embed query → vector search                 │
                    │          ChromaStore (local) / BigQueryStore (prod)  │
                    └──────────────────────────────────────────────────────┘
```

The hard boundary between build-time and serve-time keeps the deployed image lean: `retrieval/` never imports from `embed/` or `ingest/`.

## Package layout

```
cosmere_rag/
  core/       — domain types: Chunk, RetrievedChunk, Retriever protocol
  ingest/     — Coppermind parsing + chunking (build-time only)
  embed/      — Embedder (LangChain wrapper, L2-normalized), parquet cache, cosmere-embed CLI
  index/      — joins JSONL + parquet, upserts to a backend (build-time only)
  retrieval/  — ChromaStore, BigQueryStore, run_retrieval_chain (serve-time only)
  agent/      — LangGraph QA agent, answer(), cosmere-ask CLI
  slack/      — Slack Bolt Socket-Mode app (cosmere-slack)
  eval/       — LangSmith eval harness (IR metrics + LLM-judge tracks)
```

## Setup

**Requirements:** Python ≥ 3.13, [uv](https://github.com/astral-sh/uv)

```bash
# Install everything (dev + all optional groups)
uv sync --all-extras

# Copy and fill in secrets
cp .env.example .env
```

## Data pipeline

The pipeline runs in three chained steps. Locally, run each CLI in sequence with intermediate files on disk. 
In production it runs as a Cloud Run Job where `/tmp` persists across the chain.

### 1. Fetch the corpus

```bash
# Clones the pinned Coppermind markdown mirror; writes data/coppermind-mirror/ + COMMIT.txt
bash scripts/fetch_corpus.sh
```

The mirror commit SHA stamps every chunk so the corpus snapshot is traceable.

### 2. Ingest + chunk

```bash
cosmere-ingest-coppermind \
  --corpus-dir data/coppermind-mirror/Cosmere \
  --out data/processed/era1.jsonl
```

Selects Era 1 articles, parses wikitext, chunks them, and writes JSONL with a sidecar titles file.

### 3. Embed

```bash
cosmere-embed \
  --chunks data/processed/era1.jsonl \
  --out data/embeddings/era1.parquet \
  [--model text-embedding-3-small]
```

Idempotent: re-embeds only on chunk-id miss, model change, or text-hash change. 
Embeddings are L2-normalized before being written so cosine similarity is just dot product at retrieval time.

### 4. Index

```bash
# Local Chroma (for cosmere-ask)
cosmere-index \
  --backend chroma \
  --chunks data/processed/era1.jsonl \
  --embeddings data/embeddings/era1.parquet \
  --collection era1

# BigQuery (for cosmere-slack in prod)
cosmere-index \
  --backend bigquery \
  --chunks data/processed/era1.jsonl \
  --embeddings data/embeddings/era1.parquet \
  --collection era1
```

The CLI enforces that chunks and embeddings share the same embedding model and refuses to mix models within one collection.

## Asking questions

### CLI

```bash
# One-shot
cosmere-ask --collection era1 "What metals does Vin use?"

# Interactive REPL (omit the question)
cosmere-ask --collection era1
```

Uses the local Chroma store at `data/chroma`. Thread state is kept in memory so follow-up questions work within a session.

### Slack bot

```bash
cosmere-slack
```

Socket-Mode listener — no public ingress required. DM the bot or `@mention` it in a channel. Slack threads are mapped to LangGraph conversation threads so follow-ups stay in context.

Set `SLACK_INCLUDE_TRACE_URL=true` in `.env` to get a LangSmith trace link in each reply (useful for debugging; leave off in shared workspaces).

## Evaluation

```bash
# Run against a LangSmith dataset (IR + LLM-judge)
cosmere-eval run

# Upload a golden JSONL set to LangSmith
cosmere-eval upload-dataset

# Offline IR-only run (no LangSmith required)
cosmere-eval run --offline
```

The harness measures retrieval quality (recall, MRR) and answer quality (LLM-as-judge). Reports land in `reports/`.

## Cloud Run deployment

Infrastructure is managed by Terraform (slow-moving: APIs, Artifact Registry, IAM, Secret Manager, BigQuery, Cloud Run service + job shells). Image tag rolls use `make` + `gcloud` directly.

### First-time setup

```bash
# Bootstrap Terraform remote state bucket
make bootstrap-tfstate

# Provision all GCP infrastructure
make tf-init
make tf-apply
```

### Build and deploy

Two Docker images with separate dependency sets:

| Image | Dockerfile | Deps group | Used by |
|---|---|---|---|
| `serve` | `Dockerfile.serve` | `.[serve]` | `cosmere-slack` Cloud Run service |
| `job` | `Dockerfile.job` | `.[build]` | pipeline Cloud Run Job |

```bash
# Build via Cloud Build + push to Artifact Registry, then roll onto Cloud Run
make release-serve   # serve image: build + deploy
make release-job     # job image: build + deploy

# Run the full ingest → embed → index pipeline as a Cloud Run Job
make run-pipeline

# Tail logs
make logs            # service
make logs-job        # job

# Force a new revision after rotating a secret (no rebuild)
make restart
```

The service runs with `min-instances=1, cpu-always-allocated` to keep the Socket-Mode WebSocket alive without a cold-start gap.

## Configuration

Copy `.env.example` to `.env` and fill in:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Embeddings (`text-embedding-3-small`) + agent LLM |
| `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `LANGSMITH_TRACING` | Tracing + eval. Set `LANGSMITH_TRACING=false` for hermetic test runs. |
| `SLACK_APP_TOKEN` | `xapp-` token with `connections:write` (Socket Mode) |
| `SLACK_BOT_TOKEN` | `xoxb-` bot token |
| `SLACK_INCLUDE_TRACE_URL` | Include LangSmith URL in bot replies (default `false`) |
| `GOOGLE_CLOUD_PROJECT`, `BIGQUERY_DATASET`, `BIGQUERY_TABLE`, `BIGQUERY_LOCATION` | BigQuery retrieval backend |
| `EMBEDDING_MODEL` | Must match the model used at index time (default `text-embedding-3-small`) |

## Tests

```bash
pytest                             # full suite
pytest tests/test_agent.py::test_x  # single test
```

## Notes on source access

Polite rate limits and legitimate public APIs only — no scraping, no book text.
Currently using a GitHub repo (linked up top) as an archive of previously downloaded material.