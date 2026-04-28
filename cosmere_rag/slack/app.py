"""`cosmere-slack` entrypoint: Socket Mode listener wired to the agent.

Builds the agent, embedder, and BigQuery retriever once at startup, then
hands them to handlers via `HandlerDeps`. The compiled graph (with its
in-memory checkpointer) is reused across requests so multi-turn threads
share conversation state.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from dotenv import load_dotenv

load_dotenv()

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from cosmere_rag.agent.agent import build_agent
from cosmere_rag.embed.embedder import Embedder
from cosmere_rag.retrieval.bigquery_store import BigQueryStore
from cosmere_rag.slack.handlers import (
    HandlerDeps,
    handle_app_mention,
    handle_direct_message,
)


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"missing required env var: {name}", file=sys.stderr)
        raise SystemExit(2)
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class _HealthHandler(BaseHTTPRequestHandler):
    """Basic health check endpoint so this app doens't get shut down by Cloud Run after a startup probe"""
    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format: str, *args) -> None:  # noqa: A002, ANN001
        pass


def _start_health_server(port: int) -> None:
    server = HTTPServer(("", port), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, name="health", daemon=True)
    thread.start()


def main() -> int:
    logging.basicConfig(level=logging.INFO)

    bot_token = _required_env("SLACK_BOT_TOKEN")
    app_token = _required_env("SLACK_APP_TOKEN")
    project = _required_env("GOOGLE_CLOUD_PROJECT")
    table = _required_env("BIGQUERY_TABLE")
    dataset = os.environ.get("BIGQUERY_DATASET", "cosmere_rag")
    location = os.environ.get("BIGQUERY_LOCATION", "US")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    include_trace_url = _env_bool("SLACK_INCLUDE_TRACE_URL", default=False)

    store = BigQueryStore(
        project=project,
        dataset=dataset,
        table_name=table,
        location=location,
    )
    embedder = Embedder(model=embedding_model)
    graph = build_agent(store, embedder)

    app = App(token=bot_token)
    bot_user_id = app.client.auth_test()["user_id"]

    deps = HandlerDeps(
        agent=graph,
        retriever=store,
        embedder=embedder,
        bot_user_id=bot_user_id,
        include_trace_url=include_trace_url,
    )

    @app.event("app_mention")
    def _on_app_mention(event, client):  # noqa: ANN001
        handle_app_mention(event, client, deps)

    @app.event("message")
    def _on_message(event, client):  # noqa: ANN001
        # The `message` event fires for every message subtype; filter to DMs
        # so we don't try to answer every channel post the bot is invited to.
        if event.get("channel_type") != "im":
            return
        handle_direct_message(event, client, deps)

    logging.getLogger(__name__).info(
        "starting cosmere-slack (table=%s.%s, embedding=%s)",
        dataset,
        table,
        embedding_model,
    )
    port = int(os.environ.get("PORT", "0"))
    if port:
        _start_health_server(port)

    SocketModeHandler(app, app_token).start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
