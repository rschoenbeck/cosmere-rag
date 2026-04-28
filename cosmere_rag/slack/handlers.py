"""Slack event handlers, factored out of `app.py` so they're testable.

Both `app_mention` and `message.im` follow the same shape: ack, post a
placeholder message, run the agent, then `chat.update` the placeholder
with the final answer. Keeping the agent + retriever bound at startup
(in `app.py`) means each handler call only has to thread the event
through.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from cosmere_rag.agent.agent import answer
from cosmere_rag.agent.types import AgentResponse
from cosmere_rag.core.retriever import Retriever
from cosmere_rag.embed.embedder import Embedder
from cosmere_rag.slack.formatting import agent_response_to_blocks
from cosmere_rag.slack.text import strip_bot_mention
from cosmere_rag.slack.threading import thread_key_for_event

log = logging.getLogger(__name__)

PLACEHOLDER_TEXT = "Searching the Coppermind…"
EMPTY_QUESTION_REPLY = "What would you like to know about the Cosmere?"
ERROR_REPLY = "Something went wrong while answering. Check the logs."
MAX_QUESTION_CHARS = 4000

AnswerFn = Callable[..., AgentResponse]


@dataclass(frozen=True)
class HandlerDeps:
    """Bundle of dependencies a handler needs.

    Wrapping these instead of accepting six positional kwargs keeps the
    `slack-bolt` registration call short and the test seams obvious.
    """

    agent: Any
    retriever: Retriever
    embedder: Embedder
    bot_user_id: str
    include_trace_url: bool = False
    answer_fn: AnswerFn = answer


def handle_app_mention(event: dict[str, Any], client: Any, deps: HandlerDeps) -> None:
    if event.get("bot_id"):
        return
    question = strip_bot_mention(event.get("text", ""), deps.bot_user_id)
    reply_thread_ts = event.get("thread_ts") or event["ts"]
    _respond(event, question, client, deps, reply_thread_ts=reply_thread_ts)


def handle_direct_message(
    event: dict[str, Any], client: Any, deps: HandlerDeps
) -> None:
    if event.get("bot_id") or event.get("subtype"):
        # Skip edits, joins, file shares, and the bot's own messages.
        return
    question = (event.get("text") or "").strip()
    _respond(event, question, client, deps, reply_thread_ts=None)


def _respond(
    event: dict[str, Any],
    question: str,
    client: Any,
    deps: HandlerDeps,
    *,
    reply_thread_ts: str | None,
) -> None:
    channel = event["channel"]

    if not question:
        client.chat_postMessage(
            channel=channel,
            thread_ts=reply_thread_ts,
            text=EMPTY_QUESTION_REPLY,
        )
        return

    if len(question) > MAX_QUESTION_CHARS:
        question = question[:MAX_QUESTION_CHARS]

    placeholder = client.chat_postMessage(
        channel=channel,
        thread_ts=reply_thread_ts,
        text=PLACEHOLDER_TEXT,
    )
    placeholder_ts = placeholder["ts"]

    thread_id = thread_key_for_event(event)

    try:
        response = deps.answer_fn(
            question,
            retriever=deps.retriever,
            embedder=deps.embedder,
            agent=deps.agent,
            thread_id=thread_id,
        )
    except Exception:
        log.exception("agent.answer failed for thread_id=%s", thread_id)
        client.chat_update(
            channel=channel,
            ts=placeholder_ts,
            text=ERROR_REPLY,
        )
        return

    client.chat_update(
        channel=channel,
        ts=placeholder_ts,
        text=response.answer or PLACEHOLDER_TEXT,
        blocks=agent_response_to_blocks(
            response, include_trace_url=deps.include_trace_url
        ),
    )
