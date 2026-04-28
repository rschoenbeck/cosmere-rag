from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from cosmere_rag.agent.types import AgentResponse, Citation
from cosmere_rag.slack import handlers as h


@pytest.fixture
def fake_response() -> AgentResponse:
    return AgentResponse(
        answer="Kelsier founded the Survivors.",
        citations=[Citation(title="Kelsier", url="https://coppermind.net/wiki/Kelsier")],
        trace_url="https://smith.langchain.com/run/abc",
    )


def _client(placeholder_ts: str = "1700000000.000200") -> MagicMock:
    client = MagicMock()
    client.chat_postMessage.return_value = {"ts": placeholder_ts}
    return client


def _deps(answer_fn=None, **overrides: Any) -> h.HandlerDeps:
    base: dict[str, Any] = dict(
        agent=object(),
        retriever=object(),
        embedder=object(),
        bot_user_id="UBOT",
        include_trace_url=False,
    )
    base.update(overrides)
    if answer_fn is not None:
        base["answer_fn"] = answer_fn
    return h.HandlerDeps(**base)


def test_app_mention_happy_path(fake_response):
    client = _client()
    answer_fn = MagicMock(return_value=fake_response)
    deps = _deps(answer_fn=answer_fn)
    event = {
        "type": "app_mention",
        "channel": "C123",
        "ts": "1700000000.000100",
        "text": "<@UBOT> who is Kelsier?",
        "channel_type": "channel",
    }

    h.handle_app_mention(event, client, deps)

    assert answer_fn.call_count == 1
    call = answer_fn.call_args
    assert call.args == ("who is Kelsier?",)
    assert call.kwargs["thread_id"] == "thread:1700000000.000100"
    assert call.kwargs["agent"] is deps.agent

    placeholder_call = client.chat_postMessage.call_args
    assert placeholder_call.kwargs["channel"] == "C123"
    assert placeholder_call.kwargs["thread_ts"] == "1700000000.000100"
    assert placeholder_call.kwargs["text"] == h.PLACEHOLDER_TEXT

    update_call = client.chat_update.call_args
    assert update_call.kwargs["channel"] == "C123"
    assert update_call.kwargs["ts"] == "1700000000.000200"
    assert update_call.kwargs["text"] == fake_response.answer
    blocks = update_call.kwargs["blocks"]
    assert blocks[0]["type"] == "section"


def test_app_mention_inside_thread_uses_thread_ts(fake_response):
    client = _client()
    answer_fn = MagicMock(return_value=fake_response)
    deps = _deps(answer_fn=answer_fn)
    event = {
        "type": "app_mention",
        "channel": "C123",
        "ts": "1700000000.000200",
        "thread_ts": "1700000000.000100",
        "text": "<@UBOT> follow-up?",
        "channel_type": "channel",
    }

    h.handle_app_mention(event, client, deps)

    assert client.chat_postMessage.call_args.kwargs["thread_ts"] == "1700000000.000100"
    assert (
        answer_fn.call_args.kwargs["thread_id"] == "thread:1700000000.000100"
    )


def test_app_mention_skips_bot_messages():
    client = _client()
    answer_fn = MagicMock()
    deps = _deps(answer_fn=answer_fn)
    event = {
        "type": "app_mention",
        "channel": "C123",
        "ts": "1700000000.000100",
        "text": "<@UBOT> hi",
        "bot_id": "B999",
    }

    h.handle_app_mention(event, client, deps)

    answer_fn.assert_not_called()
    client.chat_postMessage.assert_not_called()


def test_empty_question_after_strip_does_not_call_agent():
    client = _client()
    answer_fn = MagicMock()
    deps = _deps(answer_fn=answer_fn)
    event = {
        "type": "app_mention",
        "channel": "C123",
        "ts": "1700000000.000100",
        "text": "<@UBOT>",
        "channel_type": "channel",
    }

    h.handle_app_mention(event, client, deps)

    answer_fn.assert_not_called()
    client.chat_postMessage.assert_called_once()
    assert (
        client.chat_postMessage.call_args.kwargs["text"] == h.EMPTY_QUESTION_REPLY
    )
    client.chat_update.assert_not_called()


def test_dm_uses_channel_thread_id(fake_response):
    client = _client()
    answer_fn = MagicMock(return_value=fake_response)
    deps = _deps(answer_fn=answer_fn)
    event = {
        "type": "message",
        "channel": "D123",
        "ts": "1700000000.000100",
        "text": "who is Vin?",
        "channel_type": "im",
    }

    h.handle_direct_message(event, client, deps)

    assert answer_fn.call_args.args == ("who is Vin?",)
    assert answer_fn.call_args.kwargs["thread_id"] == "dm:D123"
    # DMs reply at top level (no thread_ts).
    assert client.chat_postMessage.call_args.kwargs["thread_ts"] is None


def test_dm_skips_message_subtypes():
    client = _client()
    answer_fn = MagicMock()
    deps = _deps(answer_fn=answer_fn)
    event = {
        "type": "message",
        "channel": "D123",
        "ts": "1700000000.000100",
        "text": "edited",
        "channel_type": "im",
        "subtype": "message_changed",
    }

    h.handle_direct_message(event, client, deps)

    answer_fn.assert_not_called()
    client.chat_postMessage.assert_not_called()


def test_agent_failure_posts_error_message():
    client = _client()
    answer_fn = MagicMock(side_effect=RuntimeError("boom"))
    deps = _deps(answer_fn=answer_fn)
    event = {
        "type": "app_mention",
        "channel": "C123",
        "ts": "1700000000.000100",
        "text": "<@UBOT> who is Kelsier?",
        "channel_type": "channel",
    }

    h.handle_app_mention(event, client, deps)

    update_call = client.chat_update.call_args
    assert update_call.kwargs["text"] == h.ERROR_REPLY
    assert update_call.kwargs["ts"] == "1700000000.000200"


def test_long_question_is_truncated(fake_response):
    client = _client()
    answer_fn = MagicMock(return_value=fake_response)
    deps = _deps(answer_fn=answer_fn)
    event = {
        "type": "message",
        "channel": "D123",
        "ts": "1700000000.000100",
        "text": "x" * (h.MAX_QUESTION_CHARS + 100),
        "channel_type": "im",
    }

    h.handle_direct_message(event, client, deps)

    assert len(answer_fn.call_args.args[0]) == h.MAX_QUESTION_CHARS


def test_trace_url_included_when_enabled(fake_response):
    client = _client()
    answer_fn = MagicMock(return_value=fake_response)
    deps = _deps(answer_fn=answer_fn, include_trace_url=True)
    event = {
        "type": "app_mention",
        "channel": "C123",
        "ts": "1700000000.000100",
        "text": "<@UBOT> who is Kelsier?",
        "channel_type": "channel",
    }

    h.handle_app_mention(event, client, deps)

    blocks = client.chat_update.call_args.kwargs["blocks"]
    serialized = str(blocks)
    assert "smith.langchain.com/run/abc" in serialized
