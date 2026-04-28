"""Map a Slack event to the agent's `thread_id`.

Channel mentions are keyed by the Slack thread they live in (top-level
mentions become a thread, replies share the parent's `thread_ts`). DMs
have no `thread_ts`, but DM channels are 1:1 with the bot, so the
channel id alone uniquely identifies the conversation.
"""
from __future__ import annotations

from typing import Any


def thread_key_for_event(event: dict[str, Any]) -> str:
    thread_ts = event.get("thread_ts")
    if thread_ts:
        return f"thread:{thread_ts}"
    if event.get("channel_type") == "im":
        return f"dm:{event['channel']}"
    # Top-level channel mention: the message itself is the start of the thread.
    return f"thread:{event['ts']}"
