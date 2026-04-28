"""Render an `AgentResponse` as Slack Block Kit blocks."""
from __future__ import annotations

from typing import Any

from cosmere_rag.agent.types import AgentResponse


def agent_response_to_blocks(
    response: AgentResponse, *, include_trace_url: bool = False
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": response.answer or "_no answer_"},
        }
    ]

    if response.citations:
        links = " · ".join(f"<{c.url}|{c.title}>" for c in response.citations)
        blocks.append(
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": f"*Sources:* {links}"}],
            }
        )

    if include_trace_url and response.trace_url:
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"<{response.trace_url}|trace>"}
                ],
            }
        )

    return blocks
