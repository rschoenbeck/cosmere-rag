"""Text helpers for Slack events."""
from __future__ import annotations

import re


def strip_bot_mention(text: str, bot_user_id: str) -> str:
    """Remove `<@BOT>` mentions of the bot from `text`, return what's left.

    Slack delivers `app_mention` events with the bot reference embedded as
    `<@U123456>` (optionally with a `|label`). We don't care where in the
    message the user put the mention — strip every occurrence and collapse
    surrounding whitespace.
    """
    pattern = re.compile(rf"<@{re.escape(bot_user_id)}(?:\|[^>]+)?>")
    return re.sub(r"\s+", " ", pattern.sub("", text)).strip()
