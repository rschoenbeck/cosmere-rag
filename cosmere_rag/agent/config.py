"""Agent-side defaults.

Kept separate from `cosmere_rag.eval.config` because the generator and the
LLM-judge have different cost/quality profiles and should be tunable
independently.
"""
from __future__ import annotations

DEFAULT_AGENT_MODEL = "gpt-5.4"
DEFAULT_AGENT_K = 8
