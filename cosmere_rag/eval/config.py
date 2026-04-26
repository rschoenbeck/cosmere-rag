"""Shared eval defaults.

Single source of truth for things every eval module would otherwise
hard-code (and drift on). Bump `DEFAULT_JUDGE_MODEL` here and the CLI,
the LangSmith experiment runner, and the DeepEval wrappers all pick it
up.
"""
from __future__ import annotations

DEFAULT_JUDGE_MODEL = "gpt-5.4-mini"
