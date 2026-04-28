"""Custom agent state schema.

Per LangChain 1.0, `state_schema` passed to `create_agent` must be a
`TypedDict` extending `AgentState` — Pydantic and dataclasses are no
longer supported here. That's the only spot in this codebase that
deviates from the Pydantic house style; we own that exception because
`AgentState` is LangChain's type, not ours.
"""
from __future__ import annotations

from langchain.agents.middleware.types import AgentState

from cosmere_rag.core.retrieved_chunk import RetrievedChunk


class CosmereAgentState(AgentState):
    retrieved_chunks: list[RetrievedChunk]
