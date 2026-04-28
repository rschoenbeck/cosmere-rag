"""LangChain v1 `create_agent` graph + public `answer()` entrypoint.

Public surface kept deliberately small: `build_agent` (for tests / future
surfaces that want to inject deps) and `answer` (one-shot or threaded).

`answer()` is the seam every surface (CLI now, Slack later) calls. Its
job is to: run the graph, post-process the retrieved chunks into a
deterministic `citations` list, and surface a LangSmith trace URL when
tracing is on.
"""
from __future__ import annotations

import uuid
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from cosmere_rag.agent.config import DEFAULT_AGENT_K, DEFAULT_AGENT_MODEL
from cosmere_rag.agent.prompts import SYSTEM_PROMPT
from cosmere_rag.agent.state import CosmereAgentState
from cosmere_rag.agent.tools import make_search_tool
from cosmere_rag.agent.types import AgentResponse, Citation
from cosmere_rag.core.retrieved_chunk import RetrievedChunk
from cosmere_rag.core.retriever import Retriever
from cosmere_rag.embed.embedder import Embedder


def build_agent(
    retriever: Retriever,
    embedder: Embedder,
    *,
    model: str | BaseChatModel = DEFAULT_AGENT_MODEL,
    k: int = DEFAULT_AGENT_K,
    checkpointer: Any | None = None,
):
    """Compile a `create_agent` graph wired to the Coppermind retrieval tool.

    `model` accepts either a model id string (passed to `init_chat_model`
    inside `create_agent`) or a pre-built `BaseChatModel` instance — the
    latter is what tests use to inject a fake.
    """
    tool = make_search_tool(retriever, embedder, default_k=k)
    return create_agent(
        model,
        tools=[tool],
        system_prompt=SYSTEM_PROMPT,
        state_schema=CosmereAgentState,
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
    )


def answer(
    question: str,
    *,
    retriever: Retriever,
    embedder: Embedder,
    thread_id: str | None = None,
    model: str | BaseChatModel = DEFAULT_AGENT_MODEL,
    k: int = DEFAULT_AGENT_K,
    agent: Any | None = None,
) -> AgentResponse:
    """Answer a Cosmere question with citations.

    Pass `agent` to reuse a compiled graph across turns (a REPL does this
    so it isn't rebuilding per question). `thread_id` keys LangGraph's
    checkpointer — same id across calls = a continuing conversation.
    """
    graph = agent if agent is not None else build_agent(
        retriever, embedder, model=model, k=k
    )
    tid = thread_id or str(uuid.uuid4())

    answer_text, chunks, trace_url = _run_traced(graph, question, tid)
    return AgentResponse(
        answer=answer_text,
        citations=_build_citations(chunks),
        trace_url=trace_url,
    )


@traceable(run_type="chain", name="cosmere_agent.answer")
def _run_traced(
    graph: Any, question: str, thread_id: str
) -> tuple[str, list[RetrievedChunk], str | None]:
    result = graph.invoke(
        {
            "messages": [{"role": "user", "content": question}],
            "retrieved_chunks": [],
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    chunks = list(result.get("retrieved_chunks") or [])
    text = _last_ai_text(result)
    run = get_current_run_tree()
    url = getattr(run, "url", None) if run is not None else None
    return text, chunks, url


def _last_ai_text(state: dict[str, Any]) -> str:
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "ai" and not getattr(msg, "tool_calls", None):
            content = msg.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                ]
                return "".join(parts)
    return ""


def _build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    seen: set[tuple[str, str]] = set()
    out: list[Citation] = []
    for r in chunks:
        key = (r.chunk.article_title, r.chunk.source_url)
        if key in seen:
            continue
        seen.add(key)
        out.append(Citation(title=r.chunk.article_title, url=r.chunk.source_url))
    return out


