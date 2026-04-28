"""Agent tools.

`make_search_tool` returns a LangChain tool that wraps `run_retrieval_chain`
so the agent can ground answers in the existing retrieval stack. Because
the chain is `@traceable`, every tool call nests under the agent run in
LangSmith without any extra wiring.

The tool returns a `Command` that updates the graph state with both:

  - `retrieved_chunks`: raw `RetrievedChunk` objects (read by `answer()`
    after `graph.invoke()` to build the deterministic citations list).
  - `messages`: a `ToolMessage` containing the model-facing summary
    (titles, urls, scores, text) so the agent can reason about what to
    cite.

This is the documented LangGraph 1.x pattern for tools that need to
write structured data into graph state. See `cosmere_rag.agent.state`
for the matching `state_schema`.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from cosmere_rag.core.retriever import Retriever
from cosmere_rag.embed.embedder import Embedder
from cosmere_rag.retrieval.chain import run_retrieval_chain


def make_search_tool(
    retriever: Retriever,
    embedder: Embedder,
    *,
    default_k: int = 8,
    where: Mapping[str, Any] | None = None,
    collection: str | None = None,
):
    """Build a `search_coppermind` tool bound to a specific retriever/embedder."""

    @tool("search_coppermind")
    def search_coppermind(
        query: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        k: int = default_k,
    ) -> Command:
        """Search the Coppermind wiki for passages relevant to a Cosmere question.

        Use this for any factual question about characters, magic systems,
        events, or world-building. Returns up to `k` passages, each with the
        article title, source URL, the passage text, and a similarity score.
        """
        out = run_retrieval_chain(
            query,
            retriever=retriever,
            embedder=embedder,
            k=k,
            where=where,
            collection=collection,
        )
        results = out["results"]
        summary = [
            {
                "chunk_id": r.chunk.chunk_id,
                "article_title": r.chunk.article_title,
                "source_url": r.chunk.source_url,
                "score": round(r.score, 4),
                "text": r.chunk.text,
            }
            for r in results
        ]
        return Command(
            update={
                "retrieved_chunks": list(results),
                "messages": [
                    ToolMessage(
                        content=json.dumps(summary),
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return search_coppermind
