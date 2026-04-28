from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool

from cosmere_rag.agent.agent import answer
from cosmere_rag.core.chunk import Chunk
from cosmere_rag.core.retrieved_chunk import RetrievedChunk
from cosmere_rag.embed.embedder import Embedder


class _ScriptedChatModel(BaseChatModel):
    """A chat model that replays a fixed list of AIMessages.

    Supports `bind_tools` (no-op pass-through) so `create_agent` can wire it
    up without a real provider.
    """

    responses: list[AIMessage]
    _cursor: int = 0

    def __init__(self, responses: list[AIMessage]) -> None:
        super().__init__(responses=responses)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools: Sequence[Any], **_: Any) -> "_ScriptedChatModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        idx = self._cursor
        self._cursor += 1
        if idx >= len(self.responses):
            raise AssertionError(
                f"scripted model exhausted at message #{idx}; provide more responses"
            )
        return ChatResult(generations=[ChatGeneration(message=self.responses[idx])])


def _chunk(
    *, chunk_id: str, title: str, url: str, text: str = "irrelevant"
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        article_title=title,
        heading_path=[],
        spoiler_scope="MB-Era1",
        series_mentioned=["Mistborn"],
        source_url=url,
        content_provenance="coppermind",
        corpus_snapshot="2026-04-01",
        ingested_at=datetime.now(tz=timezone.utc),
        token_count=4,
    )


class _StubEmbeddingProvider:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class _StubRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self._results = results
        self.last_args: dict[str, Any] = {}
        self.call_count = 0

    def add(self, chunks, embeddings) -> None:  # pragma: no cover
        raise NotImplementedError

    def count(self) -> int:  # pragma: no cover
        return len(self._results)

    def query(
        self,
        embedding: list[float],
        k: int = 8,
        where: Mapping[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        self.last_args = {"embedding": embedding, "k": k, "where": where}
        self.call_count += 1
        return list(self._results)


def _embedder() -> Embedder:
    return Embedder(model="stub-model", provider=_StubEmbeddingProvider())


def _tool_call_msg(query: str, k: int | None = None, call_id: str = "tc1") -> AIMessage:
    args: dict[str, Any] = {"query": query}
    if k is not None:
        args["k"] = k
    return AIMessage(
        content="",
        tool_calls=[{"name": "search_coppermind", "args": args, "id": call_id}],
    )


def test_happy_path_invokes_tool_and_returns_citations():
    chunks = [
        RetrievedChunk(
            chunk=_chunk(
                chunk_id="c1",
                title="Vin",
                url="https://example/coppermind/Vin",
                text="Vin burns steel.",
            ),
            score=0.91,
        ),
    ]
    retriever = _StubRetriever(chunks)
    model = _ScriptedChatModel(
        responses=[
            _tool_call_msg("who is Vin"),
            AIMessage(content="Vin is a Mistborn (Vin)."),
        ]
    )

    response = answer(
        "Who is Vin?",
        retriever=retriever,
        embedder=_embedder(),
        model=model,
    )

    assert retriever.call_count == 1
    assert response.answer == "Vin is a Mistborn (Vin)."
    assert len(response.citations) == 1
    assert response.citations[0].title == "Vin"
    assert response.citations[0].url == "https://example/coppermind/Vin"


def test_no_results_returns_empty_citations():
    retriever = _StubRetriever([])
    model = _ScriptedChatModel(
        responses=[
            _tool_call_msg("obscure question"),
            AIMessage(content="I couldn't find anything relevant in the Coppermind."),
        ]
    )

    response = answer(
        "Who is nobody?",
        retriever=retriever,
        embedder=_embedder(),
        model=model,
    )

    assert retriever.call_count == 1
    assert response.citations == []
    assert "couldn't find" in response.answer.lower() or "nothing" in response.answer.lower()


def test_citations_are_deduped_in_retrieval_order():
    chunks = [
        RetrievedChunk(
            chunk=_chunk(chunk_id="c1", title="Kelsier", url="https://example/Kelsier"),
            score=0.9,
        ),
        RetrievedChunk(
            chunk=_chunk(chunk_id="c2", title="Vin", url="https://example/Vin"),
            score=0.85,
        ),
        RetrievedChunk(
            chunk=_chunk(chunk_id="c3", title="Kelsier", url="https://example/Kelsier"),
            score=0.8,
        ),
    ]
    retriever = _StubRetriever(chunks)
    model = _ScriptedChatModel(
        responses=[_tool_call_msg("crew"), AIMessage(content="The crew.")]
    )

    response = answer(
        "Tell me about the crew",
        retriever=retriever,
        embedder=_embedder(),
        model=model,
    )

    assert [c.title for c in response.citations] == ["Kelsier", "Vin"]


def test_tool_k_arg_is_passed_through_to_retriever():
    chunks = [
        RetrievedChunk(
            chunk=_chunk(chunk_id="c1", title="Vin", url="https://example/Vin"),
            score=0.9,
        ),
    ]
    retriever = _StubRetriever(chunks)
    model = _ScriptedChatModel(
        responses=[
            _tool_call_msg("who is Vin", k=4),
            AIMessage(content="Vin."),
        ]
    )

    answer(
        "Who is Vin?",
        retriever=retriever,
        embedder=_embedder(),
        model=model,
    )

    assert retriever.last_args["k"] == 4


def test_tool_object_is_a_basetool():
    """Sanity check: make_search_tool returns something create_agent accepts."""
    from cosmere_rag.agent.tools import make_search_tool

    retriever = _StubRetriever([])
    tool_obj = make_search_tool(retriever, _embedder())
    assert isinstance(tool_obj, BaseTool)
    assert tool_obj.name == "search_coppermind"
