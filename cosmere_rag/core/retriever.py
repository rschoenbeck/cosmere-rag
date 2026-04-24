from typing import Protocol, Sequence, Mapping, Any

from cosmere_rag.core.chunk import Chunk
from cosmere_rag.core.retrieved_chunk import RetrievedChunk


class Retriever(Protocol):
    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[list[float]]) -> None: ...

    def query(
        self,
        embedding: list[float],
        k: int = 8,
        where: Mapping[str, Any] | None = None,   # metadata filter
    ) -> list[RetrievedChunk]: ...

    def count(self) -> int: ...