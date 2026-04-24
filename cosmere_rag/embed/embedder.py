"""Embedding provider wrapper.

Thin layer over a LangChain `Embeddings` implementation that adds the two
properties our retrieval layer depends on:

1. Batching on top of whatever the provider does internally — defensive
   cap so a huge corpus can't accidentally blow past provider limits.
2. L2-normalization at write time, so `score = 1 - cosine_distance`
   gives cosine similarity in [0, 1] across both retrieval backends.

The provider is injected, so tests can pass a fake `Embeddings` instead
of mocking HTTP. Default is OpenAI's `text-embedding-3-small`.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


@dataclass(frozen=True)
class EmbeddingResult:
    embeddings: list[list[float]]
    model: str
    model_version: str


class Embedder:
    MAX_INPUTS_PER_REQUEST = 256

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        provider: Embeddings | None = None,
    ):
        self.model = model
        self.provider: Embeddings = provider or OpenAIEmbeddings(model=model)

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        vecs: list[list[float]] = []
        for start in range(0, len(texts), self.MAX_INPUTS_PER_REQUEST):
            batch = list(texts[start : start + self.MAX_INPUTS_PER_REQUEST])
            for raw in self.provider.embed_documents(batch):
                vecs.append(_l2_normalize(raw))
        return EmbeddingResult(embeddings=vecs, model=self.model, model_version=self.model)

    def embed_query(self, text: str) -> list[float]:
        return _l2_normalize(self.provider.embed_query(text))


def _l2_normalize(vec: Sequence[float]) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr.tolist()
    return (arr / norm).tolist()