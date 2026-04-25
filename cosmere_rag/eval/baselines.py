"""Baseline embedders for sanity-checking the eval pipeline.

`NoiseEmbedder` returns L2-normalized random vectors that are
*deterministic per text* (same input → same vector across runs and
across documents/queries). That gives a stable noise baseline: if the
real embedder isn't substantially better than this on every metric, the
harness — or the model — is broken.

The embedder satisfies `langchain_core.embeddings.Embeddings`, so it
plugs into `Embedder(provider=NoiseEmbedder(...))` without any special
casing in the rest of the system.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
from langchain_core.embeddings import Embeddings


def _seed_for(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


class NoiseEmbedder(Embeddings):
    """Deterministic, L2-normalized random embeddings.

    `dim` defaults to 1536 to match `text-embedding-3-small`, so a noise
    parquet can be indexed into the same Chroma backend without changing
    upstream code. Vectors are normalized for parity with
    `cosmere_rag.embed.embedder._l2_normalize`.
    """

    def __init__(self, dim: int = 1536):
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        rng = np.random.default_rng(_seed_for(text))
        vec = rng.standard_normal(self.dim, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return vec.tolist()
        return (vec / norm).tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:  # type: ignore[override]
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
