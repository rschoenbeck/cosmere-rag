"""Chroma-backed implementation of the Retriever protocol.

Design notes (why things are the way they are):

- One collection per (corpus, embedding_model, version) tuple. Mixing
  embeddings from different models inside one collection is a silent
  source of bad retrieval; keep them separated at the store level.
- Embeddings are supplied by the caller, never computed inside the
  store. Keeps the embedder replaceable and the embeddings cache
  authoritative.
- Chroma metadata values must be scalars (str/int/float/bool), so list
  fields are flattened on write and reconstructed on read:
    heading_path      -> joined display string (not filterable)
    series_mentioned  -> pipe-joined string (for reconstruction)
                         plus one boolean column per series (filterable)
- Callers filter with the natural `series_mentioned` key; the store
  translates to the per-series bool columns. Keeps the public `where`
  vocabulary identical across Chroma and BigQuery backends.
- Returned score is cosine similarity in [0, 1], derived from Chroma's
  cosine *distance* as `1 - distance`. Upstream embeddings are assumed
  already L2-normalized (see `cosmere_rag.embed.embedder`).
- `add()` uses `upsert`, so re-indexing is idempotent as long as
  `chunk_id`s are stable.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from langsmith import traceable

from cosmere_rag.core.chunk import Chunk
from cosmere_rag.core.retrieved_chunk import RetrievedChunk

_HEADING_SEP = " › "
_SERIES_SEP = "|"


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_")
    return cleaned or "unknown"


def _flatten_metadata(chunk: Chunk) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "article_title": chunk.article_title,
        "heading_path": _HEADING_SEP.join(chunk.heading_path),
        "spoiler_scope": chunk.spoiler_scope,
        "series_mentioned": _SERIES_SEP.join(chunk.series_mentioned),
        "source_url": chunk.source_url,
        "content_provenance": chunk.content_provenance,
        "corpus_snapshot": chunk.corpus_snapshot,
        "ingested_at": chunk.ingested_at.isoformat(),
        "token_count": chunk.token_count,
    }
    for series in chunk.series_mentioned:
        meta[f"series_mentioned__{_slug(series)}"] = True
    return meta


def _unflatten_metadata(meta: Mapping[str, Any], chunk_id: str, text: str) -> Chunk:
    series_raw = meta.get("series_mentioned", "") or ""
    heading_raw = meta.get("heading_path", "") or ""
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        article_title=meta["article_title"],
        heading_path=heading_raw.split(_HEADING_SEP) if heading_raw else [],
        spoiler_scope=meta["spoiler_scope"],
        series_mentioned=series_raw.split(_SERIES_SEP) if series_raw else [],
        source_url=meta["source_url"],
        content_provenance=meta["content_provenance"],
        corpus_snapshot=meta["corpus_snapshot"],
        ingested_at=meta["ingested_at"],
        token_count=int(meta["token_count"]),
    )


def _translate_where(where: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not where:
        return None
    conditions: list[dict[str, Any]] = []
    for key, value in where.items():
        if key == "series_mentioned":
            values = (
                list(value["$in"])
                if isinstance(value, Mapping) and "$in" in value
                else [value]
            )
            series_conds = [{f"series_mentioned__{_slug(v)}": True} for v in values]
            conditions.append(
                series_conds[0] if len(series_conds) == 1 else {"$or": series_conds}
            )
        elif isinstance(value, Mapping) and "$in" in value:
            conditions.append({key: {"$in": list(value["$in"])}})
        else:
            conditions.append({key: {"$eq": value}})
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


class ChromaStore:
    """Retriever implementation backed by a persistent Chroma collection."""

    def __init__(self, path: Path, collection_name: str):
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must align"
            )
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[_flatten_metadata(c) for c in chunks],
            embeddings=list(embeddings),
        )

    @traceable(run_type="retriever", name="ChromaStore.query")
    def query(
        self,
        embedding: list[float],
        k: int = 8,
        where: Mapping[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=_translate_where(where),
        )
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        distances = result["distances"][0]
        out: list[RetrievedChunk] = []
        for chunk_id, text, meta, dist in zip(ids, docs, metas, distances, strict=True):
            chunk = _unflatten_metadata(meta, chunk_id=chunk_id, text=text)
            out.append(RetrievedChunk(chunk=chunk, score=1.0 - float(dist)))
        return out

    def count(self) -> int:
        return self._collection.count()