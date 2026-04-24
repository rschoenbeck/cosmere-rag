"""Stable identity for chunks.

`chunk_id` is a deterministic hash of (article_title, heading_path, text),
so re-ingestion with unchanged input produces the same id and the
embeddings parquet can be upserted idempotently.

`chunk_text_hash` covers only `text`, so the embed CLI can detect "same
chunk id, text changed" and re-embed just that row.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence


def compute_chunk_id(article_title: str, heading_path: Sequence[str], text: str) -> str:
    payload = json.dumps(
        {
            "article_title": article_title,
            "heading_path": list(heading_path),
            "text": text,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def compute_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()