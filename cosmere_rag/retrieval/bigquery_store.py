"""BigQuery-backed implementation of the Retriever protocol.

Design notes (why things are the way they are):

- One table per (corpus, embedding_model, version) tuple. Same contract
  as Chroma: mixing embedding models inside one table is a silent source
  of bad retrieval.
- Embeddings are supplied by the caller, never computed inside the store.
- BigQuery handles arrays natively, so `heading_path` and
  `series_mentioned` are stored as `ARRAY<STRING>` rather than flattened.
  That keeps the reverse-mapping a one-liner and the schema honest.
- `where` filters are translated to parameterized SQL as a *post-filter*
  on `VECTOR_SEARCH` output. For our corpus and `k<=50` this is fine;
  if filters ever become very selective, bump `top_k` and trim.
- Score normalization matches Chroma: with `COSINE` distance and
  L2-normalized embeddings, `score = 1 - distance` is cosine similarity
  in [0, 1], higher = better.
- `add()` is idempotent: rows are loaded into a per-call staging table,
  MERGEd into the target on `chunk_id`, then the staging table is
  dropped (safety-net expiration keeps leaks self-cleaning).
- Table + vector index are created lazily on first `add()` so that
  query-only deployments don't need DDL permissions.

Deployment note: the current loader path is `load_table_from_json`, which
uploads the batch directly via a free load job — fine for this corpus.
When embeddings move into versioned GCS storage we can add a
`load_table_from_uri` path without changing the `Retriever` surface.
"""
from __future__ import annotations

import secrets
from collections.abc import Mapping, Sequence
from typing import Any

from google.cloud import bigquery
from langsmith import traceable

from cosmere_rag.core.chunk import Chunk
from cosmere_rag.core.retrieved_chunk import RetrievedChunk


_TABLE_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("chunk_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("text", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
    bigquery.SchemaField("article_title", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("heading_path", "STRING", mode="REPEATED"),
    bigquery.SchemaField("spoiler_scope", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("series_mentioned", "STRING", mode="REPEATED"),
    bigquery.SchemaField("source_url", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("content_provenance", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("corpus_snapshot", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("ingested_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("token_count", "INT64", mode="REQUIRED"),
]

_MERGE_COLUMNS = [f.name for f in _TABLE_SCHEMA]


def _chunk_to_row(chunk: Chunk, embedding: Sequence[float]) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "embedding": [float(x) for x in embedding],
        "article_title": chunk.article_title,
        "heading_path": list(chunk.heading_path),
        "spoiler_scope": chunk.spoiler_scope,
        "series_mentioned": list(chunk.series_mentioned),
        "source_url": chunk.source_url,
        "content_provenance": chunk.content_provenance,
        "corpus_snapshot": chunk.corpus_snapshot,
        "ingested_at": chunk.ingested_at.isoformat(),
        "token_count": chunk.token_count,
    }


def _row_to_chunk(row: Mapping[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"],
        text=row["text"],
        article_title=row["article_title"],
        heading_path=list(row.get("heading_path") or []),
        spoiler_scope=row["spoiler_scope"],
        series_mentioned=list(row.get("series_mentioned") or []),
        source_url=row["source_url"],
        content_provenance=row["content_provenance"],
        corpus_snapshot=row["corpus_snapshot"],
        ingested_at=row["ingested_at"],
        token_count=int(row["token_count"]),
    )


def translate_where(
    where: Mapping[str, Any] | None,
) -> tuple[str, list[bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter]]:
    """Translate the shared `where` vocabulary into a SQL fragment + params.

    Returns `("", [])` when `where` is empty. Otherwise returns a clause
    beginning with `WHERE ` that applies to `base.<col>` references
    inside a `VECTOR_SEARCH(...)` result.

    Vocabulary (same as Chroma):
      - {col: value}            -> base.col = @p      (scalar equality)
      - {col: {"$in": [...]}}   -> base.col IN UNNEST(@p)
      - {"series_mentioned": v} -> @p IN UNNEST(base.series_mentioned)
      - {"series_mentioned": {"$in": [...]}}
            -> EXISTS (SELECT 1 FROM UNNEST(base.series_mentioned) s
                       WHERE s IN UNNEST(@p))
    """
    if not where:
        return "", []

    fragments: list[str] = []
    params: list[bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter] = []

    for i, (key, value) in enumerate(where.items()):
        name = f"w{i}"
        is_in = isinstance(value, Mapping) and "$in" in value
        in_values = list(value["$in"]) if is_in else None

        if key == "series_mentioned":
            if is_in:
                params.append(bigquery.ArrayQueryParameter(name, "STRING", in_values))
                fragments.append(
                    f"EXISTS (SELECT 1 FROM UNNEST(base.series_mentioned) AS s "
                    f"WHERE s IN UNNEST(@{name}))"
                )
            else:
                params.append(bigquery.ScalarQueryParameter(name, "STRING", value))
                fragments.append(f"@{name} IN UNNEST(base.series_mentioned)")
            continue

        if is_in:
            bq_type = _infer_array_type(in_values)
            params.append(bigquery.ArrayQueryParameter(name, bq_type, in_values))
            fragments.append(f"base.{key} IN UNNEST(@{name})")
        else:
            bq_type = _infer_scalar_type(value)
            params.append(bigquery.ScalarQueryParameter(name, bq_type, value))
            fragments.append(f"base.{key} = @{name}")

    return "WHERE " + " AND ".join(fragments), params


def _infer_scalar_type(value: Any) -> str:
    if isinstance(value, bool):
        return "BOOL"
    if isinstance(value, int):
        return "INT64"
    if isinstance(value, float):
        return "FLOAT64"
    return "STRING"


def _infer_array_type(values: Sequence[Any]) -> str:
    if not values:
        return "STRING"
    return _infer_scalar_type(values[0])


class BigQueryStore:
    """Retriever implementation backed by a BigQuery table + vector index."""

    def __init__(
        self,
        project: str,
        dataset: str,
        table_name: str,
        *,
        location: str = "US",
        client: bigquery.Client | None = None,
    ):
        self._project = project
        self._dataset = dataset
        self._table_name = table_name
        self._location = location
        self._client = client or bigquery.Client(project=project, location=location)

    @property
    def _table_ref(self) -> str:
        return f"`{self._project}.{self._dataset}.{self._table_name}`"

    def _ensure_table(self) -> None:
        self._client.create_dataset(
            bigquery.Dataset(f"{self._project}.{self._dataset}"),
            exists_ok=True,
        )
        table = bigquery.Table(
            f"{self._project}.{self._dataset}.{self._table_name}",
            schema=_TABLE_SCHEMA,
        )
        self._client.create_table(table, exists_ok=True)

    # BigQuery rejects CREATE VECTOR INDEX on tables with fewer rows than
    # this (IVF needs enough rows to form meaningful centroids). Below
    # the threshold, VECTOR_SEARCH already falls back to brute force,
    # which is faster than an index anyway at this scale.
    _MIN_ROWS_FOR_IVF_INDEX = 5000

    def _ensure_vector_index(self) -> None:
        # Must run *after* rows exist: BQ infers the embedding dimension
        # from the data, and needs enough rows for IVF centroids.
        if self.count() < self._MIN_ROWS_FOR_IVF_INDEX:
            return
        index_name = f"{self._table_name}__vec_idx"
        self._client.query(
            f"CREATE VECTOR INDEX IF NOT EXISTS `{index_name}` "
            f"ON {self._table_ref}(embedding) "
            f"OPTIONS(index_type='IVF', distance_type='COSINE')"
        ).result()

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

        self._ensure_table()

        staging_name = f"{self._table_name}__staging_{secrets.token_hex(4)}"
        staging_fqn = f"{self._project}.{self._dataset}.{staging_name}"
        staging_ref = f"`{staging_fqn}`"

        rows = [_chunk_to_row(c, e) for c, e in zip(chunks, embeddings, strict=True)]

        try:
            staging = bigquery.Table(staging_fqn, schema=_TABLE_SCHEMA)
            # One-hour safety net: if MERGE fails and the drop is skipped,
            # BigQuery will garbage-collect the staging table itself.
            staging.expires = _one_hour_from_now()
            self._client.create_table(staging)

            load_job = self._client.load_table_from_json(
                rows,
                staging_fqn,
                job_config=bigquery.LoadJobConfig(
                    schema=_TABLE_SCHEMA,
                    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
                ),
            )
            load_job.result()

            update_cols = [c for c in _MERGE_COLUMNS if c != "chunk_id"]
            set_clause = ", ".join(f"{c} = source.{c}" for c in update_cols)
            insert_cols = ", ".join(_MERGE_COLUMNS)
            insert_vals = ", ".join(f"source.{c}" for c in _MERGE_COLUMNS)

            merge_sql = (
                f"MERGE {self._table_ref} AS target "
                f"USING {staging_ref} AS source "
                f"ON target.chunk_id = source.chunk_id "
                f"WHEN MATCHED THEN UPDATE SET {set_clause} "
                f"WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})"
            )
            self._client.query(merge_sql).result()
        finally:
            self._client.delete_table(staging_fqn, not_found_ok=True)

        self._ensure_vector_index()

    @traceable(run_type="retriever", name="BigQueryStore.query")
    def query(
        self,
        embedding: list[float],
        k: int = 8,
        where: Mapping[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        where_sql, where_params = translate_where(where)

        sql = (
            "SELECT base.chunk_id, base.text, base.article_title, base.heading_path, "
            "base.spoiler_scope, base.series_mentioned, base.source_url, "
            "base.content_provenance, base.corpus_snapshot, base.ingested_at, "
            "base.token_count, distance "
            "FROM VECTOR_SEARCH("
            f"TABLE {self._table_ref}, 'embedding', "
            "(SELECT @query_embedding AS embedding), "
            "top_k => @k, distance_type => 'COSINE') "
            f"{where_sql} "
            "ORDER BY distance ASC"
        )

        params: list[
            bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter
        ] = [
            bigquery.ArrayQueryParameter(
                "query_embedding", "FLOAT64", [float(x) for x in embedding]
            ),
            bigquery.ScalarQueryParameter("k", "INT64", k),
        ]
        params.extend(where_params)

        job_config = bigquery.QueryJobConfig(query_parameters=params)
        rows = list(self._client.query(sql, job_config=job_config).result())

        out: list[RetrievedChunk] = []
        for row in rows:
            chunk = _row_to_chunk(row)
            out.append(RetrievedChunk(chunk=chunk, score=1.0 - float(row["distance"])))
        return out

    def count(self) -> int:
        sql = f"SELECT COUNT(*) AS n FROM {self._table_ref}"
        row = next(iter(self._client.query(sql).result()))
        return int(row["n"])


def _one_hour_from_now():
    from datetime import datetime, timedelta, timezone

    return datetime.now(tz=timezone.utc) + timedelta(hours=1)
