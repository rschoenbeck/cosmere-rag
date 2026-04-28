# Schema mirrors cosmere_rag/retrieval/bigquery_store.py:42-55. Keep in
# sync if that schema ever changes — drift here is a silent retrieval bug.
resource "google_bigquery_dataset" "cosmere_rag" {
  project    = var.project_id
  dataset_id = var.dataset_name
  location   = var.dataset_location
  description = "Cosmere RAG corpus + embeddings (one table per corpus + embedding model)."

  depends_on = [google_project_service.enabled]
}

resource "google_bigquery_table" "chunks" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.cosmere_rag.dataset_id
  table_id            = var.table_name
  deletion_protection = true

  schema = jsonencode([
    { name = "chunk_id",          type = "STRING",    mode = "REQUIRED" },
    { name = "text",              type = "STRING",    mode = "REQUIRED" },
    { name = "embedding",         type = "FLOAT64",   mode = "REPEATED" },
    { name = "article_title",     type = "STRING",    mode = "REQUIRED" },
    { name = "heading_path",      type = "STRING",    mode = "REPEATED" },
    { name = "spoiler_scope",     type = "STRING",    mode = "REQUIRED" },
    { name = "series_mentioned",  type = "STRING",    mode = "REPEATED" },
    { name = "source_url",        type = "STRING",    mode = "REQUIRED" },
    { name = "content_provenance", type = "STRING",   mode = "REQUIRED" },
    { name = "corpus_snapshot",   type = "STRING",    mode = "REQUIRED" },
    { name = "ingested_at",       type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "token_count",       type = "INT64",     mode = "REQUIRED" },
  ])
}
