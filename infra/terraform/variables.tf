variable "project_id" {
  description = "GCP project ID hosting cosmere-rag."
  type        = string
}

variable "region" {
  description = "Region for Cloud Run, Artifact Registry, and Cloud Build."
  type        = string
  default     = "us-central1"
}

variable "dataset_name" {
  description = "BigQuery dataset for the corpus + embeddings table."
  type        = string
  default     = "cosmere_rag"
}

variable "dataset_location" {
  description = "BigQuery multi-region or region for the dataset. Cloud Run region must be inside this."
  type        = string
  default     = "US"
}

variable "table_name" {
  description = "BigQuery table for chunks + embeddings (one table per corpus + embedding model)."
  type        = string
  default     = "era1"
}

variable "image_repo" {
  description = "Artifact Registry Docker repo name (holds both serve and job images)."
  type        = string
  default     = "cosmere-rag"
}

variable "embedding_model" {
  description = "OpenAI embedding model used at index + query time. Must match what `cosmere-index` was run with."
  type        = string
  default     = "text-embedding-3-small"
}

variable "langsmith_project" {
  description = "LangSmith project name for prod tracing. Kept separate from local-dev project."
  type        = string
  default     = "cosmere-rag"
}
