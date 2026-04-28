output "service_name" {
  description = "Cloud Run service name (used by `make deploy-serve`)."
  value       = google_cloud_run_v2_service.slack.name
}

output "service_url" {
  description = "Cloud Run service URL (internal-only ingress, primarily for the health endpoint)."
  value       = google_cloud_run_v2_service.slack.uri
}

output "job_name" {
  description = "Cloud Run Job name (used by `make deploy-job` and `make run-pipeline`)."
  value       = google_cloud_run_v2_job.pipeline.name
}

output "slack_service_account" {
  description = "Email of the Slack service identity. Verify BigQuery audit logs against this."
  value       = google_service_account.slack.email
}

output "pipeline_service_account" {
  description = "Email of the pipeline (Job) service identity."
  value       = google_service_account.pipeline.email
}

output "artifact_registry_repo" {
  description = "Fully-qualified Artifact Registry repo. Image tags push under <repo>/{serve,job}:<tag>."
  value       = "${google_artifact_registry_repository.cosmere_rag.location}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.cosmere_rag.repository_id}"
}

output "bigquery_table" {
  description = "Fully-qualified BigQuery table for chunks + embeddings."
  value       = "${var.project_id}.${google_bigquery_dataset.cosmere_rag.dataset_id}.${google_bigquery_table.chunks.table_id}"
}

output "secret_ids" {
  description = "Secret Manager secret IDs (containers only — populate versions out of band)."
  value       = { for k, s in google_secret_manager_secret.secrets : k => s.secret_id }
}
