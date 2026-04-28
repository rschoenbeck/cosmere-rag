resource "google_service_account" "slack" {
  project      = var.project_id
  account_id   = "cosmere-slack-sa"
  display_name = "cosmere-slack Cloud Run service identity"

  depends_on = [google_project_service.enabled]
}

resource "google_service_account" "pipeline" {
  project      = var.project_id
  account_id   = "cosmere-pipeline-sa"
  display_name = "cosmere-rag-pipeline Cloud Run Job identity"

  depends_on = [google_project_service.enabled]
}

# --- BigQuery: jobUser is project-scoped (no dataset-scoped equivalent) ---

resource "google_project_iam_member" "slack_bq_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.slack.email}"
}

resource "google_project_iam_member" "pipeline_bq_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.pipeline.email}"
}

# --- BigQuery: dataset-scoped read/write ---

resource "google_bigquery_dataset_iam_member" "slack_data_viewer" {
  project    = var.project_id
  dataset_id = google_bigquery_dataset.cosmere_rag.dataset_id
  role       = "roles/bigquery.dataViewer"
  member     = "serviceAccount:${google_service_account.slack.email}"
}

resource "google_bigquery_dataset_iam_member" "pipeline_data_editor" {
  project    = var.project_id
  dataset_id = google_bigquery_dataset.cosmere_rag.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.pipeline.email}"
}

# --- Secret Manager: per-secret accessor bindings (least privilege) ---

locals {
  # Both SAs need OpenAI + LangSmith. Slack-* are the bot's alone.
  secret_accessors = {
    slack_bot_token  = [google_service_account.slack.email]
    slack_app_token  = [google_service_account.slack.email]
    openai_api_key   = [google_service_account.slack.email, google_service_account.pipeline.email]
    langsmith_api_key = [google_service_account.slack.email, google_service_account.pipeline.email]
  }

  secret_accessor_bindings = merge([
    for secret_key, sa_emails in local.secret_accessors : {
      for email in sa_emails :
      "${secret_key}:${email}" => {
        secret_key = secret_key
        email      = email
      }
    }
  ]...)
}

resource "google_secret_manager_secret_iam_member" "accessor" {
  for_each  = local.secret_accessor_bindings
  project   = var.project_id
  secret_id = google_secret_manager_secret.secrets[each.value.secret_key].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${each.value.email}"
}
