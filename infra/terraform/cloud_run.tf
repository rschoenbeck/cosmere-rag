locals {
  service_name = "cosmere-slack"
  job_name     = "cosmere-rag-pipeline"
  # Placeholder image used on first apply. Make targets overwrite via
  # `gcloud run services|jobs update --image=...` afterwards. The
  # lifecycle.ignore_changes below keeps subsequent plans from reverting.
  placeholder_image = "gcr.io/cloudrun/hello"

  base_env = {
    GOOGLE_CLOUD_PROJECT = var.project_id
    BIGQUERY_DATASET     = var.dataset_name
    BIGQUERY_TABLE       = var.table_name
    BIGQUERY_LOCATION    = var.dataset_location
    EMBEDDING_MODEL      = var.embedding_model
    LANGSMITH_TRACING    = "true"
    LANGSMITH_PROJECT    = var.langsmith_project
  }

  service_env = merge(local.base_env, {
    SLACK_INCLUDE_TRACE_URL = "false"
  })
}

# --- Cloud Run service: cosmere-slack ---

resource "google_cloud_run_v2_service" "slack" {
  project  = var.project_id
  location = var.region
  name     = local.service_name

  ingress              = "INGRESS_TRAFFIC_INTERNAL_ONLY"
  invoker_iam_disabled = false
  deletion_protection  = false

  template {
    service_account = google_service_account.slack.email

    scaling {
      min_instance_count = 1
      max_instance_count = 1
    }

    containers {
      image = local.placeholder_image

      resources {
        cpu_idle = false
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }

      dynamic "env" {
        for_each = local.service_env
        content {
          name  = env.key
          value = env.value
        }
      }

      env {
        name = "SLACK_BOT_TOKEN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.secrets["slack_bot_token"].secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "SLACK_APP_TOKEN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.secrets["slack_app_token"].secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "OPENAI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.secrets["openai_api_key"].secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "LANGSMITH_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.secrets["langsmith_api_key"].secret_id
            version = "latest"
          }
        }
      }
    }
  }

  depends_on = [
    google_project_service.enabled,
    google_secret_manager_secret_iam_member.accessor,
    google_bigquery_dataset_iam_member.slack_data_viewer,
    google_project_iam_member.slack_bq_job_user,
  ]

  lifecycle {
    ignore_changes = [
      client,
      client_version,
      template[0].containers[0].image,
    ]
  }
}

# --- Cloud Run job: cosmere-rag-pipeline ---

resource "google_cloud_run_v2_job" "pipeline" {
  project  = var.project_id
  location = var.region
  name     = local.job_name

  deletion_protection = false

  template {
    template {
      service_account = google_service_account.pipeline.email
      max_retries     = 1
      timeout         = "3600s"

      containers {
        image = local.placeholder_image

        resources {
          limits = {
            cpu    = "2"
            memory = "2Gi"
          }
        }

        dynamic "env" {
          for_each = local.base_env
          content {
            name  = env.key
            value = env.value
          }
        }

        env {
          name = "OPENAI_API_KEY"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.secrets["openai_api_key"].secret_id
              version = "latest"
            }
          }
        }

        env {
          name = "LANGSMITH_API_KEY"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.secrets["langsmith_api_key"].secret_id
              version = "latest"
            }
          }
        }
      }
    }
  }

  depends_on = [
    google_project_service.enabled,
    google_secret_manager_secret_iam_member.accessor,
    google_bigquery_dataset_iam_member.pipeline_data_editor,
    google_project_iam_member.pipeline_bq_job_user,
  ]

  lifecycle {
    ignore_changes = [
      client,
      client_version,
      template[0].template[0].containers[0].image,
    ]
  }
}
