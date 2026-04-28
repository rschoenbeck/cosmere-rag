# Containers only. Secret *versions* are added out-of-band so credential
# values never enter Terraform state or the repo. See README for the flow.
locals {
  secrets = {
    slack_bot_token  = "slack-bot-token"
    slack_app_token  = "slack-app-token"
    openai_api_key   = "openai-api-key"
    langsmith_api_key = "langsmith-api-key"
  }
}

resource "google_secret_manager_secret" "secrets" {
  for_each  = local.secrets
  project   = var.project_id
  secret_id = each.value

  replication {
    auto {}
  }

  depends_on = [google_project_service.enabled]
}
