resource "google_artifact_registry_repository" "cosmere_rag" {
  project       = var.project_id
  location      = var.region
  repository_id = var.image_repo
  format        = "DOCKER"
  description   = "Container images for cosmere-rag (serve + job)."

  depends_on = [google_project_service.enabled]
}
