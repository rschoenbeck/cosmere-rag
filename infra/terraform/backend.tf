# Bucket name is supplied via partial backend config on `terraform init`:
#   terraform init -backend-config="bucket=$(gcloud config get-value project)-tfstate"
# Pre-create the bucket once via `make bootstrap-tfstate` (see infra/terraform/README.md).
terraform {
  backend "gcs" {
    prefix = "cosmere-rag"
  }
}
