PROJECT     ?= $(shell gcloud config get-value project)
REGION      ?= us-central1
REPO        ?= cosmere-rag
SERVE_IMAGE := $(REGION)-docker.pkg.dev/$(PROJECT)/$(REPO)/serve
JOB_IMAGE   := $(REGION)-docker.pkg.dev/$(PROJECT)/$(REPO)/job
TAG         ?= $(shell git rev-parse --short HEAD)

# Chains all three pipeline CLIs in one Cloud Run Job execution so the
# intermediate /tmp parquet survives across ingest → embed → index.
PIPELINE_SH := bash scripts/fetch_corpus.sh \
  && cosmere-ingest-coppermind --corpus-dir data/coppermind-mirror/Cosmere --out /tmp/era1.jsonl \
  && cosmere-embed --chunks /tmp/era1.jsonl --out /tmp/era1.parquet \
  && cosmere-index --backend bigquery --chunks /tmp/era1.jsonl --embeddings /tmp/era1.parquet

.PHONY: bootstrap-tfstate \
        tf-init tf-plan tf-apply tf-destroy \
        build-serve build-job \
        deploy-serve deploy-job \
        release-serve release-job \
        run-pipeline logs logs-job restart

## One-time: create GCS bucket for Terraform remote state (idempotent).
bootstrap-tfstate:
	gcloud storage buckets create gs://$(PROJECT)-tfstate \
	  --location=$(REGION) \
	  --uniform-bucket-level-access || true
	gcloud storage buckets update gs://$(PROJECT)-tfstate --versioning

## Terraform wrappers
tf-init:
	cd infra/terraform && terraform init

tf-plan:
	cd infra/terraform && terraform plan

tf-apply:
	cd infra/terraform && terraform apply

tf-destroy:
	cd infra/terraform && terraform destroy

## Build images via Cloud Build and push to Artifact Registry
build-serve:
	gcloud builds submit --tag $(SERVE_IMAGE):$(TAG) -f Dockerfile.serve .

build-job:
	gcloud builds submit --tag $(JOB_IMAGE):$(TAG) -f Dockerfile.job .

## Roll image onto the TF-managed Cloud Run service / job
deploy-serve:
	gcloud run services update cosmere-slack \
	  --image=$(SERVE_IMAGE):$(TAG) \
	  --region=$(REGION)

deploy-job:
	gcloud run jobs update cosmere-rag-pipeline \
	  --image=$(JOB_IMAGE):$(TAG) \
	  --region=$(REGION)

## Most common iteration: build + deploy in one step
release-serve: build-serve deploy-serve

release-job: build-job deploy-job

## Execute the full build pipeline in Cloud Run Job (fetch → ingest → embed → index)
run-pipeline:
	gcloud run jobs execute cosmere-rag-pipeline \
	  --region=$(REGION) \
	  --wait \
	  --command=sh \
	  --args="-c,$(PIPELINE_SH)"

## Ops
logs:
	gcloud run services logs tail cosmere-slack --region=$(REGION)

logs-job:
	gcloud run jobs logs read cosmere-rag-pipeline --region=$(REGION) --limit=200

## Force a new revision after a secret rotation — no rebuild needed.
restart:
	gcloud run services update cosmere-slack \
	  --region=$(REGION) \
	  --update-env-vars=RESTART_NONCE=$(shell date +%s)
