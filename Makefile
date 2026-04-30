PROJECT     ?= $(shell gcloud config get-value project)
REGION      ?= us-central1
REPO        ?= cosmere-rag
SERVE_IMAGE := $(REGION)-docker.pkg.dev/$(PROJECT)/$(REPO)/serve
JOB_IMAGE   := $(REGION)-docker.pkg.dev/$(PROJECT)/$(REPO)/job
TAG         ?= $(shell git rev-parse --short HEAD)

# Chains all three pipeline CLIs in one Cloud Run Job execution so the
# intermediate /tmp parquet survives across ingest → embed → index.
COPPERMIND_SHA := 2a1945c24f48c313a32b20483a8160e86aa1c047
COPPERMIND_REPO := https://github.com/Malthemester/CoppermindScraper

PIPELINE_SH := git clone --filter=blob:none $(COPPERMIND_REPO) /tmp/coppermind-mirror \
  && git -C /tmp/coppermind-mirror fetch --depth=1 origin $(COPPERMIND_SHA) \
  && git -C /tmp/coppermind-mirror checkout --detach $(COPPERMIND_SHA) \
  && cosmere-ingest-coppermind --corpus-dir /tmp/coppermind-mirror/Cosmere --out /tmp/era1.jsonl \
  && cosmere-embed --chunks /tmp/era1.jsonl --out /tmp/era1.parquet \
  && cosmere-index --backend bigquery --chunks /tmp/era1.jsonl --embeddings /tmp/era1.parquet --collection era1

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
	gcloud builds submit --config=cloudbuild.yaml \
	  --substitutions=_IMAGE=$(SERVE_IMAGE):$(TAG),_DOCKERFILE=Dockerfile.serve .

build-job:
	gcloud builds submit --config=cloudbuild.yaml \
	  --substitutions=_IMAGE=$(JOB_IMAGE):$(TAG),_DOCKERFILE=Dockerfile.job .

## Roll image onto the TF-managed Cloud Run service / job
deploy-serve:
	gcloud run services update cosmere-slack \
	  --image=$(SERVE_IMAGE):$(TAG) \
	  --region=$(REGION) \
	  --min-instances=1 \
	  --max-instances=1

deploy-job:
	gcloud run jobs update cosmere-rag-pipeline \
	  --image=$(JOB_IMAGE):$(TAG) \
	  --region=$(REGION) \
	  --command=sh \
	  --args="-c,$(PIPELINE_SH)"

## Most common iteration: build + deploy in one step
release-serve: build-serve deploy-serve

release-job: build-job deploy-job

## Execute the pipeline job as configured by the last deploy-job
run-pipeline:
	gcloud run jobs execute cosmere-rag-pipeline \
	  --region=$(REGION) \
	  --wait

## Ops
logs:
	gcloud beta run services logs tail cosmere-slack --region=$(REGION)

logs-job:
	gcloud run jobs logs read cosmere-rag-pipeline --region=$(REGION) --limit=200

## Force a new revision after a secret rotation — no rebuild needed.
restart:
	gcloud run services update cosmere-slack \
	  --region=$(REGION) \
	  --update-env-vars=RESTART_NONCE=$(shell date +%s)
