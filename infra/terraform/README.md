# cosmere-rag Terraform module

Manages the slow-moving GCP infra for `cosmere-slack` and the `cosmere-rag-pipeline` Cloud Run Job: APIs, Artifact Registry, service accounts + IAM, Secret Manager containers, BigQuery dataset/table, and the Cloud Run service + job shells.

Image tags are **not** managed here — they're rolled by `make release-serve` / `make release-job` via `gcloud run … update --image=…`. Both Cloud Run resources have `lifecycle.ignore_changes` on the image field so subsequent plans don't try to revert the live tag.

## One-time bootstrap (state bucket)

The GCS bucket holding remote state has to exist before `terraform init`. Run once:

```sh
make bootstrap-tfstate
```

(Or, by hand: `gcloud storage buckets create gs://$PROJECT-tfstate --location=us-central1 --uniform-bucket-level-access` then `gcloud storage buckets update gs://$PROJECT-tfstate --versioning`.)

## Initial apply

```sh
cp infra/terraform/terraform.tfvars.example infra/terraform/terraform.tfvars
# edit terraform.tfvars: set project_id (and any non-default overrides)

make tf-init      # passes -backend-config="bucket=$PROJECT-tfstate"
make tf-plan
make tf-apply
```

What you get after `apply`:

- 6 GCP APIs enabled (Run, Cloud Build, Artifact Registry, Secret Manager, BigQuery, IAM).
- `cosmere-rag` Artifact Registry Docker repo in the configured region.
- `cosmere-slack-sa` and `cosmere-pipeline-sa` service accounts with least-privilege bindings.
- 4 Secret Manager secret containers — **no versions yet**.
- BigQuery dataset (`cosmere_rag` by default) + chunks table with the schema mirroring `cosmere_rag/retrieval/bigquery_store.py:42-55`.
- `cosmere-slack` Cloud Run service and `cosmere-rag-pipeline` Cloud Run Job, both running the `gcr.io/cloudrun/hello` placeholder until the first image push.

## Populate secrets

After the first apply, add credential values from `.env` (one version per secret):

```sh
echo -n "$SLACK_BOT_TOKEN"    | gcloud secrets versions add slack-bot-token   --data-file=-
echo -n "$SLACK_APP_TOKEN"    | gcloud secrets versions add slack-app-token   --data-file=-
echo -n "$OPENAI_API_KEY"     | gcloud secrets versions add openai-api-key    --data-file=-
echo -n "$LANGSMITH_API_KEY"  | gcloud secrets versions add langsmith-api-key --data-file=-
```

`-n` matters — without it the trailing newline ends up in the secret value and breaks token auth. Cloud Run resolves `version: "latest"` at container start, so a freshly-rotated secret only takes effect on the next revision; use `make restart` to force one without rebuilding.

## Common ops

| Need | Command |
|---|---|
| Plan / apply infra changes | `make tf-plan && make tf-apply` |
| Roll a new serve image | `make release-serve` (build + `gcloud run services update`) |
| Roll a new job image | `make release-job` |
| Rotate a secret | `gcloud secrets versions add … --data-file=-` then `make restart` |
| See live service config | `gcloud run services describe cosmere-slack --region us-central1` |
| Tear it all down | `make tf-destroy` (won't touch state bucket or pushed images) |

## Notes / gotchas

- **State bucket is not in TF.** Chicken-and-egg — the backend can't manage its own bucket. Recreate manually if you nuke the project.
- **Secret values are never in TF state.** Only the container resources are. If you `terraform destroy`, you'll lose the containers; the underlying credential material in `.env` still exists.
- **BigQuery vector index is runtime-managed.** `BigQueryStore._ensure_vector_index` creates it on first `add()` once the table clears 5K rows (`bigquery_store.py:200-210`). For ~1K Era 1 chunks, vector search runs as brute force — faster than IVF anyway at that scale.
- **Image field drift.** If `terraform plan` ever wants to set the image back to `gcr.io/cloudrun/hello`, the `lifecycle.ignore_changes` block fell off — re-add it before applying.
