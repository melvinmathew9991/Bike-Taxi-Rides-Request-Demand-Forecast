# Deploying the demo to Google Cloud

The API runs on **Cloud Run** in `asia-south1` (Mumbai), serving the promoted
model from a **private Cloud Storage bucket**. Only aggregated files are
uploaded; the booking-level data never leaves your machine.

```
           public URL                      private
 user ──▶  Cloud Run  ──reads (read-only)──▶  Cloud Storage bucket
           (API image)                       grid · model · centres · registry
              │
              └──reads at startup──▶ Secret Manager (API key)
```

## What is uploaded, and what is not

`biketaxi stage-demo` builds the upload from an allow-list:

| File | Contents | Size |
|---|---|---|
| `Data_Prepared_<ver>.csv.gz` | requests per cluster per 30 min, **last 14 days only** | ~0.2 MB |
| `prediction_model_with_lag_<ver>.joblib` | the promoted model | ~0.7 MB |
| `pickup_cluster_model_<ver>.joblib` | the 50 cluster centres **only** | <0.01 MB |
| `model_registry.json` | the promoted entry only | <0.01 MB |

Never uploaded: `clean_data_*` (rider IDs and exact coordinates), the raw data,
and the clusterer's per-booking labels. Staging refuses a grid with any column
outside the aggregated set. Fourteen days is enough: forecasting reads seven,
and a forecast from the trimmed history matches one from the full history
exactly.

## One-time setup

You need `gcloud` logged in (`gcloud auth login`) and a billing account (the
free tier covers this demo). Project IDs are global, so pick a unique one.

```bash
PROJECT_ID=bike-taxi-demand-demo-<something-unique>
gcloud projects create "$PROJECT_ID" --name "Bike-taxi demand demo"
gcloud billing accounts list                     # note the ACCOUNT_ID
gcloud billing projects link "$PROJECT_ID" --billing-account <ACCOUNT_ID>
```

Strongly recommended: a budget alert, so a surprise shows up as an email rather
than an invoice. In the console: **Billing → Budgets & alerts → Create budget**,
scoped to this project, e.g. ₹500 with alerts at 50/90/100%.

## Deploy

From the repository root, in Git Bash (Windows) or any shell with `bash`:

```bash
PROJECT_ID=<your-project> PYTHON="py -3.11" bash deploy/gcp_deploy.sh
```

(`PYTHON` defaults to `python3`; on this Windows machine the interpreter with
the project's dependencies is `py -3.11`.)

The script enables the APIs, builds the image with **Cloud Build** (no local
Docker needed), stages and uploads the files, creates the API key, a runtime
service account that can read only the bucket and the key, and deploys. It
ends by checking `/health`, that a request without the key gets 401, and that
`/model` with the key reports the expected model. Re-running it is safe and
deploys a new revision.

Get the key to share with whoever you send the link to:

```bash
gcloud secrets versions access latest --secret biketaxi-api-key --project "$PROJECT_ID"
```

## Updating

| Change | Do this |
|---|---|
| New model trained and promoted locally | re-run the deploy script |
| Code change | re-run the deploy script |
| Rotate the key | `openssl rand -base64 33 \| tr -d '/+=\r\n' \| gcloud secrets versions add biketaxi-api-key --data-file=- --project "$PROJECT_ID"`, then re-run the script so a new revision reads it |
| Open the API to everyone (once the data source's terms allow it) | `gcloud run services update bike-taxi-forecast --region asia-south1 --remove-secrets BIKETAXI_API_KEY --project "$PROJECT_ID"` |

## Cost

With `--min-instances 0` the service scales to zero when idle, and
`--max-instances 2` caps it if the URL is hammered. Expected cost for a demo:
Cloud Run and Cloud Build within the free tier; storage for ~1 MB in Mumbai,
fractions of a rupee a month; Artifact Registry about ₹10 a month per GB of
images beyond the free 0.5 GB - delete old images to stay near zero:

```bash
gcloud artifacts docker images list asia-south1-docker.pkg.dev/$PROJECT_ID/bike-taxi/api --project "$PROJECT_ID"
```

The first request after idle takes a few seconds (a cold start loads the
model and grid).

## The demo model ages

The data ends on 2021-03-26, so every forecast is for 2021-03-27 onwards, and
28 days after the model was trained `/model` and every forecast start reporting
it as stale. Both are accurate: a frozen demo is not a live service.

## Tear down

Everything lives in the one project:

```bash
gcloud projects delete "$PROJECT_ID"
```

## Troubleshooting

- **`PERMISSION_DENIED` on the first run in a new project, as the project
  owner.** Permissions on a just-enabled API take a minute or two to
  propagate. The first deployment hit this twice, at Artifact Registry and at
  Cloud Build. Wait a minute and re-run; the script resumes where it stopped.
- **Running from Git Bash on Windows.** Two things the first deployment found
  and the script now handles: Git Bash rewrites Unix-looking arguments, which
  turned the `/app/output` mount path into a Windows path (only the volume
  arguments are exempted - turning conversion off altogether breaks gcloud's
  own wrapper); and Windows `openssl` ends its output with CRLF, which put a
  carriage return inside the first API key.
- **Cloud Build fails with a permission error.** New projects build as the
  Compute Engine default service account. Grant it
  `roles/artifactregistry.writer`, `roles/storage.objectViewer` and
  `roles/logging.logWriter` in **IAM**, and re-run.
- **`/health` reports `ready: false`.** Check the logs (`gcloud run services
  logs read bike-taxi-forecast --region asia-south1`). The usual cause is a
  model saved by a newer xgboost or scikit-learn than the image installs.
