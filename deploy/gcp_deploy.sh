#!/usr/bin/env bash
# Deploy the forecast API to Cloud Run, serving from a private bucket.
#
#   PROJECT_ID=<your-project> bash deploy/gcp_deploy.sh
#
# Safe to re-run: every resource is created only if missing, and each run
# builds a new image, re-uploads the staged files and deploys a new revision.
# See deploy/README.md for the one-time project setup this assumes.
#
# What ends up where:
#   Artifact Registry (private)  the image - code only, no data
#   Cloud Storage (private)      aggregated files from scripts/stage_demo_output.py,
#                                mounted read-only at /app/output
#   Secret Manager               the API key, read by the service at startup
#   Cloud Run (public URL)       the API; every endpoint but /health needs the key
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID to the GCP project to deploy into}"
REGION="${REGION:-asia-south1}"
SERVICE="${SERVICE:-bike-taxi-forecast}"
REPO="${REPO:-bike-taxi}"
BUCKET="${BUCKET:-${PROJECT_ID}-demo-artifacts}"
SECRET="${SECRET:-biketaxi-api-key}"
RUNTIME_SA_NAME="${RUNTIME_SA_NAME:-bike-taxi-api}"
# Unquoted where used, so PYTHON="py -3.11" works.
PYTHON="${PYTHON:-python3}"
STAGING="${STAGING:-deploy/.staging}"

RUNTIME_SA="${RUNTIME_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
TAG="$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/api:${TAG}"
G=(gcloud --project "$PROJECT_ID" --quiet)

step() { printf '\n==> %s\n' "$*"; }

step "Enabling APIs"
"${G[@]}" services enable run.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com secretmanager.googleapis.com storage.googleapis.com

step "Artifact Registry repository ${REPO}"
"${G[@]}" artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1 \
  || "${G[@]}" artifacts repositories create "$REPO" --location "$REGION" \
       --repository-format docker --description "Bike-taxi demand forecast images"

step "Building ${IMAGE} with Cloud Build (no local Docker needed)"
"${G[@]}" builds submit --tag "$IMAGE" .

step "Staging aggregated files (never the booking-level data)"
$PYTHON scripts/stage_demo_output.py --out "$STAGING"

step "Private bucket gs://${BUCKET}"
"${G[@]}" storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1 \
  || "${G[@]}" storage buckets create "gs://${BUCKET}" --location "$REGION" \
       --uniform-bucket-level-access --public-access-prevention
# Mirror exactly: a file no longer staged is removed from the bucket too.
"${G[@]}" storage rsync "$STAGING" "gs://${BUCKET}" --recursive \
  --delete-unmatched-destination-objects

step "API key in Secret Manager (${SECRET})"
if ! "${G[@]}" secrets describe "$SECRET" >/dev/null 2>&1; then
  # Generated here and never printed; read it back with the command at the end.
  openssl rand -base64 33 | tr -d '/+=\n' \
    | "${G[@]}" secrets create "$SECRET" --replication-policy automatic --data-file=-
fi

step "Runtime service account ${RUNTIME_SA}"
"${G[@]}" iam service-accounts describe "$RUNTIME_SA" >/dev/null 2>&1 \
  || "${G[@]}" iam service-accounts create "$RUNTIME_SA_NAME" \
       --display-name "Bike-taxi forecast API runtime"
# Read the bucket and the one secret - nothing else.
"${G[@]}" storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member "serviceAccount:${RUNTIME_SA}" --role roles/storage.objectViewer >/dev/null
"${G[@]}" secrets add-iam-policy-binding "$SECRET" \
  --member "serviceAccount:${RUNTIME_SA}" --role roles/secretmanager.secretAccessor >/dev/null

step "Deploying ${SERVICE} to Cloud Run"
# --allow-unauthenticated makes the URL reachable; the API key is what guards it.
# max-instances caps the bill if the URL gets hammered.
"${G[@]}" run deploy "$SERVICE" \
  --image "$IMAGE" --region "$REGION" --port 8000 \
  --cpu 1 --memory 1Gi --min-instances 0 --max-instances 2 --concurrency 20 \
  --execution-environment gen2 \
  --service-account "$RUNTIME_SA" \
  --allow-unauthenticated \
  --set-secrets "BIKETAXI_API_KEY=${SECRET}:latest" \
  --add-volume "name=artifacts,type=cloud-storage,bucket=${BUCKET},readonly=true" \
  --add-volume-mount "volume=artifacts,mount-path=/app/output"

URL="$("${G[@]}" run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')"
KEY="$("${G[@]}" secrets versions access latest --secret "$SECRET")"

step "Checking the deployment"
curl -fsS "${URL}/health"; echo
code="$(curl -s -o /dev/null -w '%{http_code}' "${URL}/forecast?steps=2")"
echo "without key: HTTP ${code} (expect 401)"
curl -fsS -H "X-API-Key: ${KEY}" "${URL}/model" \
  | $PYTHON -c "import json,sys; b=json.load(sys.stdin); print('serving', b['model_name'], '| gate MASE', b['baseline_mase'], '| history to', b['history_ends_at'])"

cat <<EOF

Deployed: ${URL}
  Docs:      ${URL}/docs
  Key:       gcloud secrets versions access latest --secret ${SECRET} --project ${PROJECT_ID}
  Forecast:  curl -H "X-API-Key: <key>" "${URL}/forecast?steps=48&cluster=7"
EOF
