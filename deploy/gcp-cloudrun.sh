#!/bin/bash
# ── Deploy Cascade Predict to Google Cloud Run ───────────────────
# Run this from your LOCAL machine (needs gcloud CLI installed).
#
# Prerequisites:
#   1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install
#   2. gcloud auth login
#   3. gcloud config set project YOUR_PROJECT_ID
#   4. Enable APIs: gcloud services enable run.googleapis.com containerregistry.googleapis.com
#
# Cloud Run auto-scales: 0 instances when idle, up to N under load.
# Each instance handles multiple users. Set max-instances based on budget.
# ─────────────────────────────────────────────────────────────────

set -e

PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"         # change to your preferred region
SERVICE_NAME="cascade-predict"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "=== Building Docker Image ==="
gcloud builds submit --tag ${IMAGE}

echo "=== Deploying to Cloud Run ==="
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE} \
    --platform managed \
    --region ${REGION} \
    --port 8501 \
    --memory 2Gi \
    --cpu 2 \
    --min-instances 1 \
    --max-instances 5 \
    --concurrency 20 \
    --timeout 300 \
    --allow-unauthenticated

echo ""
echo "=== DEPLOYED ==="
gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)'
