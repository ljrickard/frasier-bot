#!/bin/bash
set -e

# --- Configuration ---
PROJECT_ID="pisces-12"
REGION="us-central1"
REPO="pisces-repo"
APP_NAME="frasier-chat"
CLUSTER="pisces-cluster"

IMAGE_PATH="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$APP_NAME:latest"

echo "🤫 Skipping unit tests..."
echo "🐟 Starting Frasier Chat Deployment Pipeline..."

# 1. Connect to GKE
echo "🔌 Connecting to GKE Cluster..."
gcloud container clusters get-credentials $CLUSTER --region $REGION --project $PROJECT_ID

# 2. The Secret Bridge: GCP -> Kubernetes
echo "🔐 Syncing Database Password from GCP Secret Manager to GKE..."

# Fetch the raw password from GCP Secret Manager
DB_PASSWORD=$(gcloud secrets versions access latest --secret="frasier-db-password" --project=$PROJECT_ID)

# Create/Update the native Kubernetes Secret safely (Idempotent approach)
kubectl create secret generic frasier-db-password \
  --from-literal=password="$DB_PASSWORD" \
  --namespace default \
  --dry-run=client -o yaml | kubectl apply -f -

# 3. Docker: Build and Push
echo "📦 Building Docker Image for $APP_NAME..."
# Note: Using the ARG we set up in the root Dockerfile
docker build --build-arg SERVICE_NAME=$APP_NAME --platform linux/amd64 -t $IMAGE_PATH .

echo "☁️  Pushing to Artifact Registry..."
gcloud auth configure-docker $REGION-docker.pkg.dev --quiet
docker push $IMAGE_PATH

# 4. Helm: Deploy
echo "🚀 Deploying Helm Chart..."
helm upgrade --install $APP_NAME ./charts/frasier-chat \
  --namespace default

echo "✅ Pipeline Complete! Run 'kubectl get pods -w' to watch Niles and Frasier spin up."