#!/bin/bash
# ── Deploy Cascade Predict to Azure Container Apps ───────────────
# Run from LOCAL machine with Azure CLI installed.
#
# Prerequisites:
#   1. Install Azure CLI: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli
#   2. az login
#   3. az account set --subscription YOUR_SUBSCRIPTION_ID
# ─────────────────────────────────────────────────────────────────

set -e

RESOURCE_GROUP="cascade-predict-rg"
LOCATION="eastus"
ACR_NAME="cascadepredictacr"
APP_NAME="cascade-predict"
ENV_NAME="cascade-env"

echo "=== Creating Resource Group ==="
az group create --name ${RESOURCE_GROUP} --location ${LOCATION}

echo "=== Creating Container Registry ==="
az acr create --resource-group ${RESOURCE_GROUP} --name ${ACR_NAME} --sku Basic --admin-enabled true

echo "=== Building and Pushing Image ==="
az acr build --registry ${ACR_NAME} --image ${APP_NAME}:latest .

echo "=== Creating Container App Environment ==="
az containerapp env create \
    --name ${ENV_NAME} \
    --resource-group ${RESOURCE_GROUP} \
    --location ${LOCATION}

echo "=== Deploying Container App ==="
ACR_PASSWORD=$(az acr credential show --name ${ACR_NAME} --query "passwords[0].value" -o tsv)

az containerapp create \
    --name ${APP_NAME} \
    --resource-group ${RESOURCE_GROUP} \
    --environment ${ENV_NAME} \
    --image ${ACR_NAME}.azurecr.io/${APP_NAME}:latest \
    --registry-server ${ACR_NAME}.azurecr.io \
    --registry-username ${ACR_NAME} \
    --registry-password ${ACR_PASSWORD} \
    --target-port 8501 \
    --ingress external \
    --cpu 1 \
    --memory 2Gi \
    --min-replicas 1 \
    --max-replicas 5

echo ""
echo "=== DEPLOYED ==="
az containerapp show --name ${APP_NAME} --resource-group ${RESOURCE_GROUP} --query "properties.configuration.ingress.fqdn" -o tsv
