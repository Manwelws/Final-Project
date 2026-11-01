# Azure Cloud Deployment Guide: Churn Prediction Microservice

This guide walks through deploying the containerized Customer Churn Prediction Service to **Azure Container Apps (ACA)** using **Azure Container Registry (ACR)** and the **Azure CLI**.

---

## 1. Prerequisites

- Active Azure Subscription.
- Azure CLI installed and authenticated:
  ```bash
  az login
  az account set --subscription "<YOUR_SUBSCRIPTION_ID>"
  ```
- Variables setup:
  ```bash
  RESOURCE_GROUP="rg-churn-production"
  LOCATION="eastus"
  ACR_NAME="acrchurnprod$(date +%s)"
  CONTAINER_APP_NAME="churn-prediction-api"
  ACA_ENV="aca-env-churn-prod"
  IMAGE_TAG="v1"
  ```

---

## 2. Provision Azure Infrastructure

### Step 2.1: Create Resource Group
```bash
az group create --name $RESOURCE_GROUP --location $LOCATION
```

### Step 2.2: Create Azure Container Registry (ACR)
```bash
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Basic \
  --admin-enabled true
```

### Step 2.3: Build & Push Container Image directly with ACR Tasks
ACR Tasks builds the multi-stage `Dockerfile` in the cloud without requiring a local Docker daemon:
```bash
az acr build \
  --registry $ACR_NAME \
  --image churn-prediction-api:$IMAGE_TAG \
  --file Dockerfile .
```
> [!NOTE]
> Per ADR 0002 (Decision D4), the production model artifact (`models/model.joblib`) and metadata (`models/metadata.json`) are committed into source control and baked directly into the image. At container startup, `ModelBundle.load` computes the SHA-256 hash of the artifact and validates it against `metadata.json` before deserialization, ensuring zero-trust bytecode integrity.

---

## 3. Deploy to Azure Container Apps (ACA)

### Step 3.1: Create Container App Environment
```bash
az containerapp env create \
  --name $ACA_ENV \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION
```

### Step 3.2: Retrieve ACR Admin Credentials
```bash
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv)
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer -o tsv)
```

### Step 3.3: Deploy the Container App with Decoupled Health Probes
Configure separate liveness (`/health`) and readiness (`/ready`) HTTP probes:
```bash
az containerapp create \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment $ACA_ENV \
  --image "$ACR_LOGIN_SERVER/churn-prediction-api:$IMAGE_TAG" \
  --registry-server $ACR_LOGIN_SERVER \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --target-port 8000 \
  --ingress external \
  --cpu 0.5 \
  --memory 1.0Gi \
  --min-replicas 1 \
  --max-replicas 5
```

---

## 4. Verify Live Cloud Deployment

### Step 4.1: Retrieve the Public FQDN
```bash
APP_URL=$(az containerapp show \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query "properties.configuration.ingress.fqdn" -o tsv)

echo "Service live at: https://$APP_URL"
```

### Step 4.2: Health and Readiness Probes

#### Liveness Probe (`GET /health`):
Lightweight probe that verifies the server process is responsive (always returns 200 OK):
```bash
curl -s -X GET "https://$APP_URL/health" | jq .
```
Expected output:
```json
{
  "status": "healthy",
  "algorithm": "RandomForestClassifier",
  "model_algorithm": "RandomForestClassifier",
  "optimal_threshold": 0.36,
  "holdout_roc_auc": 0.8436,
  "holdout_pr_auc": 0.6442,
  "model_timestamp": "2026-11-01T20:57:46Z",
  "feature_count": 27
}
```

#### Readiness Probe (`GET /ready`):
Returns 200 OK when the model artifact has passed SHA-256 verification and is ready to score; returns 503 if unloaded:
```bash
curl -s -X GET "https://$APP_URL/ready" | jq .
```
Expected output:
```json
{
  "status": "ready",
  "algorithm": "RandomForestClassifier",
  "optimal_threshold": 0.36
}
```

### Step 4.3: Real-Time Scoring (`POST /predict`)
Scores a complete raw CRM customer profile against the cost-optimal decision threshold:
```bash
curl -s -X POST "https://$APP_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Female",
    "Senior Citizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "Tenure Months": 2.0,
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "Fiber optic",
    "Online Security": "No",
    "Online Backup": "No",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "Yes",
    "Streaming Movies": "Yes",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Monthly Charges": 95.0,
    "Total Charges": 190.0,
    "CLTV": 3200.0
  }' | jq .
```

### Step 4.4: Frontline Retention Triage (`POST /explain`)
Returns parent SHAP attributions using the cached explainer:
```bash
curl -s -X POST "https://$APP_URL/explain" \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Female",
    "Senior Citizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "Tenure Months": 2.0,
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "Fiber optic",
    "Online Security": "No",
    "Online Backup": "No",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "Yes",
    "Streaming Movies": "Yes",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Monthly Charges": 95.0,
    "Total Charges": 190.0,
    "CLTV": 3200.0
  }' | jq .
```

### Step 4.5: Interactive OpenAPI Documentation
Open in browser:
```
https://<YOUR-APP-URL>/docs
```

---

## 5. Local Docker Testing (Alternative)

To run locally with Docker Compose:
```bash
docker compose up --build -d
```
Test liveness:
```bash
curl -s http://localhost:8000/health
```
Stop container:
```bash
docker compose down
```
