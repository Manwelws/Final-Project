# Milestone 3 Specification: Production Prediction Service & Containerized Deployment

## Problem Statement

With Milestone 2 complete, the Customer Churn Retraining Pipeline possesses high-performance candidate models (LightGBM/RandomForest), Bayesian hyperparameter optimization (Optuna + MLflow), financial cost-utility threshold optimization ($C_{\text{FN}}=\$200, C_{\text{FP}}=\$20$, optimal decision cutoff `0.12`), SHAP explainability, and a closed-loop retraining simulator adhering to ADR 0001.

However, the system currently lacks a production-ready HTTP serving interface and deployment packaging. External consumer systems—such as frontline CRM applications, billing platforms, webhooks, and the upcoming Milestone 4 Streamlit operations dashboard—cannot interact with the model via standard network protocols. Calling Python code directly couples clients to internal package internals, exposes raw dummy-encoded feature details, and prevents cloud or containerized deployment.

The team requires a production-grade, containerized REST API service built with FastAPI and Pydantic v2 that:
1. Validates incoming customer attributes robustly with dual schema support (friendly raw business fields with on-the-fly encoding adapter, alongside fast preprocessed numeric arrays).
2. Scores customer churn risk using the serialized `ModelBundle` artifact, applying the calibrated cost-optimal decision threshold (`0.12`) to return risk probabilities, binary retention decisions, and operational risk tiers (`Low`, `Medium`, `High`).
3. Supports bulk/batch customer scoring for tabular datasets.
4. Delivers on-demand, real-time SHAP interpretability via a dedicated triage endpoint (`POST /explain`), breaking down individual churn drivers into signed positive risk drivers and negative protective factors.
5. Packages cleanly into a production-grade multi-stage Docker container with non-root security, healthcheck probes, and Docker Compose orchestration, ready for Azure Container Apps or local deployment.

---

## Solution Architecture & Design

### Architectural Structure (`final_project.serving`)

```
src/final_project/serving/
├── __init__.py
├── schemas.py       # Pydantic v2 validation models, Dual Input Schemas, Enums
├── adapter.py       # Raw customer profile to 27-feature numeric matrix transformer
└── api.py           # FastAPI app, lifespan state management, REST endpoints
```

1. **Lifespan State Management**:
   The FastAPI application loads `ModelBundle.load()` during application startup (`lifespan` context manager) and stores it in `app.state.bundle`. This ensures zero per-request disk I/O or model deserialization latency.
2. **Dual Input Schema & Adapter**:
   - `CustomerProfileRaw`: User-friendly raw fields matching business terminology (e.g., `Contract="Month-to-month"`, `InternetService="Fiber optic"`, `TenureMonths=12`, `MonthlyCharges=75.50`).
   - `CustomerFeaturesVector`: Strict 27 numeric feature dictionary/model matching the exact training columns for high-speed tabular ingestion.
   - `encode_raw_profile()`: Pure deterministic transformer converting raw business fields into the exact 27-column numeric vector required by `ModelBundle`.
3. **Threshold-Calibrated Decisions**:
   Predictions apply the loaded `metadata.optimal_threshold` (`0.12`). Operational risk tiers are assigned:
   - `High`: Probability $\ge 0.50$
   - `Medium`: $0.12 \le \text{Probability} < 0.50$
   - `Low`: $\text{Probability} < 0.12$
4. **Frontline Retention Triage (`POST /explain`)**:
   Invokes `explain_customer_instance()` on the incoming customer profile, computing exact TreeExplainer Shapley values and aggregating dummy variables into parent categories. Outputs signed `risk_drivers` ($+$) and `protective_factors` ($-$).
5. **Containerization Topology**:
   - Multi-stage `Dockerfile` (`python:3.14-slim`), installing dependencies via `uv`, running as non-root `appuser`.
   - Native `HEALTHCHECK` probing `GET /health` with `curl`.
   - `docker-compose.yml` exposing port `8000` with mounted `./models` volume for zero-downtime model artifact refreshes.

---

## Endpoints Specification

| Method | Endpoint | Description | Request Body | Response Status & Body |
|---|---|---|---|---|
| `GET` | `/health` | Service liveness, model architecture, metadata version, and optimal threshold | None | `200 OK`: `HealthResponse` |
| `POST` | `/predict` | Real-time single customer churn risk scoring | `CustomerPredictionRequest` (Raw profile or feature vector) | `200 OK`: `SinglePredictionResponse` |
| `POST` | `/predict/batch` | Bulk customer scoring for offline or tabular lists | `BatchPredictionRequest` (List of customers) | `200 OK`: `BatchPredictionResponse` |
| `POST` | `/explain` | Real-time single customer SHAP risk attribution | `CustomerExplainRequest` (Raw profile or feature vector) | `200 OK`: `CustomerExplainResponse` |

---

## User Stories

1. **As an API client**, I want a `GET /health` endpoint reporting service status, active model algorithm, and optimal threshold, so that orchestrators and load balancers can verify readiness.
2. **As a CRM developer**, I want to send raw human-readable customer attributes (e.g. `Contract: "Month-to-month"`) to `POST /predict`, so that our frontend does not have to implement one-hot dummy encoding.
3. **As a data engineer**, I want to send preprocessed 27-numeric feature vectors to `POST /predict`, so that high-throughput pipeline scoring avoids redundant encoding overhead.
4. **As a retention manager**, I want every prediction response to include the calibrated optimal threshold (`0.12`) alongside the probability, binary decision, and operational risk tier (`Low`, `Medium`, `High`), so that outreach workflows can prioritize urgent cases.
5. **As a batch processing service**, I want a `POST /predict/batch` endpoint that scores up to 1,000 customers in a single HTTP transaction, so that daily batch churn scoring is efficient.
6. **As a retention specialist**, I want a `POST /explain` endpoint that returns signed parent category drivers (e.g., `Contract: +0.14` pushing toward churn, `Tenure: -0.08` protecting against churn), so that I can tailor customer retention incentives on live calls.
7. **As a platform engineer**, I want malformed requests to return standard `422 Unprocessable Entity` responses with descriptive field validation messages.
8. **As a DevOps engineer**, I want a multi-stage `Dockerfile` running as a non-root user with an internal health check, so that containerized deployment complies with cloud security baselines.
9. **As a cloud architect**, I want a `docker-compose.yml` file and documented Azure Container Apps deployment guide, so that the service can be run locally or cut over to Azure with zero configuration friction.
10. **As a QA engineer**, I want comprehensive integration tests using FastAPI's `TestClient` covering all endpoints, boundary values, and error states, so that regressions are blocked in CI.

---

## Implementation Plan: Granular Tracer Bullets

1. **Ticket #12 (Issue 1)**: `feat(serving): Pydantic v2 dual schemas and raw-to-encoded feature adapter`
   - Define `CustomerProfileRaw`, `CustomerFeaturesVector`, `PredictionResponse`, `BatchPredictionResponse`, `CustomerExplainResponse`, and `HealthResponse` in `src/final_project/serving/schemas.py`.
   - Implement `encode_raw_profile()` in `src/final_project/serving/adapter.py` mapping raw categorical choices into the 27 Telco numeric columns.
2. **Ticket #13 (Issue 2)**: `feat(serving): FastAPI application with /health, /predict, and /predict/batch endpoints`
   - Implement `src/final_project/serving/api.py` with FastAPI app, lifespan `ModelBundle` loading, and scoring endpoints applying optimal threshold (`0.12`) and risk tiers.
3. **Ticket #14 (Issue 3)**: `feat(serving): real-time customer triage and explainability endpoint (/explain)`
   - Implement `POST /explain` endpoint connecting incoming customer profiles to `explain_customer_instance()`, returning top risk drivers and protective factors.
4. **Ticket #15 (Issue 4)**: `test(serving): automated test suite for FastAPI service, dual schemas, and error boundaries`
   - Implement `tests/test_api.py` with `httpx.AsyncClient` / `TestClient`, validating health, raw input prediction, vector input prediction, batch scoring, explainability, and 422 validations.
5. **Ticket #16 (Issue 5)**: `chore(docker): production multi-stage Dockerfile, docker-compose, and Azure deployment guide`
   - Create multi-stage `Dockerfile` (`python:3.14-slim`, `uv`, non-root user `appuser`, healthcheck).
   - Create `docker-compose.yml` mounting `./models`.
   - Create `docs/deployment/azure.md` with step-by-step Azure Container Apps deployment commands.

---

## Testing Decisions

### What Makes a Good Test
- Tests must exercise external HTTP contracts (status codes, response JSON structures, error semantics) using `fastapi.testclient.TestClient`.
- Verify that both raw profile requests and preprocessed feature vector requests produce valid, identical risk scores.
- Verify that missing or invalid categorical values trigger clear `422 Unprocessable Entity` responses.
- Verify that `POST /explain` outputs valid signed floats with parent feature attributions matching known categories.

### Modules Tested
- `src/final_project/serving/schemas.py`
- `src/final_project/serving/adapter.py`
- `src/final_project/serving/api.py`
- `Dockerfile` build & container health probe.

---

## As-Built Deltas

- **Health Probe Architecture**: Separated endpoint contracts into `/health` (shallow liveness check for container orchestrators returning 200 without model dependencies) and `/ready` (deep readiness check confirming the model bundle is deserialized in memory and returning model algorithm/cutoff).
- **Flexible Batch Scoring Ingestion**: Extended `/predict/batch` to natively accept both JSON arrays of items (`[{"tenure": 12, ...}]`) and object dictionaries (`{"items": [...]}` or `{"profiles": [...]}`).
- **Pydantic v2 Strictness**: All schemas use Pydantic v2 strict typing and support customer identifiers preserved end-to-end.

