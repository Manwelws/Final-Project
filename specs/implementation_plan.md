# Implementation Plan: Customer Churn Prediction & Automated Retraining System

## Goal Description

Deliver an end-to-end, production-grade Customer Churn Prediction & Automated Retraining System for **Project 1**:
- Complete data pipeline, candidate benchmarking, Bayesian hyperparameter optimization, and holdout evaluation with cost-utility threshold optimization.
- Global and individualized model interpretability using SHAP TreeExplainer with parent categorical feature grouping.
- Closed-loop retraining simulation adhering to **ADR 0001**, utilizing the production model artifact with dual drift monitoring (Unsupervised Population Stability Index [PSI] and Supervised performance decay) and exponential sample decay retraining.
- High-performance REST prediction service using **FastAPI** and **Pydantic v2**, exposing health, single prediction, batch scoring, and real-time SHAP explanation endpoints.
- Production containerization via multi-stage **Docker** and **Docker Compose**.
- Stakeholder **Streamlit** dashboard for executive KPI tracking, drift monitoring, and retention triage.
- Thorough automated testing suite and comprehensive documentation.

---

## Architectural & Design Decisions

| Dimension | Decision & Implementation | Status |
|---|---|:---:|
| **Candidate Models** | Tree-based ensembles (`RandomForestClassifier`, `LGBMClassifier`, `XGBClassifier`). LightGBM ranked #1 (ROC-AUC `0.8528 ± 0.0104`). Deep learning (PyTorch/TF) excluded per spec to avoid overfitting tabular data. | **Delivered** |
| **Class Imbalance** | Native algorithmic class weighting (`class_weight='balanced'`, `scale_pos_weight`). SMOTE excluded to avoid synthetic distortion in sparse one-hot spaces. | **Delivered** |
| **Hyperparameter Optimization** | Bayesian Optimization using **Optuna** (`TPESampler`, `MedianPruner`) across candidates, tracking parameters, metrics, and artifacts to **MLflow** (`./mlruns`). | **Delivered** |
| **Threshold Optimization** | Cost-Utility Matrix ($C_{\text{FN}}=\$200, C_{\text{FP}}=\$20$) sweep over $[0.01, 0.99]$. Optimal cutoff: **`0.12`**, reducing expected business loss by **59.5%** (**`$21,860.00`** net savings; Recall: `93.05%`). | **Delivered** |
| **Model Interpretability** | SHAP `TreeExplainer` with one-hot dummy variables aggregated into parent categories (`Contract`, `Internet Service`, `Payment Method`). Summary, beeswarm, dependence, and waterfall plots. | **Delivered** |
| **Retraining Simulation (ADR 0001)** | `notebooks/simulate_retraining.py` consumes production model artifact (`models/model.joblib`). Implemented **Unsupervised PSI** on weekly batches, **Supervised decay trigger** (rolling live AUC), and **Exponential Sample Decay** ($0.85^{\text{weeks\_old}}$). | **Delivered** |
| **Unified CLI** | Console script `churn` with subcommands: `train`, `benchmark`, `evaluate`, `explain`. | **Delivered** |
| **Serving API (Milestone 3)** | **FastAPI** with **Pydantic v2** validation. Endpoints: `GET /health`, `POST /predict`, `POST /predict/batch`, and `POST /explain` with real-time SHAP parent feature attributions. | **Active (Next)** |
| **Containerization (Milestone 3)** | Multi-stage `Dockerfile` (Python 3.14-slim, non-root user, health check probes) and `docker-compose.yml`. | **Active (Next)** |
| **Dashboard (Milestone 4)** | Multi-page **Streamlit** dashboard (Overview KPIs, Benchmark & Evaluation, Drift Timeline & PSI Heatmap, Live Prediction & Triage, MLflow Experiment Viewer). | **Planned** |

---

## System Architecture Overview

```mermaid
flowchart TB
    subgraph DataLayer["Data Layer"]
        RAW[("Raw Excel Data\n7,043 × 33")]
        CLEAN[("Cleaned Features & Target\nX_features.csv + y_target.csv")]
        LIVE[("Live Streaming Batches\ndata/live/features & labels")]
        DRIFT[("Drift Metrics & Artifacts\ndata/drift/ & data/retraining/")]
    end

    subgraph PipelineLayer["Pipeline & Training Layer (Milestone 2 - Complete)"]
        LOADER["Data Loader & Seams\nfinal_project.data"]
        BENCH["Candidate Benchmark\nRF, LightGBM, XGBoost"]
        OPTUNA["Bayesian Optimization\nOptuna + MedianPruner"]
        MLFLOW[("MLflow Tracking\n./mlruns")]
        EVAL["Evaluation & Cost Matrix\nOptimal cutoff: 0.12"]
        EXPLAIN["SHAP Explainability\nParent feature grouping"]
        REGISTRY[("Model Registry\nmodels/model.joblib + metadata.json")]
    end

    subgraph SimulationLayer["Retraining Simulation Layer (ADR 0001 - Complete)"]
        SIM["Retraining Simulator\nsimulate_retraining.py"]
        PSI_MON["Unsupervised Drift Monitor\nPopulation Stability Index"]
        DECAY_TRIG["Supervised Decay Trigger\nRolling 2-week live AUC"]
        EXP_WEIGHT["Exponential Sample Decay\n0.85^weeks_old"]
    end

    subgraph ServingLayer["Serving Layer (Milestone 3 - Active)"]
        API["FastAPI Prediction Service\n/health, /predict, /predict/batch, /explain"]
        DOCKER["Containerized Deployment\nDockerfile + docker-compose.yml"]
    end

    subgraph DashboardLayer["Dashboard & Ops Layer (Milestone 4 - Planned)"]
        DASH["Streamlit Interactive Dashboard\n5 multi-page views"]
    end

    RAW --> CLEAN
    CLEAN --> LOADER
    LOADER --> BENCH
    BENCH --> OPTUNA
    OPTUNA <--> MLFLOW
    OPTUNA --> EVAL
    EVAL --> EXPLAIN
    EVAL --> REGISTRY
    REGISTRY --> SIM
    CLEAN --> SIM
    SIM --> LIVE
    SIM --> PSI_MON
    SIM --> DECAY_TRIG
    DECAY_TRIG --> EXP_WEIGHT
    EXP_WEIGHT --> DRIFT
    REGISTRY --> API
    API --> DOCKER
    DRIFT --> DASH
    MLFLOW --> DASH
    API -.->|live scoring| DASH

    style DataLayer fill:#1e293b,stroke:#3b82f6,color:#f8fafc
    style PipelineLayer fill:#0f172a,stroke:#10b981,color:#f8fafc
    style SimulationLayer fill:#172554,stroke:#6366f1,color:#f8fafc
    style ServingLayer fill:#312e81,stroke:#8b5cf6,color:#f8fafc
    style DashboardLayer fill:#3b0764,stroke:#ec4899,color:#f8fafc
```

---

## Use Case Diagram

```mermaid
flowchart LR
    subgraph Actors
        DS["Data Scientist"]
        BA["Business Analyst / Executive"]
        RT["Retention Specialist"]
        APP["Client Application"]
        SIM_SYS["Simulation / Retraining System"]
    end

    subgraph UseCases["Churn Prediction & Retraining System"]
        UC1["Benchmark Candidate Models\nchurn benchmark"]
        UC2["Tune Hyperparameters\nchurn train --tune"]
        UC3["Evaluate on Holdout & Optimize Threshold\nchurn evaluate"]
        UC4["Inspect SHAP Explainability\nchurn explain"]
        UC5["Score Customer Churn Risk\nPOST /predict"]
        UC6["Batch Score Customers\nPOST /predict/batch"]
        UC7["Triage Risk Drivers (SHAP)\nPOST /explain"]
        UC8["Monitor Feature Drift (PSI)\nUnsupervised Monitor"]
        UC9["Detect Performance Decay\nSupervised Trigger"]
        UC10["Retrain with Exponential Sample Decay\nsimulate_retraining.py"]
        UC11["View Interactive Dashboard\nStreamlit UI"]
    end

    DS --> UC1
    DS --> UC2
    DS --> UC3
    DS --> UC4
    DS --> UC10
    BA --> UC3
    BA --> UC4
    BA --> UC11
    RT --> UC7
    RT --> UC11
    APP --> UC5
    APP --> UC6
    SIM_SYS --> UC8
    SIM_SYS --> UC9
    SIM_SYS --> UC10

    UC3 -.->|sets optimal cutoff| UC5
    UC4 -.->|powers real-time attributions| UC7
    UC9 -.->|triggers| UC10
```

---

## Data Flow Diagram

```mermaid
flowchart LR
    subgraph Sources["Data Storage"]
        XF["X_features.csv\n7,043 × 27 numeric"]
        YF["y_target.csv\nChurn Value binary"]
    end

    subgraph SplitModule["Data Ingestion & Splitting"]
        SPLIT["Stratified 80/20 Split\nfinal_project.data"]
        TRAIN_SET["Train Set (5,634 rows)"]
        TEST_SET["Holdout Test Set (1,409 rows)"]
    end

    subgraph Modeling["Model Training & Optimization"]
        CAND["Candidate Models\nRF, LightGBM, XGBoost"]
        OPT["Optuna Bayesian Search\n5-Fold Stratified CV"]
        BEST_M["Best Model Artifact\nLightGBM / RF (models/model.joblib)"]
    end

    subgraph EvaluationFlow["Evaluation & Explainability"]
        COST["Cost Matrix Sweep\nCFN=$200, CFP=$20"]
        CUTOFF["Optimal Cutoff: 0.12\n$21,860 Savings"]
        SHAP_MOD["TreeExplainer & Grouping\nContract, Internet, Payment"]
    end

    subgraph SimulationFlow["Retraining Simulator (ADR 0001)"]
        WEEKLY["Weekly Batches (Weeks 1-8)\nCovariate & Concept Drift"]
        PSI_CALC["PSI Calculation\nFlag if PSI > 0.25"]
        DECAY["Live AUC Decay Trigger\nDrop > 0.05"]
        EXP["Exponential Decay Retrain\n0.85^weeks_old"]
    end

    subgraph ServingFlow["REST API Serving"]
        FASTAPI["FastAPI Application"]
        PRED_OUT["Probability + Decision + Risk Level"]
        EXPLAIN_OUT["Parent Feature Attributions"]
    end

    XF & YF --> SPLIT
    SPLIT --> TRAIN_SET & TEST_SET
    TRAIN_SET --> CAND --> OPT --> BEST_M
    BEST_M & TEST_SET --> COST --> CUTOFF
    BEST_M & TEST_SET --> SHAP_MOD
    BEST_M & XF & YF --> WEEKLY
    WEEKLY --> PSI_CALC
    WEEKLY --> DECAY --> EXP
    BEST_M & CUTOFF --> FASTAPI
    SHAP_MOD --> FASTAPI
    FASTAPI --> PRED_OUT & EXPLAIN_OUT
```

---

## Detailed Milestone Status & Roadmap

```
Final Project/
├── data/
│   ├── processed/                      # Source of truth: X_features.csv, y_target.csv
│   ├── live/                           # Simulated streaming batches (features, labels, pending)
│   ├── reference/                      # Static pre-drift holdout baseline
│   ├── drift/drift_metrics.csv         # Simulation metrics (PSI, AUC, triggers)
│   └── retraining/week_N/              # Artifacts from simulation retraining triggers
├── models/
│   ├── model.joblib                    # Serialized production model artifact
│   └── metadata.json                   # Architecture, CV score, optimal threshold (0.12)
├── reports/
│   ├── evaluation_report.md            # Holdout evaluation markdown report
│   └── figures/                        # 5 evaluation curves + 6 SHAP visualizations
├── src/final_project/
│   ├── __init__.py                     # Public API exports
│   ├── data.py                         # Stratified data loading and splitting
│   ├── models.py                       # Candidate model factory & cross-validation
│   ├── tuning.py                       # Optuna Bayesian optimization & MLflow integration
│   ├── evaluation.py                   # Holdout evaluation, cost matrix optimization
│   ├── explainability.py               # SHAP TreeExplainer & parent feature grouping
│   ├── drift.py                        # Population Stability Index & exponential sample decay
│   ├── pipeline.py                     # High-level training orchestration
│   ├── cli.py                          # Unified churn CLI (train, benchmark, evaluate, explain)
│   ├── serving/                        # [MILESTONE 3]
│   │   ├── __init__.py
│   │   ├── api.py                      # FastAPI application
│   │   └── schemas.py                  # Pydantic v2 input/output schemas
│   └── dashboard/                      # [MILESTONE 4]
│       ├── __init__.py
│       └── app.py                      # Streamlit interactive application
├── tests/                              # Automated test suite (29 tests passing)
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_candidates.py
│   ├── test_tuning.py
│   ├── test_pipeline.py
│   ├── test_pipeline_tuning.py
│   ├── test_evaluation.py
│   ├── test_evaluation_cv.py
│   ├── test_explainability.py
│   ├── test_drift_monitoring.py
│   ├── test_simulator_integration.py
│   ├── test_cli.py
│   ├── test_cli_benchmark.py
│   ├── test_cli_explain.py
│   └── test_api.py                     # [MILESTONE 3]
├── notebooks/
│   ├── cleaning_for_retraining.ipynb   # Verified data cleaning notebook
│   └── simulate_retraining.py          # ADR 0001 production retraining simulation
├── Dockerfile                          # [MILESTONE 3] Production multi-stage Dockerfile
├── docker-compose.yml                  # [MILESTONE 3] Containerized service definition
└── pyproject.toml                      # uv-managed Python project definition
```

---

### Milestone 1: Data Collection & Preprocessing — `[COMPLETED]`

- **Status**: Complete & Verified.
- **Deliverables**:
  - `data/processed/X_features.csv` (7,043 × 27, all numeric, 0 NaNs).
  - `data/processed/y_target.csv` (Churn Value, 5,174 retained / 1,869 churned).
  - Cleaning logic documented inline in `notebooks/cleaning_for_retraining.ipynb`.

---

### Milestone 2: Production Churn Modeling Pipeline, Evaluation & Explainability — `[COMPLETED]`

- **Status**: 100% Implemented & Tested (PR #10 ready for review, 29/29 tests passing).
- **Tracer Bullets Delivered**:
  1. **#2 Baseline Pipeline Tracer Bullet**: Core seams (`final_project.data`, `models`, `pipeline`), stratified 80/20 splitting, baseline Random Forest with 5-fold Stratified CV, artifact serialization, and `churn train` CLI.
  2. **#3 Multi-Model Candidates & Class Weighting**: Unified candidate factory (`RandomForestClassifier`, `LGBMClassifier`, `XGBClassifier`) with native imbalance handling (`balanced`, `scale_pos_weight`). Exposed `churn benchmark` CLI (LightGBM ranked #1 with ROC-AUC `0.8528`).
  3. **#4 Bayesian Hyperparameter Optimization & Experiment Tracking**: Optuna Bayesian search with `MedianPruner`, MLflow tracking backend (`mlruns`), and `churn train --tune`.
  4. **#5 Holdout Evaluation & Cost-Utility Thresholding**: Evaluation on unseen 20% test split ($N=1,409$). Cost matrix ($C_{\text{FN}}=\$200, C_{\text{FP}}=\$20$) optimization identified optimal decision cutoff **`0.12`**, saving **`$21,860.00`** in net business loss (59.5% reduction; Recall: `93.05%`). Exported `reports/evaluation_report.md` and 5 diagnostic plots. Exposed `churn evaluate`.
  5. **#6 Model Interpretability with SHAP**: TreeExplainer exact Shapley values. One-hot dummy features aggregated into parent categories (`Contract`, `Internet Service`, `Payment Method`). Summary, beeswarm, dependence plots, and individual waterfall plots. Exposed `churn explain`.
  6. **#7 Retraining Simulator Integration (ADR 0001)**: Replaced toy logistic regression with serialized production model artifact. Added Unsupervised Data Drift Monitor calculating **Population Stability Index (PSI)** on incoming weekly batches, and Exponential Sample Decay ($0.85^{\text{weeks\_old}}$) for adaptive retraining.

---

### Milestone 3: Production Prediction Service & Containerized Deployment — `[ACTIVE / NEXT FRONTIER]`

- **Status**: Ready for implementation.
- **Objectives**:
  - Build a high-performance REST prediction service using **FastAPI** and **Pydantic v2**.
  - Expose operational endpoints:
    - `GET /health` — service status, model architecture, metadata version.
    - `POST /predict` — single-customer scoring returning churn probability, risk category, and binary decision based on the cost-optimal cutoff (`0.12`).
    - `POST /predict/batch` — bulk scoring for tabular datasets.
    - `POST /explain` — on-demand SHAP parent feature attributions for frontline retention triage.
  - Package into a production-grade multi-stage **Dockerfile** (non-root security, health checks) and **Docker Compose**.

#### API Request/Response Sequence

```mermaid
sequenceDiagram
    autonumber
    participant Client as Client / Frontline App
    participant API as FastAPI (src/final_project/serving/api.py)
    participant VAL as Pydantic v2 Validator
    participant REG as Model Artifact & Metadata
    participant SHAP_ENG as SHAP TreeExplainer Engine

    Note over Client,SHAP_ENG: 1. Health & Metadata Verification
    Client->>API: GET /health
    API->>REG: check model loaded & metadata version
    API-->>Client: 200 OK {"status": "healthy", "model": "LightGBM", "threshold": 0.12}

    Note over Client,SHAP_ENG: 2. Single Customer Prediction
    Client->>API: POST /predict {customer_features}
    API->>VAL: validate 27 numeric feature fields
    alt Validation Failure
        VAL-->>API: 422 Unprocessable Entity
        API-->>Client: Error details
    else Validation Success
        API->>REG: model.predict_proba(features)
        REG-->>API: proba = 0.28
        API->>API: apply optimal cutoff (0.28 >= 0.12 -> churn=1, risk="high")
        API-->>Client: 200 OK {churn_probability: 0.28, prediction: 1, risk_level: "high"}
    end

    Note over Client,SHAP_ENG: 3. Real-Time SHAP Triage Explanation
    Client->>API: POST /explain {customer_features}
    API->>SHAP_ENG: compute sample TreeExplainer values
    SHAP_ENG->>SHAP_ENG: group dummy variables into parent categories
    SHAP_ENG-->>API: parent attributions (Contract: +0.14, Tenure: -0.08)
    API-->>Client: 200 OK {top_risk_drivers: [...], top_protective_drivers: [...]}
```

#### Docker Deployment Topology

```mermaid
flowchart TB
    subgraph Host["Production Host / Cloud VM"]
        subgraph DockerNet["Isolated Bridge Network"]
            subgraph APIContainer["churn-api Container (python:3.14-slim)"]
                NONROOT["Non-root user (appuser)"]
                UVI["uvicorn ASGI Server (:8000)"]
                FAST["FastAPI App (final_project.serving.api)"]
                MODELS_CACHE["Baked/Mounted models/ (model.joblib, metadata.json)"]
                HEALTH_PROBE["HEALTHCHECK curl /health"]
                NONROOT --> UVI --> FAST --> MODELS_CACHE
                HEALTH_PROBE -.->|probes| UVI
            end
        end
    end

    CLIENT["Client App / Webhook"] -->|HTTP :8000| UVI
    AZURE["Azure Container Apps / App Service"] -.->|deploys image| APIContainer

    style Host fill:#1e293b,stroke:#475569,color:#f8fafc
    style DockerNet fill:#0f172a,stroke:#3b82f6,color:#f8fafc
    style APIContainer fill:#1e1b4b,stroke:#8b5cf6,color:#f8fafc
```

---

### Milestone 4: MLOps, Monitoring & Web Dashboard — `[PLANNED]`

- **Status**: Scheduled following Milestone 3.
- **Objectives**:
  - Multi-page interactive **Streamlit** dashboard (`src/final_project/dashboard/app.py`):
    - **Page 1: Executive KPI Overview** — dataset metrics, churn rates, financial impact of threshold optimization ($21,860 net savings).
    - **Page 2: Model Benchmark & Diagnostics** — ROC curves, PR curves, calibration diagrams, confusion matrices across candidate models.
    - **Page 3: Drift & Retraining Monitor** — live timeline of rolling AUC, Population Stability Index (PSI) heatmaps, retraining events.
    - **Page 4: Live Retention Scoring & Triage** — interactive form to input customer attributes, receive churn risk gauge, and view real-time SHAP waterfall charts.
    - **Page 5: MLflow Experiment Hub** — embedded trial parameters, loss curves, and artifact exploration.

#### Retraining Decision State Machine (ADR 0001 Verified)

```mermaid
stateDiagram-v2
    [*] --> UnlabeledWeekArrival: Week N features arrive
    UnlabeledWeekArrival --> UnsupervisedDriftCheck: Compute PSI vs Reference Baseline

    state UnsupervisedDriftCheck {
        [*] --> CheckPSI
        CheckPSI --> Stable: PSI < 0.10
        CheckPSI --> ModerateShift: 0.10 <= PSI < 0.25
        CheckPSI --> SignificantShift: PSI >= 0.25 (Drift Alert)
    }

    UnsupervisedDriftCheck --> LabelRelease: Wait 2 weeks (Week N+2)
    LabelRelease --> SupervisedDecayEvaluation: Labels released for Week N

    state SupervisedDecayEvaluation {
        [*] --> EvaluateRollingLiveAUC: Compute AUC on recent 2-week live window
        EvaluateRollingLiveAUC --> StablePerformance: live_auc >= best_auc - 0.05
        EvaluateRollingLiveAUC --> PerformanceDecayTriggered: live_auc < best_auc - 0.05
    }

    StablePerformance --> Monitoring: No action needed
    PerformanceDecayTriggered --> RetrainingPipeline: Fire Retraining Trigger

    state RetrainingPipeline {
        [*] --> ApplyExponentialSampleDecay: Weight historical samples by 0.85^weeks_old
        ApplyExponentialSampleDecay --> FitProductionModel: Clone and fit production ensemble
        FitProductionModel --> ValidateNewAUC: Evaluate on static reference & live window
        ValidateNewAUC --> SaveRetrainedArtifacts: Export data/retraining/week_N/
    }

    RetrainingPipeline --> Monitoring: Update best_recent_auc
    Monitoring --> UnlabeledWeekArrival: Next week arrives
```

---

### Milestone 5: Documentation & Presentation — `[PLANNED]`

- **Status**: Scheduled for final project wrap-up.
- **Deliverables**:
  - Comprehensive `README.md` with system diagrams, quick-start guide, and performance benchmarks.
  - Architecture Decision Records (`docs/adr/`).
  - Executive presentation slide deck summarizing methodology, financial optimization findings, and operational deployment.

---

## Milestone-to-Deliverable Traceability

```mermaid
flowchart LR
    subgraph M1["Milestone 1: Data & EDA"]
        D1A["Cleaned Dataset (7,043 × 27)"]
        D1B["Cleaning Notebook"]
    end

    subgraph M2["Milestone 2: Production Pipeline"]
        D2A["Candidate Models (RF, LightGBM, XGBoost)"]
        D2B["Bayesian HPO & MLflow Tracking"]
        D2C["Holdout Evaluation & Cost Optimization (0.12)"]
        D2D["SHAP Interpretability & Grouping"]
        D2E["Retraining Simulator & Dual Drift (ADR 0001)"]
        D2F["Unified CLI (churn)"]
    end

    subgraph M3["Milestone 3: Serving & Containers"]
        D3A["FastAPI REST Service (/predict, /explain)"]
        D3B["Multi-stage Docker & docker-compose"]
    end

    subgraph M4["Milestone 4: MLOps & Dashboard"]
        D4A["Streamlit Multi-Page Dashboard"]
        D4B["Live Drift & PSI Monitoring UI"]
    end

    subgraph M5["Milestone 5: Documentation"]
        D5A["Comprehensive README & ADRs"]
        D5B["Executive Presentation"]
    end

    D1A & D1B --> D2A
    D2A --> D2B --> D2C --> D2D --> D2E --> D2F
    D2F --> D3A --> D3B
    D2C & D2D & D2E --> D4A
    D3A -.->|live triage API| D4A
    M1 & M2 & M3 & M4 --> D5A & D5B

    style D1A fill:#059669,stroke:#047857,color:#fff
    style D1B fill:#059669,stroke:#047857,color:#fff
    style D2A fill:#059669,stroke:#047857,color:#fff
    style D2B fill:#059669,stroke:#047857,color:#fff
    style D2C fill:#059669,stroke:#047857,color:#fff
    style D2D fill:#059669,stroke:#047857,color:#fff
    style D2E fill:#059669,stroke:#047857,color:#fff
    style D2F fill:#059669,stroke:#047857,color:#fff
    style D3A fill:#2563eb,stroke:#1d4ed8,color:#fff
    style D3B fill:#2563eb,stroke:#1d4ed8,color:#fff
    style D4A fill:#d97706,stroke:#b45309,color:#fff
    style D4B fill:#d97706,stroke:#b45309,color:#fff
    style D5A fill:#64748b,stroke:#475569,color:#fff
    style D5B fill:#64748b,stroke:#475569,color:#fff
```

*(Legend: **Green** = Completed & Verified; **Blue** = Active Frontier [Milestone 3]; **Amber** = Planned [Milestone 4]; **Slate** = Documentation [Milestone 5])*

---

## Verification Plan

### Automated Test Suite Execution

```bash
# Run complete test suite (29 existing tests + new API tests)
uv run pytest -v

# Run candidate model benchmark across 5 stratified folds
uv run churn benchmark --data-dir data/processed --cv-folds 5

# Run Bayesian hyperparameter optimization study
uv run churn train --data-dir data/processed --output-dir models --tune --n-trials 20

# Run holdout evaluation and threshold optimization
uv run churn evaluate --data-dir data/processed --model-path models/model.joblib --report-dir reports

# Run SHAP explainability generation
uv run churn explain --data-dir data/processed --model-path models/model.joblib --output-dir reports/figures

# Run retraining simulator with dual drift monitoring (ADR 0001)
uv run python notebooks/simulate_retraining.py
```

### Milestone 3 Verification (API & Container)

```bash
# 1. Run FastAPI service locally
uv run uvicorn final_project.serving.api:app --host 0.0.0.0 --port 8000 --reload

# 2. Check service health & metadata
curl -X GET http://localhost:8000/health

# 3. Test single-customer prediction (verifying optimal 0.12 cutoff application)
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d @tests/fixtures/sample_customer.json

# 4. Test real-time SHAP triage attribution
curl -X POST http://localhost:8000/explain \
     -H "Content-Type: application/json" \
     -d @tests/fixtures/sample_customer.json

# 5. Build and run Docker container
docker build -t churn-prediction-api:latest .
docker run -d -p 8000:8000 --name churn-service churn-prediction-api:latest
curl http://localhost:8000/health
docker stop churn-service && docker rm churn-service
```
