# ⚡ ChurnOps: Production Customer Churn Prediction & Automated Retraining System

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![Package Manager: uv](https://img.shields.io/badge/managed%20by-uv-purple.svg)](https://github.com/astral-sh/uv)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com)
[![Test Suite](https://img.shields.io/badge/tests-75%2F75%20passing-brightgreen.svg)](tests/)

An enterprise-grade, closed-loop machine learning engineering system designed to detect telecom customer churn, optimize financial intervention economics, explain predictions to frontline retention specialists in real time, and continuously retrain models under covariate and concept drift.

Developed as **Project 1** for the **DEPI Machine Learning Engineering Track**.

---

## 🎯 Executive Summary & Business Economics

In subscription-based telecommunications, customer acquisition costs 5–7× more than retention. Undetected churn causes substantial financial bleeding, while untargeted retention incentives waste capital on non-churners.

ChurnOps bridges advanced data science modeling and executive financial decisioning through a calibrated **Cost-Utility Matrix**:
- **Cost of Missed Churner ($C_{\text{FN}}$)**: **$200.00** (lost future customer lifetime value).
- **Cost of Retention Incentive ($C_{\text{FP}}$)**: **$20.00** (targeted discount or service upgrade).
- **Optimal Decision Cutoff**: **`0.12`** (vs standard default `0.50`).

### 💰 Business Impact on Holdout Portfolio (1,409 Customers)

| Metric | Standard Baseline (Cutoff 0.50) | ChurnOps Optimized (Cutoff 0.12) | Business Advantage |
|---|---|---|---|
| **Holdout Churn Recall** | 48.40% (181 / 374 captured) | **93.05%** (348 / 374 captured) | **+44.65% more churners saved** |
| **Undetected Churners (FN)** | 193 missed | **26 missed** | **86.5% reduction in missed churn** |
| **Financial Loss** | $36,740.00 | **$14,880.00** | **$21,860.00 Net Savings** |
| **Business Cost Reduction** | Baseline | **-59.5%** | **Maximized Net Portfolio ROI** |

---

## 📐 Architecture & Engineering Diagrams

The system architecture, data flows, execution sequences, state machines, and deployment topologies specified in the engineering plan are modeled below in native Mermaid format:

### Figure 1: High-Level System Architecture & 4-Stage Layered Component Topology

```mermaid
flowchart LR
    subgraph DataLayer["1. DATA LAYER"]
        RAW["Raw Telco CSV\n(7,043 rows × 33 cols)"]
        NOTEBOOK["Cleaning Pipeline\n(cleaning_for_retraining.ipynb)"]
        CLEAN["Cleaned Dataset\n(X_features: 7043×27, y_target)"]
        STREAM["Reference & Live Stream\n(Baseline Holdout & Weeks 1–8)"]
        RAW --> NOTEBOOK --> CLEAN --> STREAM
    end

    subgraph TrainLayer["2. TRAINING & EVALUATION LAYER"]
        FACTORY["Candidate Model Factory\n(RF, LightGBM, XGBoost)"]
        WEIGHTS["Imbalance Strategy\n(Native Class Weights / pos_wt)"]
        OPTUNA["Bayesian Optimization\n(Optuna TPE + MedianPruner)"]
        MLFLOW[("MLflow Tracking Hub\n./mlruns Store")]
        EVAL["Cost-Utility Evaluation\n(C_FN=$200, C_FP=$20 -> Cutoff 0.12)"]
        SHAP["SHAP Explainability\n(Parent Domain Aggregation)"]

        FACTORY --> WEIGHTS --> OPTUNA
        OPTUNA <--> MLFLOW
        OPTUNA --> EVAL --> SHAP
    end

    subgraph ServingLayer["3. SERVING & DEPLOYMENT LAYER"]
        REGISTRY[("Model Artifact Registry\nmodels/model.joblib\nmodels/metadata.json")]
        API["FastAPI REST Microservice\n(:8000 /health, /predict, /explain)"]
        DOCKER["Docker Containerization\n(python:3.14-slim, Non-Root appuser)"]

        REGISTRY --> API --> DOCKER
    end

    subgraph OpsLayer["4. MLOps & BI RETRAINING LOOP (ADR 0001)"]
        DASH["Streamlit Operations Dashboard\n(:8501 5 Interactive Pages)"]
        DRIFT["Unsupervised PSI Monitor\n(Weekly Feature Population Shift)"]
        DECAY["Supervised Decay Trigger\n(Rolling 2-Week Live AUC Drop > 0.05)"]
        SIM["Closed-Loop Retraining Engine\n(Exponential Decay Weight: 0.85^weeks)"]

        DRIFT --> SIM
        DECAY --> SIM
    end

    CLEAN --> FACTORY
    EVAL --> REGISTRY
    STREAM --> DRIFT
    STREAM --> DECAY
    REGISTRY --> DASH
    REGISTRY -.->|prior model| SIM
    SIM ==>|promotes retrained model artifact| REGISTRY
    API -.->|live frontline scoring| DASH

    style DataLayer fill:#0B1120,stroke:#3B82F6,color:#F9FAFB
    style TrainLayer fill:#064E3B,stroke:#10B981,color:#F9FAFB
    style ServingLayer fill:#4C1D95,stroke:#8B5CF6,color:#F9FAFB
    style OpsLayer fill:#1E1B4B,stroke:#EC4899,color:#F9FAFB
```

---

### Figure 2: System Actor & Use Case Model Across Roles

```mermaid
flowchart LR
    subgraph OperationalActors["Operational Actors"]
        CLIENT["Client App / Webhook"]
        FRONT["Frontline Retention Agent"]
        SCHED["Weekly Scheduler / CI"]
    end

    subgraph SystemBoundary["ChurnOps Prediction & Retraining System Boundary"]
        UC1["Predict Churn Risk (Single Customer)\n[POST /predict]"]
        UC2["Batch Score Customers\n[POST /predict/batch]"]
        UC3["Explain Prediction (SHAP Triage)\n[POST /explain]"]
        UC4["Explore Data & Diagnostics\n[EDA Notebook & Dashboard]"]
        UC5["Benchmark Candidate Models\n[churn benchmark]"]
        UC6["Tune Hyperparameters (Optuna + MLflow)\n[churn train --tune]"]
        UC7["Evaluate Model & Optimize Threshold\n[churn evaluate (cutoff 0.12)]"]
        UC8["Monitor Feature Drift (PSI)\n[Unsupervised Live Monitor]"]
        UC9["Detect Performance Decay & Retrain\n[ADR 0001 Retraining Simulator]"]
        UC10["View Executive KPIs & Financial ROI\n[Streamlit Operations Dashboard]"]
    end

    subgraph AnalyticalActors["Analytical & Engineering Actors"]
        DS["Data Scientist / ML Engineer"]
        EXEC["Business Analyst / Executive"]
    end

    CLIENT --> UC1
    CLIENT --> UC3
    SCHED --> UC2
    SCHED --> UC8
    SCHED --> UC9
    FRONT --> UC1
    FRONT --> UC3
    FRONT --> UC10

    DS --> UC4
    DS --> UC5
    DS --> UC6
    DS --> UC7
    DS --> UC9
    EXEC --> UC7
    EXEC --> UC10

    UC1 -.->|«extends»| UC3
    UC5 -.->|«includes»| UC6
    UC7 -.->|«includes»| UC10

    style OperationalActors fill:#0B1120,stroke:#3B82F6,color:#F9FAFB
    style SystemBoundary fill:#111827,stroke:#6B7280,color:#F9FAFB
    style AnalyticalActors fill:#064E3B,stroke:#10B981,color:#F9FAFB
```

---

### Figure 3: End-to-End Data Processing, Splitting & Benchmarking Pipeline

```mermaid
flowchart LR
    subgraph Stage1["1. Source"]
        RAW["Telco Customer Churn\nRaw Dataset\n(7,043 × 33)"]
    end

    subgraph Stage2["2. Processing"]
        CLEAN_STEP["Cleaning & Imputation\nDrop Leakage IDs\nOne-Hot Encode Categoricals"]
        FEAT["X_features.csv\n(7,043 × 27 numeric, 0 NaNs)"]
        TARG["y_target.csv\n(5,174 retain / 1,869 churn)"]
        RAW --> CLEAN_STEP --> FEAT & TARG
    end

    subgraph Stage3["3. Split & Weight"]
        SPLIT["Stratified 80/20 Split\n(5,634 Train / 1,409 Test)"]
        PROD_WEIGHT["[PRODUCTION]\nNative Algorithmic Class Weights\n(balanced / scale_pos_weight)"]
        STUDY_SMOTE["[STUDY ONLY - EXCLUDED]\nSMOTE Oversampling\n(Distorts sparse one-hot space)"]
        FEAT & TARG --> SPLIT
        SPLIT ==> PROD_WEIGHT
        SPLIT -.-> STUDY_SMOTE
    end

    subgraph Stage4["4. Model Suite"]
        LGBM["#1 LightGBM Classifier\n(ROC-AUC: 0.8528 ± 0.0104)"]
        XGB["#2 XGBoost Classifier\n(ROC-AUC: 0.8492 ± 0.0118)"]
        RF["#3 Random Forest Classifier\n(ROC-AUC: 0.8474 ± 0.0107)"]
        BASELINES["Linear / SVM / Dummy Baselines\n(ROC-AUC: 0.76–0.81)"]
        PROD_WEIGHT --> LGBM & XGB & RF
        STUDY_SMOTE -.-> BASELINES
    end

    subgraph Stage5["5. Outputs"]
        ARTIFACT["Production Model Artifact\nmodels/model.joblib\nmodels/metadata.json (Cutoff: 0.12)"]
        EVAL_REPORTS["Holdout Evaluation & SHAP Reports\n(5 Diagnostic Curves & 6 SHAP Plots)"]
        MLFLOW_RUNS[("MLflow Tracking Hub\n./mlruns Store")]
        LGBM & XGB & RF --> ARTIFACT & EVAL_REPORTS & MLFLOW_RUNS
    end

    style Stage1 fill:#0B1120,stroke:#3B82F6,color:#F9FAFB
    style Stage2 fill:#111827,stroke:#6B7280,color:#F9FAFB
    style Stage3 fill:#064E3B,stroke:#10B981,color:#F9FAFB
    style Stage4 fill:#1E1B4B,stroke:#6366F1,color:#F9FAFB
    style Stage5 fill:#4C1D95,stroke:#8B5CF6,color:#F9FAFB
```

---

### Figure 4: Exploratory Data Analysis (EDA) 8-Stage Execution Flow & Validation Pipeline

```mermaid
flowchart LR
    subgraph Row1["Row 1: Exploration, Profiling & Collinearity Audit"]
        S1["1. Load Dataset\n7,043 × 33 raw\nInspect dtypes & nulls"]
        S2["2. Target Balance\n5,174 Retained (73.5%)\n1,869 Churned (26.5%)\n2.77:1 Imbalance Ratio"]
        S3["3. Feature Distributions\n27 numeric features\nHistograms & KDEs\nStratified by churn"]
        S4["4. Correlations\nFull matrix heatmap\nTop-10 correlation pairs\nCollinearity audit"]
        S1 --> S2 --> S3 --> S4
    end

    subgraph Row2["Row 2: Cohorts, Imbalance Strategy & Artifact Export"]
        S5["5. Bivariate Cohorts\nChurn by Contract\nTenure & Internet tier\nCohort retention curves"]
        S6["6. Imbalance Strategy Audit\nClass Weighting vs SMOTE\nIdentified SMOTE distortion\nin sparse dummy space"]
        S7["7. Statistical Tests\nPoint-Biserial (continuous)\nChi-Square (categorical)\nFeature ranking p < 0.001"]
        S8["8. Export Artifacts\nX_features.csv (7,043 × 27)\ny_target.csv (0 NaNs)\nSaved to data/processed/"]
        S5 --> S6 --> S7 --> S8
    end

    S4 ==>|Serpentine Flow| S5

    style Row1 fill:#0B1120,stroke:#3B82F6,color:#F9FAFB
    style Row2 fill:#064E3B,stroke:#10B981,color:#F9FAFB
```

---

### Figure 5: Dual-Strategy Benchmark, Optuna Bayesian Tuning & MLflow Logging Sequence

```mermaid
sequenceDiagram
    autonumber
    actor DS as Data Scientist / CLI
    participant DL as DataLoader (final_project.data)
    participant TP as TrainPipeline (final_project.pipeline)
    participant TUNER as Optuna Tuner (final_project.tuning)
    participant MLF as MLflow Store (./mlruns)
    participant EV as Evaluator (final_project.evaluation)
    participant REG as ModelRegistry (final_project.registry)

    DS->>DL: 1. load_features_and_target("data/processed")
    DL-->>DS: X_features (7,043 × 27), y_target
    DS->>DL: 2. get_stratified_split(X, y, test_size=0.20)
    DL-->>DS: X_train (5,634), X_test (1,409), y_train, y_test

    rect rgb(30, 41, 59)
        Note over DS,MLF: Strategy A: SMOTE Oversampling Benchmark (Evaluated & Excluded)
        DS->>TP: 3. benchmark_smote_baseline(X_train, y_train)
        TP->>MLF: 4. log_metrics(smote_runs, cv_auc=0.835)
        Note over TP,MLF: Distorts categorical binary distributions - discarded for production
    end

    rect rgb(6, 78, 59)
        Note over DS,REG: Strategy B: Native Class Weights + Optuna Bayesian Optimization [PRODUCTION]
        DS->>TUNER: 5. run_optuna_study(n_trials=50, candidates=[RF, LGBM, XGB])
        loop 50 Bayesian Trials
            TUNER->>TUNER: sample hyperparameters (TPESampler)
            TUNER->>MLF: log_params(trial_params), log_metrics(fold_auc)
            opt Intermediate validation below threshold
                TUNER->>TUNER: prune trial early (MedianPruner)
                TUNER->>MLF: set_tag("status", "PRUNED")
            end
        end
        TUNER-->>DS: Best trial: LightGBM (CV ROC-AUC: 0.8528 ± 0.0104)
        DS->>EV: 6. evaluate_model(best_model, X_test, y_test, cost_fn=200, cost_fp=20)
        EV-->>DS: Optimal Cutoff = 0.12 (Loss: $14,880 vs $36,740 - Net Savings: $21,860)
        DS->>REG: 7. ModelBundle.save("models", model, metadata)
        DS->>EV: 8. generate_evaluation_report() & generate_shap_plots()
        EV-->>DS: Markdown report & diagnostic figures saved to reports/
    end
```

---

### Figure 6: Data, Training, Evaluation, Explainability & Model Registry Class Architecture (UML)

```mermaid
classDiagram
    class DataLoader {
        +load_features_and_target(data_dir: Path) tuple[DataFrame, Series]
        +get_stratified_split(X: DataFrame, y: Series, test_size: float, seed: int) tuple
        +load_and_split_data(features_path, target_path, test_size, seed) DataSplit
    }

    class CandidateFactory {
        +build_candidate_models(balanced: bool, seed: int) dict[str, BaseEstimator]
        +benchmark_candidates(X_train: DataFrame, y_train: Series, cv_folds: int) DataFrame
    }

    class TuningAndPipeline {
        +run_optuna_study(X_train, y_train, n_trials: int, cv_folds: int) TuningResult
        +run_training(data_dir, output_dir, algorithm, balanced, tune, n_trials) TrainingSummary
    }

    class ModelMetadata {
        +timestamp: str
        +algorithm: str
        +balanced: bool
        +optimal_threshold: float
        +cost_savings: float
        +holdout_roc_auc: float
        +holdout_pr_auc: float
        +features: list[str]
        +load(path: Path) ModelMetadata
        +save(path: Path) None
    }

    class ModelBundle {
        +model: BaseEstimator
        +metadata: ModelMetadata
        +load(model_dir: Path) ModelBundle
        +save(model_dir: Path) None
        +predict_risk(X: DataFrame) RiskPrediction
        +predict_proba(X: DataFrame) ndarray
        +predict(X: DataFrame) ndarray
    }

    class CostMatrix {
        +cost_fn: float = 200.0
        +cost_fp: float = 20.0
        +compute_cost(tp, fp, tn, fn) float
    }

    class Evaluator {
        +compute_metrics_at_threshold(y_true, y_probs, threshold, cost_fn, cost_fp) ThresholdMetrics
        +find_optimal_threshold(y_true, y_probs, cost_matrix) tuple[float, DataFrame]
        +evaluate_model(model, X_test, y_test, cost_matrix) EvaluationSummary
        +run_evaluation(data_dir, model_path, metadata_path, report_dir) EvaluationSummary
    }

    class SHAPExplainer {
        +explain_customer_instance(model, customer_features) CustomerExplanation
        +aggregate_shap_by_parent(shap_values, feature_names) DataFrame
        +run_explanation(data_dir, model_path, output_dir, max_samples) ExplanationResult
    }

    class DriftMonitor {
        +calculate_psi(reference: ndarray, current: ndarray, n_bins: int) float
        +compute_dataset_psi(ref_df: DataFrame, curr_df: DataFrame) dict
        +compute_sample_weights(n_weeks: int, decay_rate: float) ndarray
        +StreamingBatchEngine
    }

    class ServingService {
        +app: FastAPI
        +predict_single(request: CustomerPredictionRequest) SinglePredictionResponse
        +predict_batch(requests: list) BatchPredictionResponse
        +explain_single(request: CustomerPredictionRequest) CustomerExplanationResponse
        +health_check() HealthResponse
    }

    class DashboardRouter {
        +main()
        +render_kpi_overview_page()
        +render_model_diagnostics_page()
        +render_drift_monitor_page()
        +render_retention_triage_page()
        +render_experiment_hub_page()
    }

    DataLoader --> CandidateFactory : provides training splits
    CandidateFactory --> TuningAndPipeline : candidate estimators
    TuningAndPipeline --> ModelBundle : packages best model
    ModelBundle *-- ModelMetadata : contains typed metadata
    ModelBundle --> Evaluator : scored on holdout
    CostMatrix --> Evaluator : optimizes cutoff
    ModelBundle --> SHAPExplainer : explains predictions
    ModelBundle --> DriftMonitor : monitors live drift
    ModelBundle --> ServingService : serves REST requests
    ModelBundle --> DashboardRouter : visualizes system state
```

---

### Figure 7: Closed-Loop Retraining Decision State Machine (ADR 0001 Verified)

```mermaid
stateDiagram-v2
    [*] --> UnlabeledBatchArrival: Week N Features Stream Arrives

    state UnlabeledBatchArrival {
        [*] --> IngestWeeklyFeatures
        IngestWeeklyFeatures --> ComputeWeeklyPSI: Against Static Reference (data/reference/)
    }

    UnlabeledBatchArrival --> UnsupervisedDriftGate

    state UnsupervisedDriftGate {
        [*] --> EvaluatePSI
        EvaluatePSI --> PSI_Stable: PSI < 0.10 (Routine)
        EvaluatePSI --> PSI_Moderate: 0.10 <= PSI < 0.25 (Warning)
        EvaluatePSI --> PSI_Severe: PSI >= 0.25 (Drift Alert Flagged)
    }

    UnsupervisedDriftGate --> PendingLabelQueue: Wait 2-Week Operational Label Lag

    state PendingLabelQueue {
        [*] --> HoldInPending: data/live/pending/
        HoldInPending --> ReleaseLabels: Week N+2 Arrives (Churn Labels Confirmed)
    }

    PendingLabelQueue --> SupervisedPerformanceGate: Evaluate Live Rolling AUC

    state SupervisedPerformanceGate {
        [*] --> ComputeLiveAUC: Rolling 2-Week Window
        ComputeLiveAUC --> PerformanceHealthy: live_auc >= best_auc - 0.05
        ComputeLiveAUC --> PerformanceDecay: live_auc < best_auc - 0.05 (Retrain Triggered!)
    }

    PerformanceHealthy --> LogMetricsToDriftCSV: Log to data/drift/drift_metrics.csv
    PerformanceDecay --> RetrainingEngine: Initiate Automated Retraining Execution

    state RetrainingEngine {
        [*] --> ApplyExponentialWeighting: w = 0.85^(weeks_old)
        ApplyExponentialWeighting --> FitCandidatePipeline: Fit Production Model with Sample Weights
        FitCandidatePipeline --> ValidateOnReferenceAndLive: Verify Reference AUC >= 0.84 & Live AUC Recovery
        ValidateOnReferenceAndLive --> ExportRetrainedArtifacts: Export to models/retrained/week_N/
    }

    RetrainingEngine --> PromoteActiveModel: Overwrite models/model.joblib & metadata.json
    PromoteActiveModel --> LogMetricsToDriftCSV: Update best_recent_auc benchmark
    LogMetricsToDriftCSV --> [*]: Ready for Week N+1 Batch
```

---

### Figure 8: FastAPI Request Validation, Prediction, Real-Time SHAP Attribution & Health Check Sequence

```mermaid
sequenceDiagram
    autonumber
    participant Client as Client / Frontline App / Browser
    participant API as FastAPI Router (serving/api.py)
    participant VAL as Pydantic v2 Validator (serving/schemas.py)
    participant ADP as Adapter (serving/adapter.py)
    participant BND as ModelBundle (models/model.joblib)
    participant SHAP as TreeExplainer (explainability.py)

    Note over Client,SHAP: 1. Health & Readiness Verification
    Client->>API: GET /health
    API->>BND: verify model loaded & inspect optimal threshold
    API-->>Client: 200 OK {"status": "healthy", "algorithm": "RandomForestClassifier", "optimal_threshold": 0.12}

    Note over Client,SHAP: 2. Single Customer Scoring & Decisioning
    Client->>API: POST /predict {"Contract": "Month-to-month", "Monthly Charges": 89.5, ...}
    API->>VAL: validate input schema (CustomerProfileRaw / Vector)
    alt Validation Error
        VAL-->>API: 422 Unprocessable Entity (Missing/Invalid fields)
        API-->>Client: 422 Response with validation error breakdown
    else Schema Valid
        API->>ADP: encode_raw_profile(profile) -> 1x27 numeric DataFrame
        ADP-->>API: encoded_df
        API->>BND: bundle.predict_risk(encoded_df)
        BND-->>API: RiskPrediction(prob=0.83, pred=1, tier="High", threshold=0.12)
        API-->>Client: 200 OK {"churn_probability": 0.83, "churn_prediction": 1, "risk_tier": "High", "threshold_applied": 0.12}
    end

    Note over Client,SHAP: 3. Real-Time Frontline Retention SHAP Triage
    Client->>API: POST /explain {"profile": {...}}
    API->>ADP: encode_raw_profile(profile)
    ADP-->>API: encoded_df
    API->>SHAP: explain_customer_instance(bundle.model, encoded_df)
    SHAP->>SHAP: compute TreeExplainer SHAP values & group one-hot features to parent domains
    SHAP-->>API: CustomerExplanation(risk_drivers, protective_factors, base_value, prob)
    API-->>Client: 200 OK {"risk_drivers": [["Tenure Months", +0.114], ["Contract", +0.112]], "protective_factors": [["Multiple Lines", -0.031]]}

    Note over Client,SHAP: 4. High-Throughput Batch Scoring
    Client->>API: POST /predict/batch [{"profile": {...}}, {"profile": {...}}]
    API->>ADP: vectorized batch encoding
    API->>BND: bundle.predict_risk(batch_df)
    API-->>Client: 200 OK {"predictions": [...], "count": N}
```

---

### Figure 9: Multi-Stage Docker Container & Volume Deployment Topology (Milestone 3 & 4)

```mermaid
flowchart TB
    subgraph Clients["External Clients & Consumers"]
        REST_CLIENT["REST API Consumer\n(Postman / Webhooks / cURL)\nhttp://localhost:8000"]
        BROWSER_USER["Frontline Retention Agents & Executives\n(Web Browser)\nhttp://localhost:8501"]
    end

    subgraph DockerHost["Production Docker Host / Cloud VM Environment"]
        subgraph DockerCompose["docker-compose.yml Runtime Network (bridge)"]
            subgraph ChurnService["Container 1: churn-service (:8000)"]
                APPUSER_API["User: appuser (non-root UID 1000)"]
                FASTAPI_APP["FastAPI Application\n(final_project.serving.api)"]
                UVICORN["Uvicorn ASGI Server (:8000)"]
                HEALTH["HEALTHCHECK Probe\n(curl -f http://localhost:8000/health)"]
                APPUSER_API --> UVICORN --> FASTAPI_APP
                HEALTH -.->|probes every 30s| UVICORN
            end

            subgraph DashboardService["Container 2: dashboard (:8501)"]
                APPUSER_DASH["User: appuser (non-root UID 1000)"]
                STREAMLIT["Streamlit Operations Dashboard\n(5 Interactive Pages)"]
                APPUSER_DASH --> STREAMLIT
            end
        end

        subgraph PersistentVolumes["Host Mounted Storage Volumes"]
            VOL_MODELS[("./models/\n(Read-Only Artifact Mount)\nmodel.joblib, metadata.json")]
            VOL_DATA[("./data/ & ./reports/\n(Processed Data, Baseline, Live Batches, Plots)")]
            VOL_MLRUNS[("./mlruns/\n(MLflow Experiment & Trial Tracking Store)")]
        end
    end

    REST_CLIENT -->|HTTP :8000| UVICORN
    BROWSER_USER -->|HTTP :8501| STREAMLIT

    STREAMLIT -.->|internal REST triage requests| UVICORN
    VOL_MODELS ==>|ro mount| ChurnService & DashboardService
    VOL_DATA ==>|rw mount| DashboardService
    VOL_MLRUNS ==>|rw mount| DashboardService

    style Clients fill:#0B1120,stroke:#3B82F6,color:#F9FAFB
    style DockerHost fill:#111827,stroke:#4B5563,color:#F9FAFB
    style DockerCompose fill:#1E1B4B,stroke:#8B5CF6,color:#F9FAFB
    style ChurnService fill:#064E3B,stroke:#10B981,color:#F9FAFB
    style DashboardService fill:#4A044E,stroke:#EC4899,color:#F9FAFB
    style PersistentVolumes fill:#1F2937,stroke:#F59E0B,color:#F9FAFB
```

---

## 🏆 Model Benchmark & Evaluation Diagnostics

All candidate tree ensembles were benchmarked across **5-Fold Stratified Cross-Validation** on 5,634 training samples:

| Rank | Model Architecture | Weighting Strategy | CV ROC-AUC (mean ± std) | CV PR-AUC (mean ± std) | Selection Status |
|:---:|---|---|:---:|:---:|:---:|
| 🥇 | **LightGBM** | Native Unweighted | **0.8528 ± 0.0104** | **0.6385 ± 0.0210** | **Best Candidate** |
| 🥈 | **XGBoost** | Native Unweighted | **0.8492 ± 0.0118** | **0.6310 ± 0.0225** | Candidate |
| 🥉 | **Random Forest** | Native Unweighted | **0.8474 ± 0.0107** | **0.6276 ± 0.0205** | Production Default |
| 4 | LightGBM | Balanced (`scale_pos_weight`) | 0.8415 ± 0.0121 | 0.6220 ± 0.0214 | Evaluated |
| 5 | XGBoost | Balanced (`scale_pos_weight`) | 0.8380 ± 0.0130 | 0.6190 ± 0.0230 | Evaluated |
| 6 | Random Forest | Balanced (`class_weight`) | 0.8350 ± 0.0115 | 0.6150 ± 0.0210 | Evaluated |

### 🔍 Interpretability & Frontline Explainability

ChurnOps decomposes 27 one-hot preprocessed features back into intuitive parent business domains using SHAP `TreeExplainer`:
- **Top Churn Risk Drivers (+)**: Month-to-month contracts, Fiber optic internet without security add-ons, Electronic check payment, Short tenure (<6 months).
- **Top Protective Factors (-)**: Long-term multi-year contracts, DSL / bundled Tech Support & Online Backup, Automatic credit card / bank transfer billing.

---

## 🔄 Streaming Retraining Pipeline (ADR 0001)

Simulates production streaming conditions over an 8-week horizon with realistic operational constraints:
- **2-Week Label Lag**: Customer churn status is confirmed only after 14 days of inactivity.
- **Covariate & Concept Drift**: Injected at Week 5 (Monthly Charges ×1.25, Tenure ×0.7, 30% label flip on high-charge retained customers).
- **Dual Triggering Architecture**:
  1. *Unsupervised PSI Monitor*: Detects feature distribution shifts (`Monthly Charges` PSI jumps from <0.04 to 4.35 at Week 5).
  2. *Supervised Decay Trigger*: Fires when rolling 2-week live AUC drops $\ge 0.05$ below the historical peak.
- **Exponential Sample Decay**: Applies recency weight $w = 0.85^{\text{weeks\_old}}$ during retraining, preventing stale data from diluting recent drift patterns.

---

## ⚡ Quickstart Guide

### 1. Prerequisites & Installation

Ensure Python 3.14+ and `uv` are installed on your machine:

```bash
# Clone repository
git clone https://github.com/Manwelws/Final-Project.git
cd "Final Project"

# Install all dependencies with uv
uv sync
```

### 2. Launch the Streamlit Operations Dashboard

The dashboard provides a unified UI spanning all 5 system pages:

```bash
# Launch via CLI subcommand
uv run churn dashboard --port 8501

# Or run directly via streamlit
uv run streamlit run src/final_project/dashboard/app.py
```
Open **`http://localhost:8501`** in your browser.

### 3. Launch the FastAPI Prediction Service

```bash
# Start FastAPI REST API via CLI subcommand
uv run churn serve --port 8000

# Or run directly via Uvicorn with auto-reload
uv run uvicorn final_project.serving.api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation is available at **`http://localhost:8000/docs`**.

### 4. Run via Docker Compose

```bash
# Build and run containerized service
docker-compose up --build
```
The REST API will be accessible on port `8000` with automated container healthcheck probes.

### 5. Run Retraining Simulation

```bash
# Run 8-week streaming retraining pipeline
python notebooks/simulate_retraining.py
```

### 6. Run Automated Test Suite

```bash
# Run all 75 automated tests
uv run pytest
```

---

## 💻 CLI Command Reference

The `churn` console CLI provides end-to-end management of the pipeline:

```bash
# 1. Benchmark candidate models with 5-fold cross-validation
uv run churn benchmark --cv-folds 5

# 2. Train baseline model or run Bayesian optimization
uv run churn train --algorithm random_forest
uv run churn train --tune --n-trials 50 --cv-folds 5

# 3. Evaluate model on holdout set and optimize cost cutoff
uv run churn evaluate --cost-fn 200 --cost-fp 20 --report-dir reports

# 4. Generate SHAP explainability plots and importance rankings
uv run churn explain --max-samples 150 --output-dir reports/figures

# 5. Run weekly streaming retraining & drift simulation
uv run churn simulate --n-weeks 8

# 6. Launch high-performance FastAPI inference microservice
uv run churn serve --port 8000 --host 0.0.0.0

# 7. Launch interactive Streamlit Operations Dashboard
uv run churn dashboard --port 8501 --host 0.0.0.0
```

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description | Sample Payload |
|---|---|---|---|
| `GET` | `/health` | Service health status, model algorithm, and active cutoff | None |
| `POST` | `/predict` | Single customer prediction with risk tier | `{"Contract": "Month-to-month", "Monthly Charges": 85.0, ...}` |
| `POST` | `/predict/batch` | High-throughput batch scoring | `[{"Contract": "One year", ...}, ...]` |
| `POST` | `/explain` | Signed parent SHAP decomposition | `{"profile": {"Contract": "Month-to-month", ...}}` |

---

## 📁 Repository Structure

```
├── data/
│   ├── processed/                # Authoritative source of truth
│   │   ├── X_features.csv        # 7,043 × 27 preprocessed numeric features
│   │   ├── y_target.csv          # Churn Value binary labels (0/1)
│   │   └── Telco_customer_churn_cleaned.csv
│   ├── raw/                      # Original source Excel file
│   ├── reference/                # Reference dataset for PSI monitoring
│   ├── live/                     # Streaming weekly batches (weeks 1–8)
│   ├── drift/                    # Output drift metrics CSV
│   └── retraining/               # Retraining artifacts per week
├── models/
│   ├── model.joblib              # Serialized production model artifact
│   └── metadata.json             # Typed training & evaluation metadata
├── notebooks/
│   ├── cleaning_for_retraining.ipynb
│   └── simulate_retraining.py    # Retraining simulation runner
├── reports/                      # Holdout evaluation markdown & SHAP plots
├── specs/                        # Engineering specifications & ADRs
│   ├── implementation_plan.md    # Full project roadmap & diagrams
│   ├── milestone_2_spec.md
│   ├── milestone_3_spec.md
│   └── milestone_4_spec.md
├── src/final_project/
│   ├── cli.py                    # Unified console CLI (`churn`)
│   ├── data.py                   # Data loaders & stratified splitting
│   ├── domain/                   # CostMatrix & business domain models
│   ├── drift.py                  # PSI & streaming drift algorithms
│   ├── evaluation.py             # Evaluation & threshold optimization
│   ├── explainability.py         # SHAP TreeExplainer & parent grouping
│   ├── models.py                 # Tree ensemble factories & benchmark
│   ├── pipeline.py               # Training pipeline orchestration
│   ├── registry.py               # ModelBundle & metadata schemas
│   ├── simulation.py             # Streaming engine implementation
│   ├── tuning.py                 # Optuna & MLflow Bayesian tuning
│   ├── serving/                  # FastAPI REST service & adapters
│   └── dashboard/                # Streamlit Operations Dashboard
│       ├── app.py                # Main router & design shell
│       ├── styles.py             # Dark modern design system tokens
│       ├── state.py              # Cached loaders & KPI calculations
│       └── pages/                # 5 dedicated dashboard views
├── tests/                        # 75 automated pytest test cases across 22 modules
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## 📄 License & Academic Integrity

Developed for educational and technical portfolio assessment within the **DEPI Machine Learning Engineering Track**. All rights reserved.
