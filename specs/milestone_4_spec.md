# Milestone 4 Specification: MLOps, Monitoring & Web Dashboard

## Goal Description

Deliver an interactive, enterprise-grade **Streamlit operations dashboard** for the Customer Churn Retraining & Prediction System. The dashboard bridges technical data science outputs and executive decision-making, providing real-time visibility into financial ROI, candidate model diagnostics, streaming drift and retraining events, live frontline retention scoring with SHAP waterfall attribution, and MLflow experiment runs.

---

## Architectural Principles & Aesthetics

1. **Rich & Modern Aesthetics**:
   - Sleek dark theme palette (`#0E1117`, `#161B22`, `#1F2937`, `#3B82F6`, `#10B981`, `#EF4444`, `#F59E0B`).
   - Custom CSS injected with modern typography (`Outfit` / `Inter` Google Fonts).
   - Glassmorphic KPI cards with subtle border gradients and micro-hover states.
   - Plotly interactive visualizations matching the design system (dark template, styled tooltips, responsive margins).

2. **Decoupled Architecture**:
   - Dashboard views consume domain models (`CostMatrix`, `ModelBundle`, `StreamingBatchEngine`) and reporting functions directly without duplicating logic.
   - Dual execution mode for Live Scoring: can query the local `ModelBundle` or the FastAPI REST service (`POST /predict`, `POST /explain`).
   - Caching via `@st.cache_data` and `@st.cache_resource` for instant page loads and zero redundant disk I/O.

3. **Unified CLI Integration**:
   - Accessible via console command `churn dashboard [--port 8501] [--host 0.0.0.0]`.

---

## Page Architecture & Functional Modules

```
src/final_project/dashboard/
├── __init__.py                 # Dashboard module exports
├── app.py                      # Main Streamlit entrypoint & router
├── styles.py                   # Custom CSS, Google Fonts, theme tokens
├── state.py                    # Session state, cached data loaders
└── pages/
    ├── __init__.py
    ├── kpi_overview.py         # Page 1: Executive KPI & Financial Economics
    ├── model_diagnostics.py    # Page 2: Benchmark & Evaluation Diagnostics
    ├── drift_monitor.py        # Page 3: Streaming Drift Timeline & PSI Heatmap
    ├── retention_triage.py     # Page 4: Real-time Customer Scoring & SHAP Triage
    └── experiment_hub.py       # Page 5: MLflow Experiment & Parameter Viewer
```

### Page 1: Executive KPI & Financial Economics
- **Dataset Metrics**: Total customers (7,043), base churn rate (26.54%), retained count (5,174), churned count (1,869).
- **Cost-Utility Economics**:
  - Interactive Cost Matrix configuration sliders: Cost of Missed Churn ($C_{\text{FN}}$, default $200) vs Cost of Retention Offer ($C_{\text{FP}}$, default $20).
  - Business Impact Metrics: Expected loss at default cutoff (0.50: $36,740) vs optimal cutoff (0.12: $14,880).
  - Net Business Savings: **$21,860.00** (**59.5% loss reduction**).
  - Recall @ Cutoff: **93.05%** (detecting 348 out of 374 holdout churners).
- **Waterfall & ROI Breakdown**: Visual comparison of baseline loss vs optimized strategy.

### Page 2: Benchmark & Model Diagnostics
- **Candidate Benchmark**: Multi-model comparison table (LightGBM, Random Forest, XGBoost) across 5-fold CV ROC-AUC, PR-AUC, F1, Recall.
- **Interactive Evaluation Curves**:
  - ROC Curve with AUC score and random baseline.
  - Precision-Recall Curve with optimal cutoff point (0.12) highlighted.
  - Cost Curve vs Probability Cutoff displaying the minimum loss valley.
  - Interactive Confusion Matrix heatmap (TP: 348, FN: 26, FP: 348, TN: 687).
  - Calibration Curve with Brier score assessing probability reliability.

### Page 3: Streaming Drift Timeline & PSI Heatmap (ADR 0001)
- **8-Week Retraining Simulation**:
  - Dual line chart comparing rolling 2-week live AUC for Adaptive Retraining vs Never-Retrain baseline.
  - Visual indicators for Covariate & Concept Drift onset at Week 5.
  - Retraining trigger event markers at Week 3 (initial) and Week 7 (performance decay: 0.8198 < 0.8716 − 0.05).
- **Population Stability Index (PSI) Heatmap**:
  - Feature-by-week heatmap displaying drift progression across weeks 1–8.
  - Color-coded severity zones: Stable (< 0.10, Green), Moderate (0.10–0.25, Yellow), Significant Drift (>= 0.25, Red).
  - Highlighting severe shift in `Monthly Charges` and `Tenure Months`.
- **Sample Weighting Visualizer**:
  - Exponential sample decay curve ($0.85^{\text{weeks\_old}}$) showing how older observations are discounted.

### Page 4: Real-Time Customer Scoring & Retention Triage
- **Interactive Customer Form**:
  - Demographic & Account info: Gender, Senior Citizen, Partner, Dependents, Tenure Months.
  - Services: Phone, Multiple Lines, Internet Service, Online Security, Tech Support, Streaming TV/Movies, etc.
  - Financials: Contract type, Paperless Billing, Payment Method, Monthly Charges, Total Charges.
  - Presets: Quick-load buttons for "High Risk Customer", "Low Risk Loyal Customer", "Drifted High-Charge Customer".
- **Real-Time Prediction Engine**:
  - Outputs Churn Probability, Risk Tier (Low / Medium / High), and Binary Decision at optimal 0.12 cutoff.
  - Plotly Speedometer Gauge with color-coded risk bands.
- **Frontline SHAP Triage Explanation**:
  - Signed parent feature attributions (Risk Drivers pushing toward churn in Crimson vs Protective Factors in Emerald).
  - Actionable retention recommendations based on top drivers (e.g., offer 1-year contract incentive if month-to-month contract is top driver; offer tech support discount if lack of tech support is top driver).

### Page 5: MLflow Experiment & Parameter Hub
- **Experiment Runs Table**:
  - Reads `./mlruns` tracking directory.
  - Displays Run ID, Timestamp, Candidate Architecture, CV ROC-AUC, Holdout Loss, and Hyperparameters.
- **Hyperparameter Sensitivity Plot**:
  - Interactive scatter/parallel coordinates of learning rate, num leaves, max depth vs ROC-AUC.
- **Artifact Explorer**:
  - Direct links and previews of logged models, evaluation reports, and figures.

---

## Tracer Bullets & Work Breakdown

| Issue | Title | Scope |
|---|---|---|
| **#23** | `feat(dashboard): application shell, design system, theme tokens, and executive KPI page` | App shell, navigation, custom CSS/typography, Page 1 KPI & Financial economics. |
| **#24** | `feat(dashboard): model benchmark, ROC/PR curves, and evaluation diagnostics page` | Page 2 candidate benchmark comparison and interactive Plotly diagnostic curves. |
| **#25** | `feat(dashboard): drift monitoring timeline, PSI heatmaps, and retraining trigger page` | Page 3 8-week simulation timeline, PSI heatmaps, and exponential sample decay view. |
| **#26** | `feat(dashboard): real-time customer retention scoring, triage, and interactive SHAP waterfall` | Page 4 interactive customer form, risk gauge, and signed SHAP retention triage. |
| **#27** | `feat(dashboard): MLflow experiment viewer, CLI command (churn dashboard), and test suite` | Page 5 MLflow hub, `churn dashboard` CLI entrypoint, and automated unit test suite. |

---

## Verification & Acceptance Criteria

1. **Dashboard Execution**:
   - `uv run streamlit run src/final_project/dashboard/app.py` runs without errors or warnings.
   - `uv run churn dashboard` successfully launches the app.
2. **Page Navigation & State**:
   - All 5 pages render cleanly with cohesive dark theme styling and instant caching.
3. **Accuracy & Consistency**:
   - Financial figures match `reports/evaluation_report.md` ($21,860 savings, 0.12 cutoff).
   - Drift timeline matches `data/drift/drift_metrics.csv` (weeks 3, 5, and 7 triggers).
   - Scoring matches `POST /predict` and `POST /explain` outputs.
4. **Automated Test Coverage**:
   - Dedicated test suite in `tests/test_dashboard.py` verifying state helpers, KPI calculations, and page render functions.
   - All 94 automated tests pass hermetically.

---

## As-Built Deltas

- **Truthful Empty States**: In accordance with Stage 5 hardening (#56), all dashboard pages render clear, actionable `st.info` notices when tracking databases or model artifacts are absent, eliminating fabricated mock runs.
- **Dynamic KPI Computation**: Removed hardcoded champion model strings and cutoff values; all KPIs are dynamically loaded from `models/metadata.json` and cleaned data frames.
- **Benchmark Trace in Drift Monitor**: Plotted a distinct dashed benchmark line for the historical best-ever live AUC alongside the weekly live AUC trajectory.
- **Retraining Trigger Timeline**: Simulation triggers at weeks 3, 5, and 7 on performance decay; never-retrain baseline holds at ~0.836 AUC on static reference.

