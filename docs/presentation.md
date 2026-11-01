# ⚡ ChurnOps: Executive Presentation & Project Defense Guide

> **Project**: Customer Churn Prediction & Automated Retraining System  
> **Track**: DEPI Machine Learning Engineering — Project 1  
> **Format**: Slide-by-Slide Deck Outline, Live Demo Runbook, and Defense Q&A  
> **Duration**: 15–20 minutes presentation + 10 minutes Q&A  

---

## 📑 Presentation Overview & Agenda

1. **The Business Problem & Financial Opportunity** (Slides 1–2)
2. **Data Pipeline & Exploratory Insights** (Slide 3)
3. **Model Engineering, Ensembles & Bayesian Optimization** (Slides 4–5)
4. **Cost-Utility Threshold Optimization** (Slide 6)
5. **SHAP Model Interpretability & Parent Grouping** (Slide 7)
6. **Continuous Retraining & Drift Architecture (ADR 0001)** (Slides 8–9)
7. **Production Serving & Cloud Deployment Architecture** (Slide 10)
8. **Live System Demonstration Workflow** (Slides 11–12)
9. **Future Roadmap & Strategic Enhancements** (Slide 13)
10. **Defense Q&A Preparation** (Reference Appendix)

---

## 📽️ Slide-by-Slide Deck Content & Speaker Notes

### Slide 1: Title Slide & Project Identity
* **Headline**: **ChurnOps: Closed-Loop Customer Churn Prediction & Automated Retraining**
* **Subtitle**: An enterprise-grade, financial-loss-calibrated machine learning system for telecommunications customer retention.
* **Presenter Information**: Machine Learning Engineering Team / DEPI MLE Track
* **Visual Elements**:
  - High-level architecture badge (Python 3.14, FastAPI, Streamlit, MLflow, Docker, Scikit-learn, LightGBM, XGBoost, Optuna).
* **Speaker Script**:
  > *"Good morning/afternoon, evaluators and colleagues. Today we present ChurnOps, an end-to-end machine learning system developed to solve one of the most critical challenges in subscription telecommunications: customer churn. Rather than treating churn as an abstract statistical metric, ChurnOps integrates machine learning directly with business economics, real-time explainability, and automated continuous retraining."*

---

### Slide 2: Executive Summary & Financial Problem Formulation
* **Key Points**:
  - Customer acquisition is **5–7× more expensive** than customer retention.
  - Standard machine learning models maximize classification accuracy at a default `0.50` probability cutoff, failing to account for asymmetric business costs.
  - **Asymmetric Cost Matrix**:
    - **Cost of False Negative ($C_{\text{FN}}$)**: **$200.00** (lost future lifetime value from an unaddressed churner).
    - **Cost of False Positive ($C_{\text{FP}}$)**: **$20.00** (cost of a targeted loyalty incentive or discount).
* **Quantified Bottom-Line Impact**:
  - Standard cutoff (`0.50`): Missing 172 churners $\to$ **$36,760** expected business loss.
  - ChurnOps calibrated cutoff (`0.12`): Captures **93.05% of churners** $\to$ **$14,900** expected loss (**$21,860 net savings, 59.5% financial loss reduction**).
* **Speaker Script**:
  > *"In telecommunications, letting an at-risk customer leave costs ten times more than giving them a retention incentive. If we use a naive 0.50 threshold, we miss almost half of our churners, resulting in over $36,000 in lost customer lifetime value across our holdout test set alone. By embedding an empirical Cost-Utility loss function into our pipeline, we proved mathematically that a cutoff of 0.12 maximizes return on investment, reducing business loss by nearly 60%."*

---

### Slide 3: Milestone 1 — Data Collection, Cleaning & Exploration
* **Dataset Characteristics**:
  - Telco Customer Churn: 7,043 customer accounts across 33 initial dimensions.
  - Target: Binary Churn Value (5,174 retained / 1,869 churned; ~26.5% base churn rate).
* **Preprocessing Engineering Pipeline**:
  - **Null Handling**: Handled 11 blank spaces in `Total Charges` representing brand-new zero-tenure customers (imputed cleanly to `0.0`).
  - **Feature Normalization**: Robust scaling of numeric continuous attributes (`Tenure Months`, `Monthly Charges`, `Total Charges`).
  - **Categorical Encoding**: One-hot encoding yielding a strictly numeric, NaN-free $7,043 \times 27$ feature matrix.
  - **Zero Synthetic Distortion**: Native algorithmic class weighting chosen over SMOTE to prevent synthetic sample distortion in one-hot spaces.
* **Speaker Script**:
  > *"In Milestone 1, we sanitized the dataset into an authoritative 7,043 by 27 matrix. All decisions were strictly documented with explicit trade-off rationales. Crucially, we avoided SMOTE to protect against generating artificial points in discrete one-hot categorical spaces, relying instead on algorithmic sample weighting during model optimization."*

---

### Slide 4: Milestone 2 — Model Benchmarking & Multi-Algorithm Exploration
* **Comprehensive Model Evaluation (5-Fold Stratified CV)**:
  - **LightGBM** (Rank 1): ROC-AUC `0.8528 ± 0.0104`, PR-AUC `0.6651 ± 0.0189`
  - **Stacking Ensemble** (LightGBM + XGBoost + Logistic Regression Meta-Learner): ROC-AUC `0.8491 ± 0.0112`
  - **Voting Ensemble** (Soft voting): ROC-AUC `0.8485 ± 0.0109`
  - **XGBoost**: ROC-AUC `0.8447 ± 0.0121`
  - **Random Forest**: ROC-AUC `0.8415 ± 0.0128`
  - **Bagged Logistic Regression**: ROC-AUC `0.8430 ± 0.0115`
  - **Logistic Regression Baseline**: ROC-AUC `0.8410 ± 0.0118`
* **Model Selection Rational**:
  - LightGBM delivers top-tier discriminatory power, superior native NaN/categorical handling, fastest inference latency (<1.5 ms), and clean TreeSHAP compatibility.
* **Speaker Script**:
  > *"We benchmarked seven distinct algorithm architectures using 5-fold stratified cross-validation. While ensemble stacking and voting models delivered robust competitive scores, LightGBM emerged as the champion model with a cross-validated ROC-AUC of 0.8528 and the lowest latency profile required for sub-10ms real-time API scoring."*

---

### Slide 5: Milestone 2 — Bayesian Hyperparameter Tuning & MLflow Tracking
* **Optuna Bayesian Optimization**:
  - Utilized Tree-structured Parzen Estimators (`TPESampler`) with `MedianPruner` across 50 trials.
  - Tuned parameters: `learning_rate` [0.01, 0.20], `n_estimators` [50, 400], `num_leaves` [15, 127], `min_child_samples` [10, 60], `subsample` [0.6, 1.0], `colsample_bytree` [0.6, 1.0].
* **MLflow Tracking Hub Integration**:
  - Centralized SQLite backend (`mlflow.db`) tracking every run trial, parameter combinations, ROC-AUC, PR-AUC, and serialized model artifacts.
  - Full reproducibility and lineage traceability from training to deployment.
* **Speaker Script**:
  > *"To extract maximum performance without manual guessing, we executed Bayesian optimization using Optuna. Every hyperparameter combination, pruning step, and resulting metric was logged to our MLflow tracking repository, providing a complete audit trail for compliance and model governance."*

---

### Slide 6: Milestone 2 — Holdout Evaluation & Cost-Utility Threshold Optimization
* **Holdout Set Performance (1,409 Unseen Customers)**:
  - **ROC-AUC**: `0.8390` | **PR-AUC**: `0.6276` | **Brier Calibration Score**: `0.1392` (<0.15 indicates exceptional probability calibration).
* **Threshold Comparison**:
  | Strategy | Cutoff | Precision | Recall | FN | FP | Expected Financial Loss |
  |---|---|---|---|---|---|---|
  | **Default Baseline** | `0.50` | 63.1% | 54.0% | 172 | 118 | **$36,760** |
  | **Max F1** | `0.38` | 58.1% | 69.3% | 115 | 187 | **$26,740** |
  | **Cost-Optimal (ChurnOps)** | **`0.12`** | 41.8% | **93.0%** | **26** | 485 | **$14,900 (-59.5%)** |
* **Speaker Script**:
  > *"When evaluated on our 20% holdout test set, the model achieved a Brier score of 0.139, confirming that our predicted probabilities reflect genuine risk. By dropping the decision threshold from 0.50 to 0.12, we captured 146 additional churning customers, slashing false negatives from 172 down to 26 and saving the enterprise $21,860."*

---

### Slide 7: Milestone 2 — Explainable AI (SHAP Parent Feature Grouping)
* **The Problem with Raw Dummy Explanations**:
  - One-hot encoding splits categories like `Contract` into three separate features, diluting and confusing frontline retention agents.
* **ChurnOps Innovation — Parent Category Grouping**:
  - Aggregates one-hot dummy attributions back into unified business entities (`Contract`, `Internet Service`, `Payment Method`).
* **Top Global Churn Drivers**:
  1. **Contract Type** (Month-to-month contracts strongly elevate risk).
  2. **Tenure Months** (Low tenure exhibits high vulnerability; high tenure creates loyalty).
  3. **Internet Service / Fiber Optic** (Higher pricing combined with tech-support friction drives departures).
  4. **Total / Monthly Charges** (Price sensitivity thresholds).
* **Speaker Script**:
  > *"A prediction is useless if frontline agents don't understand why a customer is churning. We utilized SHAP TreeExplainer, but introduced an engineering innovation: parent categorical aggregation. Rather than showing fragmented dummy variables, our API and dashboard synthesize attributions into cohesive drivers: Contract, Tenure, and Internet Service, giving customer agents immediate negotiation leverage."*

---

### Slide 8: Milestone 4 — Retraining Simulation & Drift Detection (ADR 0001)
* **Real-World Reality**:
  - Customer behavior changes over time (Covariate Drift), and billing policies alter customer loyalty (Concept Drift).
  - Ground-truth churn labels arrive with a **2-week label delay**.
* **ChurnOps Closed-Loop Simulator (`simulate_retraining.py`)**:
  - Simulates 8 chronological weekly batches under live conditions.
  - **Weeks 1–4**: Baseline steady-state operation.
  - **Weeks 5–8**: Synthetic covariate drift (charges $\times 1.25$, tenure $\times 0.70$) and concept drift (30% label flip on high charges).
* **Dual Monitoring Triggers**:
  1. **Unsupervised Drift**: Population Stability Index (PSI) computed on streaming features ($PSI > 0.25 \to$ Significant Shift).
  2. **Supervised Performance Decay**: Rolling live 2-week ROC-AUC decay ($AUC_{\text{recent}} < AUC_{\text{best}} - 0.05$).
* **Speaker Script**:
  > *"In ADR 0001, we replaced toy simulator equations with our actual production model. We simulated an 8-week production lifecycle with a realistic 2-week label delay. When covariate and concept drift were injected in week 5, our unsupervised PSI jumped from 0.002 to 0.23, and our rolling performance decay detector automatically triggered retraining cycles at weeks 5, 7, and 8, preventing model obsolescence."*

---

### Slide 9: Milestone 4 — Exponential Sample Weighting Strategy
* **Why Not Retrain on All Historical Data Equally?**:
  - Old customer patterns dilute recent shifts; forgetting all old data causes sample starvation.
* **Exponential Sample Decay**:
  $$w_i = 0.85^{\text{weeks\_old}}$$
  - Recent live data receives full weight ($w=1.0$), while older historical batches decay smoothly.
* **Simulation Verification**:
  - Model adapted rapidly to flipped labels, lifting live recent AUC back up to **1.000** while maintaining reference generalization at **0.843**.
* **Speaker Script**:
  > *"When retraining, we apply exponential sample decay with a factor of 0.85. This ensures the model prioritizes newly adapted customer behaviors while maintaining stability against overfitting, creating an autonomous self-healing machine learning lifecycle."*

---

### Slide 10: Milestone 3 — Production Serving, FastAPI & Container Architecture
* **FastAPI Microservice (`src/final_project/serving/`)**:
  - `GET /health` (Readiness, liveness, model metadata).
  - `POST /predict` (Single customer real-time inference with risk score and triage tier: Low, Moderate, High, Severe).
  - `POST /predict/batch` (High-throughput portfolio scoring).
  - `POST /explain` (Real-time customer-level SHAP attributions with parent group clustering).
* **Dual Schema Architecture**:
  - Flexible raw schema for CRM integrations (handles strings like `"Yes"`, `"Month-to-month"`).
  - Preprocessed numeric schema for internal batch workloads.
* **Docker Deployment**:
  - Multi-stage build (Python 3.14-slim, non-root user `appuser`, health probes).
  - Orchestrated via `docker-compose.yml` with port 8000 exposed.
* **Speaker Script**:
  > *"For deployment, we packaged the system into a high-performance FastAPI microservice validated with Pydantic v2. It exposes single, batch, and SHAP explanation endpoints. The application is containerized with a hardened multi-stage Dockerfile adhering to least-privilege security principles and automated health probes."*

---

### Slide 11: Milestone 4 — Interactive Streamlit Dashboard Walkthrough
* **Executive & Operational 5-Page Application Shell**:
  1. **Page 1: Executive KPI Overview**: Business portfolio risk distribution, projected revenue at risk, cost-utility ROI savings.
  2. **Page 2: Model Diagnostics & Evaluation**: Comparative ROC & PR curves, calibration diagrams, confusion matrix at optimal cutoff `0.12`.
  3. **Page 3: Real-Time Retention Triage Simulator**: Customer risk profile explorer, interactive feature what-if adjuster (e.g. converting month-to-month to 1-year contract), and real-time waterfall explainability.
  4. **Page 4: Drift Monitor & PSI Heatmap**: Weekly PSI drift timeline, feature shift distribution, and automated retraining event log.
  5. **Page 5: MLflow Experiment Hub**: Live hyperparameter scatter plots, run comparisons, and model registry artifacts.
* **Speaker Script**:
  > *"Our stakeholder dashboard is built with Streamlit and styled with a glassmorphism design system. It caters to both executives tracking portfolio revenue at risk and frontline retention agents conducting what-if simulations to save individual customers."*

---

### Slide 12: Live Demonstration Sequence
* **Step 1: API Scoring & Explanation (`/predict` & `/explain`)**
  - Demonstrate sending a high-risk customer payload via cURL/Swagger UI.
  - Verify response time (<15 ms), risk probability, triage tier (`SEVERE`), and top negative SHAP factors (Month-to-month, Fiber optic, No tech support).
* **Step 2: Streamlit Retention Triage What-If Analysis**
  - Load customer preset "High Risk - Fiber Optic Tech".
  - Demonstrate changing Contract from "Month-to-month" to "Two year" $\to$ Risk score drops from 84% down to 22% (Instant retention strategy validation).
* **Step 3: Continuous Retraining Verification**
  - Display the drift timeline in Page 4 showing the week 5 PSI jump and retraining triggers.
* **Speaker Script**:
  > *"During our live demo, we show how an agent enters a high-risk customer profile. The model flags them as SEVERE risk. The agent tests an intervention: offering a two-year contract discount. Immediately, the API calculates a risk reduction from 84% to 22%, validating the exact business action required to retain the account."*

---

### Slide 13: Future Roadmap & Strategic Next Steps
* **Planned Enhancements**:
  1. **A/B Testing Champion/Challenger Gateway**: Route 90% of live traffic to the production champion model and 10% to newly retrained candidates before automated cutover.
  2. **Automated CI/CD Retraining Trigger**: Replace cron execution with an event-driven webhook that triggers a GitHub Actions / Azure ML retraining job when PSI crosses threshold $0.25$.
  3. **Multi-Region Azure Deployment**: Expand deployment to Azure Container Apps with geo-redundant endpoints and automated auto-scaling.
* **Conclusion**:
  - ChurnOps bridges the gap between theoretical machine learning and enterprise financial execution.
* **Speaker Script**:
  > *"Looking ahead, our architecture is primed for shadow deployment gateways and automated webhook-based retraining triggers. ChurnOps is not merely a model; it is an autonomous, self-monitoring retention ecosystem. Thank you, and we welcome your questions."*

---

## 🎯 Defense Q&A: Anticipated Evaluator Questions & Model Answers

### Q1: Why did you choose a decision threshold of 0.12 instead of 0.50?
> **Answer**:  
> *"In customer churn, the business cost of a False Negative ($C_{\text{FN}}=\$200$, losing a customer's lifetime value) is ten times higher than the cost of a False Positive ($C_{\text{FP}}=\$20$, offering a retention discount to someone who wasn't going to leave). If we use a standard 0.50 cutoff, the model is penalized symmetrically, resulting in 172 missed churners and $36,760 in financial loss on our holdout set. By minimizing the total empirical loss function $\text{Loss} = 200 \times \text{FN} + 20 \times \text{FP}$, the optimal cutoff shifts mathematically to 0.12. This captures 93.05% of all churning customers and yields a net business savings of $21,860."*

### Q2: Why did you select LightGBM over complex deep learning or neural networks?
> **Answer**:  
> *"For tabular customer data with 7,043 rows and discrete categorical features, gradient boosted decision trees consistently outperform deep neural architectures in both predictive accuracy and calibration, without the risk of extreme overfitting. LightGBM achieved the top cross-validated ROC-AUC of 0.8528, executed inference in under 2 milliseconds, and provides native, mathematically exact TreeSHAP attributions, which are essential for our real-time explanation endpoint."*

### Q3: How do you handle the 2-week label delay during automated retraining?
> **Answer**:  
> *"In telecommunications, whether a customer has truly churned cannot be determined immediately; a billing cycle grace period is required. Our simulator strictly mirrors this reality: at week $N$, the pipeline only receives ground-truth labels for week $N-2$. During weeks $N-1$ and $N$, the system operates unsupervised, tracking drift via the Population Stability Index (PSI). Once labels arrive at week $N+2$, performance decay is evaluated on recent labeled batches, triggering retraining only when statistical degradation is confirmed."*

### Q4: Why did you group SHAP features into parent categories?
> **Answer**:  
> *"One-hot encoding splits a single domain concept like `Contract` into three collinear binary flags: `Contract_Month-to-month`, `Contract_One year`, and `Contract_Two year`. Presenting raw one-hot values to a non-technical retention agent creates confusion and spreads feature attribution across multiple small bars. By summing the signed SHAP values across child dummy variables back into the parent entity, we provide frontline agents with a clean, interpretable explanation of the primary business drivers."*

### Q5: How is data drift handled in production?
> **Answer**:  
> *"We utilize a dual-trigger architecture. First, unsupervised Population Stability Index (PSI) monitors streaming feature distributions against a reference baseline without needing labels. If PSI exceeds 0.25, a drift alert is logged. Second, as delayed labels arrive, rolling live ROC-AUC is measured against the best historical performance. If AUC drops by more than 0.05, automated retraining is triggered using exponential sample decay ($0.85^{\text{weeks\_old}}$) so the model prioritizes newly adapted customer patterns while retaining core historical relationships."*

---

## 🛠️ Verification Checklist for Presentation Day

- [x] Test suite passing: `uv run pytest` (75/75 passing).
- [x] Retraining simulator operational: `uv run python notebooks/simulate_retraining.py`.
- [x] Streamlit dashboard operational: `uv run streamlit run src/final_project/dashboard/app.py`.
- [x] FastAPI microservice operational: `uv run uvicorn final_project.serving.api:app --reload`.
- [x] Docker build verified: `docker compose build`.
- [x] Evaluation report and figures verified in `reports/` and `reports/figures/`.
