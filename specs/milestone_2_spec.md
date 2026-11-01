## Problem Statement

Telecom businesses face customer churn, leading to revenue loss and high acquisition costs. While clean feature data and a retraining simulation harness exist, the project currently lacks a production-grade machine learning model and automated pipeline. The existing simulation relies on a toy, from-scratch logistic regression model that cannot capture non-linear feature interactions, lacks hyperparameter optimization, does not handle class imbalance systematically, and provides no interpretability or formal evaluation tooling. The team needs an end-to-end, production-ready training, evaluation, and explainability system capable of finding the highest-performing model, tuning its decision threshold for financial impact, explaining churn drivers cleanly to business stakeholders, and exporting models that downstream simulation and production workflows can consume.

## Solution

Build a modular production machine learning pipeline and CLI interface that:
1. Trains and benchmarks tabular tree classification algorithms (Random Forest, LightGBM, XGBoost) using algorithmic class weighting (`class_weight='balanced'`, `scale_pos_weight`) to preserve one-hot categorical feature integrity and calibrated probabilities.
2. Performs automated Bayesian hyperparameter tuning across 5-fold stratified cross-validation using a unified Optuna study (50–60 trials total) with median/ASHA pruning, optimizing for ROC-AUC while tracking trials and experiments in local MLflow.
3. Evaluates the winning model against an isolated stratified holdout test split, reporting ROC-AUC, PR-AUC, confusion matrices, and Brier score.
4. Optimizes the classification decision threshold to minimize expected financial loss via a Business Cost-Utility Matrix ($C_{\text{FN}} = \$200$ per missed churner vs. $C_{\text{FP}} = \$20$ per unnecessary retention offer), reporting $F_1$ and $F_{\beta=2}$ alongside.
5. Provides rich model interpretability using SHAP TreeExplainer, grouping one-hot dummy variables back to parent categorical features (`Contract`, `Internet Service`, `Payment Method`) for business clarity alongside dependence and individual waterfall plots.
6. Exports serialized model artifacts and metadata for deployment.
7. Exposes an intuitive command-line interface with dedicated subcommands (`train`, `evaluate`, `explain`) alongside a clean programmatic Python API (`churn`).
8. Integrates with the retraining simulator using a Dual Monitoring Architecture (immediate Population Stability Index [PSI] on incoming weekly features + delayed live AUC decay on released labels) and an Exponential Sample Decay retraining data window.

## User Stories

1. As an ML engineer, I want an automated data loading and splitting pipeline with stratified holdout partitioning, so that models are trained and tested on representative class distributions without data leakage.
2. As an ML engineer, I want to evaluate high-performance tree ensembles (Random Forest, LightGBM, XGBoost) with native class weighting, so that minority churn detection is maximized without generating invalid synthetic points on one-hot features.
3. As an ML engineer, I want hyperparameter tuning driven by Bayesian optimization (Optuna TPE) in a unified study of 50–60 trials with early stopping, so that optimal model parameters are discovered efficiently without overfitting validation folds.
4. As an ML engineer, I want hyperparameter search evaluated across 5-fold stratified cross-validation on the training split, so that performance estimates are robust.
5. As a business stakeholder, I want models evaluated on both ROC-AUC and PR-AUC, so that ranking capability is evaluated globally while precision on minority churners is strictly observed.
6. As an ML engineer, I want all tuning trials, parameters, and metric curves logged to an MLflow experiment, so that model exploration is reproducible and auditable.
7. As a data scientist, I want comprehensive test-set evaluation including PR-AUC, confusion matrices, precision, recall, and Brier calibration score, so that both discrimination and calibration are thoroughly understood.
8. As a churn retention manager, I want the decision threshold tuned using a Business Cost-Utility Matrix, so that the financial loss of missed churners is balanced against the cost of customer retention incentives.
9. As a retention manager, I want evaluation results reported at the default 0.5 threshold, the F1-maximizing threshold, and the business cost-optimal threshold, so that operational trade-offs are transparent.
10. As an executive, I want a structured markdown evaluation report with visual diagnostic plots, so that model performance can be reviewed and approved without inspecting raw logs.
11. As a customer success manager, I want SHAP feature importance summaries grouped by parent categorical variables, so that I can clearly understand which service factors drive churn without being confused by fragmented dummy columns.
12. As an analyst, I want SHAP dependence plots for key numerical drivers like monthly charges and tenure, so that I can observe non-linear risk thresholds and tipping points.
13. As an account retention specialist, I want individualized SHAP explanation plots for sampled customers, so that retention agents can see exactly why a specific customer was flagged as a high churn risk.
14. As an MLOps engineer, I want the best trained model and its run metadata serialized to disk, so that downstream serving systems and the retraining simulator can load the model seamlessly.
15. As a developer, I want a `churn` CLI with `train`, `evaluate`, and `explain` subcommands, so that pipeline steps can be triggered independently or orchestrated via scripts.
16. As an automation engineer, I want programmatic Python entry points for training, evaluation, and explanation, so that other Python packages and automated test suites can invoke the pipeline directly in-process.
17. As an MLOps engineer, I want the retraining simulator updated to implement a Dual Trigger (immediate feature PSI on Week $N$ + delayed AUC decay on Week $N+2$) and Exponential Sample Decay retraining, so that drift monitoring and model updates reflect true production dynamics under label lag.
18. As a test engineer, I want integration tests to exercise the programmatic pipeline API on synthetic sample inputs, so that regressions in pipeline contracts or artifact creation are caught rapidly in CI.

## Implementation Decisions

### Architectural Structure & Modules
- The pipeline will be organized into modular domain components responsible for:
  - Data loading, validation, and stratified splitting.
  - Native algorithmic class weighting adapters (`class_weight='balanced'`, `scale_pos_weight`).
  - Candidate model configurations (Random Forest, LightGBM, XGBoost).
  - Hyperparameter optimization engine using Optuna (single unified multi-model study).
  - Experiment tracking adapter using MLflow.
  - Comprehensive metric computation, Brier calibration, and cost-utility threshold optimization.
  - Model serialization and metadata packaging via `joblib`.
  - Explainability and SHAP visualization generation with parent-feature grouping.
  - CLI command parsing (`churn train`, `churn evaluate`, `churn explain`).
  - Programmatic pipeline interfaces (`run_training`, `run_evaluation`, `run_explanation`).

### Algorithms & Imbalance Strategy
- Candidate algorithms are restricted to tabular tree ensembles: Random Forest, LightGBM, and XGBoost.
- SMOTE is omitted because linear Euclidean interpolation on one-hot encoded columns generates invalid fractional states and distorts probability calibration; native class weighting is used across all models.
- Upfront feature selection is omitted; all 27 numeric features are retained during training, with SHAP analysis used post-hoc to evaluate feature contribution.

### Hyperparameter Search & Experiment Tracking
- Optuna will manage Bayesian optimization using its Tree-structured Parzen Estimator (TPE) sampler.
- A unified search space evaluates algorithm selection and hyperparameters concurrently across 50–60 trials total with pruning.
- MLflow will track each trial's parameters, cross-validation metrics, best model checkpoints, and parent experiment summaries in local storage.

### Evaluation & Threshold Optimization
- Evaluation on the holdout test partition will report ROC-AUC, PR-AUC, Confusion Matrix, Classification Report (Precision, Recall, F1, F2), and Brier Score.
- A threshold optimizer sweeps classification cutoffs across [0.01, 0.99] to find the cutoff that minimizes expected business loss using a cost matrix ($C_{\text{FN}} = \$200, C_{\text{FP}} = \$20$), reporting comparative metrics against default 0.5 and maximum F1 cutoffs.
- Performance will be written to a formatted markdown report alongside exported figure artifacts (ROC curves, PR curves, calibration curves, confusion matrices).

### Interpretability with SHAP
- TreeExplainer will be utilized to generate exact Shapley values for the selected tree ensemble model on holdout data.
- SHAP values for one-hot dummy columns will be summed back into parent categorical features (`Contract`, `Internet Service`, `Payment Method`) for global summary bar charts and beeswarm visualizations, with level-specific drilldown plots preserved.
- Global feature summary plots and dependence plots for top numerical drivers (`Monthly Charges`, `Tenure`) will be exported to the reporting directory.
- Individual waterfall explanations will be generated for representative true positive churners and false positive edge cases.

### Model Serialization & Retraining Simulator Integration
- The winning model pipeline (preprocessor + model) will be persisted via `joblib`.
- An accompanying JSON metadata file will capture model architecture, hyperparameters, cross-validation score, holdout test metrics, optimal decision threshold, training date, and feature names.
- In accordance with ADR 0001, the retraining simulator (`notebooks/simulate_retraining.py`) will be updated to:
  1. Load the production model artifact.
  2. Implement an **Unsupervised Data Drift Monitor** computing Population Stability Index (PSI) on incoming Week $N$ unlabeled features.
  3. Implement a **Supervised Performance Decay Trigger** evaluating rolling 2-week live AUC on released Week $N+2$ labels.
  4. Retrain using **Exponential Sample Decay** ($0.85^{\text{weeks\_old}}$) so post-drift data guides the adaptation while preserving sample size.

### CLI & Programmatic Seams
- A unified CLI command (`churn`) with subcommands `train`, `evaluate`, and `explain` will be registered as a console script entry point, implemented using standard library `argparse`.
- Programmatic Python orchestrators will expose high-level functions that accept configuration objects and return structured result dictionaries.

## Testing Decisions

### What Makes a Good Test
- Tests must verify external observable behavior, API contracts, and artifact generation rather than private class internals.
- Tests should execute in seconds by configuring mini-runs (tiny synthetic datasets, 2 trials, single fold) rather than running the full 60-trial optimization during automated test runs.
- Assertions should verify:
  - High-level pipeline functions successfully run and return expected data structures and metric keys.
  - Model artifacts and JSON metadata files are saved with valid schemas and loadable states.
  - Evaluation reports and visual artifacts are written to the expected paths.
  - Cost-utility threshold optimization correctly identifies thresholds within bounds and produces valid financial metrics.

### Modules Tested
- The programmatic pipeline API functions (`run_training`, `run_evaluation`, `run_explanation`).
- Cost-utility threshold optimization logic under boundary condition probabilities.
- Model artifact loading, PSI drift calculation, and prediction contract conformance with the retraining simulator.

## Out of Scope
- Deep learning or neural network architectures (PyTorch, TensorFlow).
- SMOTE or synthetic oversampling techniques.
- Live HTTP/REST serving microservices (FastAPI/Flask endpoints) — scheduled for Milestone 3.
- Remote cloud experiment registries (AWS S3, Databricks MLflow) — tracking is local.
- Real-time streaming prediction or Kafka integration.

---

## As-Built Deltas

- **Experiment Tracking Storage**: MLflow tracking uses an embedded SQLite database (`sqlite:///mlflow.db`) to enable ACID queries and robust run aggregation.
- **Expanded Candidate Architectures**: Implemented `MLPClassifier`, `LogisticRegression` pipeline, `BaggingClassifier`, `KNN`, and ensemble models (`VotingClassifier`, `StackingClassifier`) alongside tree models.
- **Simulation Trigger Timeline**: Simulation triggers at weeks 3, 5, and 7 on performance decay; never-retrain baseline holds at ~0.836 AUC on static reference.
- **Retrained Artifact Directory**: Retrained models and metrics are saved to `models/retrained/week_N/`.

