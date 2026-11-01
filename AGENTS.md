# AGENTS.md — Final Project (Churn Retraining Pipeline)

## Project
uv-managed Python project (Python 3.14). Core dependencies: `pandas`, `numpy`, `matplotlib`, `openpyxl`, `scikit-learn`, `xgboost`, `lightgbm`, `optuna`, `mlflow`, `joblib`, `shap`. Per ADR 0001, Milestone 2 replaces the simulator's hand-rolled logistic regression with a production-trained model artifact.

## How to run
```
python notebooks/simulate_retraining.py
```
Run from the project root. It reads `data/processed/X_features.csv` and `y_target.csv`, simulates 8 weekly batches with 2-week label lag, injects covariate + concept drift at weeks 5–8, and triggers retraining on performance decay (rolling 2-week live AUC). Outputs go to `data/live/`, `data/drift/`, `models/retrained/`.

No lint, test, typecheck, CI, or pre-commit config exists. There is no formal test suite — verify by re-running the simulator and checking output CSVs.

## Data paths (source of truth)
- `data/processed/X_features.csv` — 7,043 × 27, all numeric, 0 NaNs
- `data/processed/y_target.csv` — Churn Value (5,174 retained / 1,869 churned)
- `data/processed/Telco_customer_churn_cleaned.csv` — full cleaned dataset
- `data/raw/Telco_customer_churn.xlsx` — source Excel (7,043 × 33)

**Do NOT** use `notebooks/data/` — it's a stale legacy copy. `data/processed/` is authoritative.

## Live data layout (produced by simulator)
- `data/live/features/week_N_features.csv` — features available at week N
- `data/live/labels/week_N_labels.csv` — labels released at week N+2 (2-week lag)
- `data/live/pending/` — weeks awaiting label release (weeks 7–8 after a run)
- `data/drift/drift_metrics.csv` — per-week AUC for both strategies
- `models/retrained/week_N/metrics.json` — retraining artifacts (weeks 3 & 7 trigger)

## Simulator behavior (verified)
Retraining triggers at weeks 3 (initial) and 7 (performance decay: recent AUC 0.8198 < best 0.8716 − 0.05). Never-retrain baseline holds flat at ~0.847 AUC on static reference. Drift starts week 5 (Monthly Charges ×1.25, Tenure ×0.7, 30% label flip on high-charge retained).

## Repo layout
- `notebooks/cleaning_for_retraining.ipynb` — cleaning notebook (complete, verified)
- `notebooks/simulate_retraining.py` — simulator script (the main entrypoint)
- `docs/Human/projects.md` — project overview (Project 1 = this churn project)
- `src/final_project/__init__.py` — package entrypoints

- `src/final_project/kodeagnt` — scratch/experimental CodeActAgent snippet, **not** part of the pipeline
- `README.md` — empty

## Conventions
- **Never commit directly to master**: Always create a feature/ticket branch (e.g. `feat/issue-<n>`), commit to that branch, push, and open a Pull Request.
- Paths in code are relative to project root (e.g. `data/processed/X_features.csv`).
- All numeric cleaning decisions are documented inline in `simulate_retraining.py` (DECISION / ALTERNATIVES / WHY NOT PICKED blocks).

## Agent skills

### Issue tracker

GitHub issues. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical roles mapped 1:1. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context (`GLOSSARY.md` + `docs/adr/`). See `docs/agents/domain.md`.
