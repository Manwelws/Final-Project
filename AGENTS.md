# AGENTS.md — Final Project (Churn Retraining Pipeline)

## Project
uv-managed Python project (Python 3.14). Only deps: `pandas`, `numpy`, `matplotlib`, `openpyxl`. **No scikit-learn** — the simulator implements logistic regression + rank-based AUC from scratch in `notebooks/simulate_retraining.py`.

## How to run
```
python notebooks/simulate_retraining.py
```
Run from the project root. It reads `data/processed/X_features.csv` and `y_target.csv`, simulates 8 weekly batches with 2-week label lag, injects covariate + concept drift at weeks 5–8, and triggers retraining on performance decay (rolling 2-week live AUC). Outputs go to `data/live/`, `data/drift/`, `data/retraining/`.

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
- `data/retraining/week_N/metrics.json` — retraining artifacts (weeks 3 & 7 trigger)

## Simulator behavior (verified)
Retraining triggers at weeks 3 (initial) and 7 (performance decay: recent AUC 0.8198 < best 0.8716 − 0.05). Never-retrain baseline holds flat at ~0.847 AUC on static reference. Drift starts week 5 (Monthly Charges ×1.25, Tenure ×0.7, 30% label flip on high-charge retained).

## Repo layout
- `notebooks/cleaning_for_retraining.ipynb` — cleaning notebook (complete, verified)
- `notebooks/simulate_retraining.py` — simulator script (the main entrypoint)
- `projects.md` — project overview (Project 1 = this churn project)
- `src/final_project/__init__.py` — stub `main()` only
- `src/final_project/kodeagnt` — scratch/experimental CodeActAgent snippet, **not** part of the pipeline
- `README.md` — empty

## Conventions
- No AGENTS.md/CLAUDE.md/CI/pre-commit existed before this file.
- Paths in code are relative to project root (e.g. `data/processed/X_features.csv`).
- All numeric cleaning decisions are documented inline in `simulate_retraining.py` (DECISION / ALTERNATIVES / WHY NOT PICKED blocks).
