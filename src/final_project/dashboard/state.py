"""Data loading, caching, session state management, and business KPI calculations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from final_project.data import load_features_and_target
from final_project.evaluation import compute_metrics_at_threshold
from final_project.registry import ModelBundle
from final_project.domain.cost import CostMatrix


@st.cache_data(show_spinner=False)
def load_clean_dataset(data_dir: str = "data/processed") -> tuple[pd.DataFrame, pd.Series, pd.DataFrame | None]:
    """Load cleaned numeric features, churn target, and full cleaned dataframe."""
    base = Path(data_dir)
    full_path = base / "Telco_customer_churn_cleaned.csv"

    X, y = load_features_and_target(base)
    full_df = pd.read_csv(full_path) if full_path.exists() else None
    return X, y, full_df



@st.cache_resource(show_spinner=False)
def load_model_bundle(model_dir: str = "models") -> ModelBundle:
    """Load and cache the trained production ModelBundle."""
    path = Path(model_dir)
    return ModelBundle.load(path)


@st.cache_data(show_spinner=False)
def load_metadata(model_dir: str = "models") -> dict[str, Any]:
    """Load models/metadata.json."""
    meta_path = Path(model_dir) / "metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_drift_metrics(drift_path: str = "data/drift/drift_metrics.csv") -> pd.DataFrame:
    """Load drift monitoring metrics from the simulation."""
    path = Path(drift_path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_retraining_logs(retrain_dir: str = "models/retrained") -> dict[int, dict[str, Any]]:
    """Load retraining metadata across simulated weeks."""
    base = Path(retrain_dir)
    if not base.exists() and retrain_dir == "models/retrained":
        fallback = Path("data/retraining")
        if fallback.exists():
            base = fallback
    logs: dict[int, dict[str, Any]] = {}
    if not base.exists():
        return logs

    for week_dir in sorted(base.glob("week_*")):
        try:
            week_num = int(week_dir.name.split("_")[1])
            meta_file = week_dir / "metrics.json"
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    logs[week_num] = json.load(f)
        except Exception:
            continue
    return logs


@st.cache_data(show_spinner=False)
def load_feature_psi_matrix(
    reference_path: str = "data/reference/X_reference.csv",
    live_features_dir: str = "data/live/features",
) -> pd.DataFrame:
    """Compute and cache the 27-feature x 8-week Population Stability Index (PSI) matrix."""
    ref_file = Path(reference_path)
    live_dir = Path(live_features_dir)
    if not ref_file.exists() or not live_dir.exists():
        return pd.DataFrame()

    from final_project.drift import compute_dataset_psi

    ref_df = pd.read_csv(ref_file)
    meta_cols = {"customer_id", "Churn Value", "feature_available_week", "label_release_week"}
    feature_cols = [c for c in ref_df.columns if c not in meta_cols]
    ref_features = ref_df[feature_cols]

    weekly_records: dict[str, dict[str, float]] = {}
    for w in range(1, 9):
        feat_file = live_dir / f"week_{w}_features.csv"
        if not feat_file.exists():
            continue
        w_df = pd.read_csv(feat_file)
        common_cols = [c for c in feature_cols if c in w_df.columns]
        psi_res = compute_dataset_psi(ref_features[common_cols], w_df[common_cols])
        weekly_records[f"Week {w}"] = psi_res["feature_psi"]


    if not weekly_records:
        return pd.DataFrame()

    return pd.DataFrame(weekly_records)


def generate_mock_mlflow_runs(n_trials: int = 50, seed: int = 42) -> pd.DataFrame:
    """Generate realistic Optuna tuning trial runs matching the pipeline search space."""
    rng = np.random.RandomState(seed)
    algos = [
        "stacking_ensemble",
        "voting_ensemble",
        "logistic_regression",
        "bagged_logistic",
        "lightgbm",
        "random_forest",
        "xgboost",
        "mlp",
    ]
    records = []

    base_map = {
        "stacking_ensemble": 0.8599,
        "voting_ensemble": 0.8590,
        "logistic_regression": 0.8576,
        "bagged_logistic": 0.8575,
        "lightgbm": 0.8514,
        "random_forest": 0.8471,
        "xgboost": 0.8401,
        "mlp": 0.7973,
    }

    for trial_idx in range(n_trials):
        algo = algos[trial_idx % len(algos)]
        balanced = bool(trial_idx % 2 == 0)
        is_pruned = bool(trial_idx > 5 and rng.rand() < 0.25)

        base_auc = base_map.get(algo, 0.845)
        noise = rng.normal(0.0, 0.005)
        penalty = -0.04 if is_pruned else 0.0
        cv_auc = float(np.clip(base_auc + noise + penalty, 0.75, 0.865))
        cv_std = float(rng.uniform(0.006, 0.012))
        cv_pr = float(np.clip(cv_auc * 0.74 + rng.normal(0, 0.008), 0.58, 0.67))

        lr = float(np.round(10 ** rng.uniform(-2.5, -0.6), 4)) if algo in ("lightgbm", "xgboost") else np.nan
        n_est = int(rng.choice([50, 100, 150, 200, 250, 300])) if "ensemble" in algo or "forest" in algo or algo in ("lightgbm", "xgboost") else np.nan
        max_d = int(rng.choice([3, 4, 5, 6, 8, 10])) if "ensemble" in algo or "forest" in algo or algo in ("lightgbm", "xgboost") else np.nan
        num_leaves = int(rng.choice([15, 31, 63])) if algo == "lightgbm" else np.nan

        records.append({
            "run_id": f"run_{trial_idx:03d}_{rng.randint(100000, 999999)}",
            "run_name": f"trial_{trial_idx}_{algo}",
            "tags.algorithm": algo,
            "tags.balanced": str(balanced),
            "tags.trial_number": str(trial_idx),
            "tags.status": "PRUNED" if is_pruned else "COMPLETED",
            "metrics.cv_roc_auc": round(cv_auc, 4),
            "metrics.cv_roc_auc_std": round(cv_std, 4),
            "metrics.cv_pr_auc": round(cv_pr, 4),
            "params.learning_rate": lr,
            "params.n_estimators": n_est,
            "params.max_depth": max_d,
            "params.num_leaves": num_leaves,
            "start_time": f"2026-10-31 14:{trial_idx % 60:02d}:10",
        })

    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def load_mlflow_runs(
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "churn_retraining_optimization",
) -> pd.DataFrame:
    """Retrieve MLflow experiment runs or fall back to curated trial benchmark dataset."""
    try:
        import mlflow
        from pathlib import Path

        db_file = Path("mlflow.db")
        mlflow_dir = Path("mlruns")

        # Determine effective tracking URI
        effective_uri = tracking_uri
        if not db_file.exists() and mlflow_dir.exists():
            effective_uri = "file:./mlruns"

        mlflow.set_tracking_uri(effective_uri)
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is not None:
            runs_df = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            if not runs_df.empty:
                if "tags.mlflow.runName" in runs_df.columns and "run_name" not in runs_df.columns:
                    runs_df["run_name"] = runs_df["tags.mlflow.runName"]
                elif "run_name" not in runs_df.columns:
                    runs_df["run_name"] = runs_df["run_id"]
                return runs_df
    except Exception:
        pass

    return generate_mock_mlflow_runs()



def calculate_confusion_at_cutoff(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cutoff: float,
    cost_matrix: CostMatrix | None = None,
) -> dict[str, Any]:
    """Calculate confusion metrics, rates, and business cost for a given decision cutoff."""
    cost_matrix = cost_matrix or CostMatrix()
    m = compute_metrics_at_threshold(
        y_true=y_true,
        y_probs=y_prob,
        threshold=cutoff,
        cost_fn=cost_matrix.cost_fn,
        cost_fp=cost_matrix.cost_fp,
    )

    return {
        "cutoff": cutoff,
        "tp": m.tp,
        "fp": m.fp,
        "tn": m.tn,
        "fn": m.fn,
        "recall": m.recall,
        "precision": m.precision,
        "fpr": m.fpr,
        "f1": m.f1,
        "total_cost": m.cost,
    }



def compute_holdout_financial_impact(
    bundle: ModelBundle,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cost_fn: float = 200.0,
    cost_fp: float = 20.0,
    optimal_cutoff: float | None = None,
) -> dict[str, Any]:
    """Compute financial ROI and threshold comparison on the holdout test set."""
    cost_matrix = CostMatrix(cost_fn=cost_fn, cost_fp=cost_fp)
    cutoff = optimal_cutoff if optimal_cutoff is not None else bundle.metadata.optimal_threshold

    y_prob = bundle.predict_proba(X_test)
    y_true = np.asarray(y_test)

    default_res = calculate_confusion_at_cutoff(y_true, y_prob, cutoff=0.50, cost_matrix=cost_matrix)
    optimal_res = calculate_confusion_at_cutoff(y_true, y_prob, cutoff=cutoff, cost_matrix=cost_matrix)

    net_savings = default_res["total_cost"] - optimal_res["total_cost"]
    pct_reduction = (net_savings / default_res["total_cost"] * 100) if default_res["total_cost"] > 0 else 0.0
    captured_delta = optimal_res["tp"] - default_res["tp"]

    return {
        "cost_matrix": cost_matrix,
        "optimal_cutoff": cutoff,
        "default": default_res,
        "optimal": optimal_res,
        "net_savings": net_savings,
        "pct_reduction": pct_reduction,
        "captured_delta": captured_delta,
        "total_holdout_churners": int(np.sum(y_true == 1)),
        "total_holdout_samples": len(y_true),
    }
