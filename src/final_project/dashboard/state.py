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
def load_retraining_logs(retrain_dir: str = "data/retraining") -> dict[int, dict[str, Any]]:
    """Load retraining metadata across simulated weeks."""
    base = Path(retrain_dir)
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

    total_negatives = m.tn + m.fp
    fpr = m.fp / total_negatives if total_negatives > 0 else 0.0

    return {
        "cutoff": cutoff,
        "tp": m.tp,
        "fp": m.fp,
        "tn": m.tn,
        "fn": m.fn,
        "recall": m.recall,
        "precision": m.precision,
        "fpr": fpr,
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
