"""Shared pytest fixtures for hermetic test execution."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from final_project.registry import ModelBundle, ModelMetadata
from final_project.serving.adapter import MODEL_FEATURE_NAMES


@pytest.fixture
def synthetic_data_paths(tmp_path: Path):
    """Generate synthetic processed dataset files in a temporary directory."""
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    n_samples = 100
    data = {feature: np.random.randn(n_samples) for feature in MODEL_FEATURE_NAMES}
    X = pd.DataFrame(data)
    y = pd.Series(np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]), name="Churn Value")

    features_path = processed_dir / "X_features.csv"
    target_path = processed_dir / "y_target.csv"
    full_path = processed_dir / "Telco_customer_churn_cleaned.csv"

    X.to_csv(features_path, index=False)
    y.to_csv(target_path, index=False)

    full_df = X.copy()
    full_df["Churn Value"] = y.values
    full_df.to_csv(full_path, index=False)

    return features_path, target_path, processed_dir


@pytest.fixture
def synthetic_model_bundle_dir(tmp_path: Path):
    """Generate a fitted ModelBundle and metadata.json in a temporary directory."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    n_samples = 40
    X = pd.DataFrame({feature: np.random.randn(n_samples) for feature in MODEL_FEATURE_NAMES})
    if "Contract_Month-to-month" in X.columns:
        X.loc[:19, "Contract_Month-to-month"] = 1.0
        X.loc[:19, "Contract_Two year"] = 0.0
        X.loc[:19, "Tenure Months"] = 2.0
        X.loc[20:, "Contract_Month-to-month"] = 0.0
        X.loc[20:, "Contract_Two year"] = 1.0
        X.loc[20:, "Tenure Months"] = 60.0
        y = np.array([1] * 20 + [0] * 20)
    else:
        y = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    meta = ModelMetadata(
        algorithm="RandomForestClassifier",
        features=MODEL_FEATURE_NAMES,
        feature_count=len(MODEL_FEATURE_NAMES),
        optimal_threshold=0.12,
        holdout_roc_auc=0.85,
        holdout_pr_auc=0.65,
        holdout_brier_score=0.12,
        cost_savings=21860.0,
    )
    bundle = ModelBundle(model=clf, metadata=meta)
    bundle.save(models_dir)

    return models_dir


@pytest.fixture
def synthetic_simulation_artifacts(tmp_path: Path):
    """Generate synthetic drift_metrics.csv and retrained artifacts in tmp_path."""
    drift_dir = tmp_path / "data" / "drift"
    drift_dir.mkdir(parents=True, exist_ok=True)
    drift_csv = drift_dir / "drift_metrics.csv"

    rows = []
    for week in range(1, 9):
        is_drift = week >= 5
        retrained = week in [3, 7]
        rows.append({
            "week": week,
            "n_labeled": week * 500,
            "psi_mean": 0.25 if is_drift else 0.002,
            "psi_max": 0.45 if is_drift else 0.02,
            "psi_status": "Significant Shift (Drift Detected)" if is_drift else "Stable",
            "never_retrain_auc_reference": 0.835,
            "retrain_model_auc_recent": 0.95 if is_drift else 0.88,
            "retrained": retrained,
            "trigger_reason": "performance decay" if retrained else "monitoring",
        })

    df = pd.DataFrame(rows)
    df.to_csv(drift_csv, index=False)

    retrain_dir = tmp_path / "models" / "retrained"
    week7_dir = retrain_dir / "week_7"
    week7_dir.mkdir(parents=True, exist_ok=True)

    with open(week7_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "week": 7,
            "trigger": "performance decay: recent AUC drop",
            "sample_decay_rate": 0.85,
            "auc_on_recent_live": 0.95,
        }, f)

    # Empty dummy file for joblib
    (week7_dir / "model.joblib").touch()

    return drift_csv, retrain_dir


@pytest.fixture
def synthetic_reference_and_live(tmp_path: Path):
    """Generate synthetic reference and live batch feature files."""
    ref_dir = tmp_path / "data" / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    live_dir = tmp_path / "data" / "live" / "features"
    live_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    ref_df = pd.DataFrame({feature: np.random.randn(50) for feature in MODEL_FEATURE_NAMES})
    ref_file = ref_dir / "X_reference.csv"
    ref_df.to_csv(ref_file, index=False)

    for w in range(1, 9):
        live_df = pd.DataFrame({feature: np.random.randn(50) for feature in MODEL_FEATURE_NAMES})
        live_file = live_dir / f"week_{w}_features.csv"
        live_df.to_csv(live_file, index=False)

    return str(ref_file), str(live_dir)
