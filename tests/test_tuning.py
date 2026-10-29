import pytest
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification
import mlflow
import optuna

from final_project.tuning import run_tuning_study, TuningResult


def test_run_tuning_study_fast_with_mlflow_logging(tmp_path):
    X_mat, y_vec = make_classification(
        n_samples=80,
        n_features=6,
        n_informative=3,
        weights=[0.75, 0.25],
        random_state=42,
    )
    feature_cols = [f"f_{i}" for i in range(6)]
    X = pd.DataFrame(X_mat, columns=feature_cols)
    y = pd.Series(y_vec, name="target")

    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    exp_name = "test_optuna_mlflow_experiment"

    result = run_tuning_study(
        X_train=X,
        y_train=y,
        n_trials=3,
        cv_folds=2,
        experiment_name=exp_name,
        tracking_uri=tracking_uri,
        random_state=42,
    )

    assert isinstance(result, TuningResult)
    assert result.best_algorithm in ["random_forest", "lightgbm", "xgboost"]
    assert isinstance(result.best_params, dict)
    assert 0.5 <= result.best_roc_auc <= 1.0
    assert hasattr(result.best_model, "predict_proba")

    # Verify model is fitted and can predict on X
    preds = result.best_model.predict_proba(X)
    assert preds.shape == (80, 2)

    # Verify MLflow recorded the runs
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(exp_name)
    assert experiment is not None
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) >= 3
    for run in runs:
        assert "algorithm" in run.data.params
        assert "cv_roc_auc" in run.data.metrics
