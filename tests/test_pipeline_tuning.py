import json
import pytest
from pathlib import Path
import pandas as pd
from sklearn.datasets import make_classification
import joblib

from final_project.pipeline import run_training, PipelineRunSummary


def test_run_training_with_tuning(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "models"
    data_dir.mkdir()

    X_mat, y_vec = make_classification(
        n_samples=60,
        n_features=5,
        n_informative=3,
        weights=[0.75, 0.25],
        random_state=42,
    )
    feature_cols = [f"col_{i}" for i in range(5)]
    pd.DataFrame(X_mat, columns=feature_cols).to_csv(data_dir / "X_features.csv", index=False)
    pd.DataFrame({"Churn Value": y_vec}).to_csv(data_dir / "y_target.csv", index=False)

    summary = run_training(
        data_dir=data_dir,
        output_dir=output_dir,
        tune=True,
        n_trials=2,
        cv_folds=2,
        experiment_name="test_pipeline_tune_exp",
        tracking_uri=f"file://{tmp_path / 'mlruns'}",
        random_state=42,
    )

    assert isinstance(summary, PipelineRunSummary)
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "metadata.json").exists()

    with open(output_dir / "metadata.json") as f:
        meta = json.load(f)

    assert meta["tuning_enabled"] is True
    assert "best_params" in meta
    assert meta["n_trials"] == 2
    assert "cv_auc_mean" in meta
