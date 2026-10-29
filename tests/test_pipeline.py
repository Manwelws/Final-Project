import json
import pytest
from pathlib import Path
import joblib
import pandas as pd
from sklearn.datasets import make_classification
from final_project.pipeline import run_training, PipelineRunSummary


def test_run_training_mock_creates_artifacts_and_metadata(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "models"
    data_dir.mkdir()

    # Create mock dataset
    X_mat, y_vec = make_classification(
        n_samples=60,
        n_features=5,
        n_informative=3,
        weights=[0.7, 0.3],
        random_state=42,
    )
    feature_cols = [f"col_{i}" for i in range(5)]
    pd.DataFrame(X_mat, columns=feature_cols).to_csv(data_dir / "X_features.csv", index=False)
    pd.DataFrame({"Churn Value": y_vec}).to_csv(data_dir / "y_target.csv", index=False)

    summary = run_training(
        data_dir=data_dir,
        output_dir=output_dir,
        cv_folds=3,
        random_state=42,
    )

    assert isinstance(summary, PipelineRunSummary)
    assert output_dir.exists()
    
    # Check model artifact exists and can be loaded
    model_path = output_dir / "model.joblib"
    assert model_path.exists()
    loaded_model = joblib.load(model_path)
    assert hasattr(loaded_model, "predict")

    # Check metadata.json exists and contains required fields
    meta_path = output_dir / "metadata.json"
    assert meta_path.exists()
    with open(meta_path) as f:
        meta = json.load(f)

    assert meta["algorithm"] == "RandomForestClassifier"
    assert meta["feature_count"] == 5
    assert meta["features"] == feature_cols
    assert "timestamp" in meta
    assert "cv_auc_mean" in meta
    assert len(meta["cv_fold_scores"]) == 3


def test_package_level_run_training_export():
    import final_project
    assert callable(final_project.run_training)
