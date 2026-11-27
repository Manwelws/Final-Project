import json
import pytest
from pathlib import Path
import joblib
from final_project.pipeline import run_training, PipelineRunSummary
from final_project.registry import ModelBundle, ModelMetadata, compute_sha256


def test_run_training_mock_creates_artifacts_and_metadata(tmp_path, classification_data_factory):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "models"

    # Create mock dataset
    X_df, _ = classification_data_factory(
        data_dir,
        n_samples=60,
        n_features=5,
        n_informative=3,
        weights=[0.7, 0.3],
        random_state=42,
        col_prefix="col_",
    )
    feature_cols = list(X_df.columns)

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


def test_run_training_persists_model_sha256_and_passes_bundle_load(tmp_path, classification_data_factory):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "models"

    classification_data_factory(
        data_dir,
        n_samples=60,
        n_features=5,
        n_informative=3,
        weights=[0.7, 0.3],
        random_state=42,
        col_prefix="col_",
    )

    run_training(
        data_dir=data_dir,
        output_dir=output_dir,
        cv_folds=3,
        random_state=42,
    )

    model_path = output_dir / "model.joblib"
    meta_path = output_dir / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    expected_sha = compute_sha256(model_path)
    assert meta.get("model_sha256")
    assert len(meta["model_sha256"]) == 64
    assert meta["model_sha256"] == expected_sha

    bundle = ModelBundle.load(output_dir)
    assert bundle.metadata.model_sha256 == expected_sha


@pytest.mark.parametrize(
    "val_size, expected_train, expected_pre_refit, expected_val",
    [
        (0.10, 48, 42, 6),
        (0.0, 48, 48, 0),
    ],
)
def test_run_training_reconciles_train_samples_and_split_counts(
    tmp_path, classification_data_factory, val_size, expected_train, expected_pre_refit, expected_val
):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "models"

    classification_data_factory(
        data_dir,
        n_samples=60,
        n_features=5,
        n_informative=3,
        weights=[0.7, 0.3],
        random_state=42,
        col_prefix="col_",
    )

    summary = run_training(
        data_dir=data_dir,
        output_dir=output_dir,
        test_size=0.20,
        val_size=val_size,
        cv_folds=3,
        random_state=42,
    )

    meta = ModelMetadata.load(output_dir / "metadata.json")
    assert summary.train_samples == expected_train
    assert summary.pre_refit_train_samples == expected_pre_refit
    assert summary.val_samples == expected_val
    assert meta.train_samples == summary.train_samples
    assert meta.pre_refit_train_samples == summary.pre_refit_train_samples
    assert meta.val_samples == summary.val_samples


def test_run_training_knn_ignores_balanced_flag_in_summary_and_metadata(
    tmp_path, classification_data_factory
):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "models"

    classification_data_factory(
        data_dir,
        n_samples=60,
        n_features=5,
        n_informative=3,
        weights=[0.7, 0.3],
        random_state=42,
        col_prefix="col_",
    )

    summary = run_training(
        data_dir=data_dir,
        output_dir=output_dir,
        algorithm="knn",
        balanced=True,
        cv_folds=3,
        random_state=42,
    )

    meta = ModelMetadata.load(output_dir / "metadata.json")
    assert summary.balanced is False
    assert meta.balanced is False
