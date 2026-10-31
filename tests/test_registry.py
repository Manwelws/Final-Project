import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from final_project.registry import ModelMetadata, ModelBundle, RiskPrediction


def test_model_metadata_roundtrip(tmp_path):
    meta = ModelMetadata(
        timestamp="2026-10-31T12:00:00Z",
        algorithm="RandomForestClassifier",
        balanced=True,
        optimal_threshold=0.12,
        cost_savings=21860.0,
        cost_at_optimal=14900.0,
        cost_at_default=36760.0,
        holdout_roc_auc=0.84,
        holdout_pr_auc=0.63,
        holdout_brier_score=0.14,
        features=["f1", "f2"],
        random_state=42,
    )

    meta_file = tmp_path / "metadata.json"
    meta.save(meta_file)

    loaded = ModelMetadata.load(meta_file)
    assert loaded.algorithm == "RandomForestClassifier"
    assert loaded.balanced is True
    assert loaded.optimal_threshold == 0.12
    assert loaded.features == ["f1", "f2"]


def test_model_bundle_predict_risk_and_tiers(tmp_path):
    # Create simple trained model
    X_train = np.array([[1.0, 2.0], [2.0, 1.0], [0.5, 0.5], [3.0, 3.0]])
    y_train = np.array([0, 1, 0, 1])
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X_train, y_train)

    meta = ModelMetadata(
        algorithm="RandomForestClassifier",
        features=["f1", "f2"],
        optimal_threshold=0.25,
    )

    bundle = ModelBundle(model=clf, metadata=meta)

    # Save and reload bundle
    bundle.save(tmp_path)
    loaded_bundle = ModelBundle.load(tmp_path)

    # Predict risk on DataFrame
    X_test = pd.DataFrame([[1.0, 2.0], [0.5, 0.5]], columns=["f1", "f2"])
    pred = loaded_bundle.predict_risk(X_test)

    assert isinstance(pred, RiskPrediction)
    assert len(pred.probabilities) == 2
    assert len(pred.predictions) == 2
    assert len(pred.risk_tiers) == 2
    assert pred.threshold_applied == 0.25
    assert set(pred.risk_tiers).issubset({"Low", "Medium", "High"})


def test_model_bundle_missing_feature_validation():
    clf = RandomForestClassifier()
    meta = ModelMetadata(features=["f1", "f2", "f3"])
    bundle = ModelBundle(model=clf, metadata=meta)

    df_bad = pd.DataFrame([[1.0, 2.0]], columns=["f1", "f2"])
    with pytest.raises(ValueError, match="missing required features"):
        bundle.predict_risk(df_bad)
