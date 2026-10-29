import pytest
import pandas as pd
from sklearn.datasets import make_classification
from final_project.models import (
    get_candidate_model,
    evaluate_model_cv,
    ModelCVMetrics,
)


def test_evaluate_model_cv_metrics_computation():
    X_mat, y_vec = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=4,
        weights=[0.7, 0.3],
        random_state=42,
    )
    feature_cols = [f"feat_{i}" for i in range(8)]
    X = pd.DataFrame(X_mat, columns=feature_cols)
    y = pd.Series(y_vec, name="target")

    model = get_candidate_model("lightgbm", balanced=False, random_state=42)
    metrics = evaluate_model_cv(
        model=model,
        X=X,
        y=y,
        cv_folds=3,
        model_name="LightGBM",
        balanced=False,
        random_state=42,
    )

    assert isinstance(metrics, ModelCVMetrics)
    assert metrics.model_name == "LightGBM"
    assert metrics.balanced is False
    assert 0.5 <= metrics.roc_auc_mean <= 1.0
    assert 0.2 <= metrics.pr_auc_mean <= 1.0
    assert metrics.roc_auc_std >= 0.0
    assert metrics.pr_auc_std >= 0.0
