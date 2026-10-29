import pytest
import pandas as pd
from sklearn.datasets import make_classification
from final_project.models import benchmark_candidates


def test_benchmark_candidates_structure_and_sorting():
    X_mat, y_vec = make_classification(
        n_samples=80,
        n_features=6,
        n_informative=3,
        weights=[0.75, 0.25],
        random_state=42,
    )
    feature_cols = [f"f_{i}" for i in range(6)]
    X = pd.DataFrame(X_mat, columns=feature_cols)
    y = pd.Series(y_vec, name="churn")

    df_results = benchmark_candidates(X, y, cv_folds=2, random_state=42)

    assert isinstance(df_results, pd.DataFrame)
    # 3 models x 2 modes = 6 rows
    assert len(df_results) == 6

    expected_cols = {
        "model",
        "weighting",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "pr_auc_std",
    }
    assert expected_cols.issubset(set(df_results.columns))

    # Verify descending sort on roc_auc_mean
    roc_scores = df_results["roc_auc_mean"].tolist()
    assert roc_scores == sorted(roc_scores, reverse=True)
