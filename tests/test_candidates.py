import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from final_project.models import get_candidate_model, compute_scale_pos_weight


def test_compute_scale_pos_weight():
    # 70 negative (0), 30 positive (1) -> 70 / 30 = 2.3333
    y = pd.Series([0] * 70 + [1] * 30)
    weight = compute_scale_pos_weight(y)
    assert pytest.approx(weight, rel=1e-3) == 70 / 30


def test_get_candidate_model_instantiations():
    # Random Forest
    rf_unweighted = get_candidate_model("random_forest", balanced=False)
    rf_balanced = get_candidate_model("random_forest", balanced=True)
    assert isinstance(rf_unweighted, RandomForestClassifier)
    assert rf_unweighted.class_weight is None
    assert rf_balanced.class_weight == "balanced"

    # LightGBM
    lgb_unweighted = get_candidate_model("lightgbm", balanced=False)
    lgb_balanced = get_candidate_model("lightgbm", balanced=True)
    assert isinstance(lgb_unweighted, LGBMClassifier)
    assert lgb_unweighted.class_weight is None
    assert lgb_balanced.class_weight == "balanced"

    # XGBoost
    xgb_unweighted = get_candidate_model("xgboost", balanced=False)
    xgb_balanced = get_candidate_model("xgboost", balanced=True, scale_pos_weight=2.5)
    assert isinstance(xgb_unweighted, XGBClassifier)
    assert xgb_unweighted.scale_pos_weight == 1.0
    assert xgb_balanced.scale_pos_weight == 2.5

    # Logistic Regression (Scaled)
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import BaggingClassifier, VotingClassifier, StackingClassifier

    lr_unweighted = get_candidate_model("logistic_regression", balanced=False)
    lr_balanced = get_candidate_model("logistic_regression", balanced=True)
    assert isinstance(lr_unweighted, Pipeline)
    assert isinstance(lr_balanced.named_steps["classifier"], LogisticRegression)
    assert lr_unweighted.named_steps["classifier"].class_weight is None
    assert lr_balanced.named_steps["classifier"].class_weight == "balanced"

    # Bagged Logistic Regression
    bagged_lr = get_candidate_model("bagged_logistic", balanced=True, n_estimators=20)
    assert isinstance(bagged_lr, BaggingClassifier)
    assert bagged_lr.n_estimators == 20

    # Soft Voting Ensemble
    voting = get_candidate_model("voting_ensemble", balanced=True)
    assert isinstance(voting, VotingClassifier)
    assert voting.voting == "soft"
    assert len(voting.estimators) == 3

    # Stacking Ensemble
    stacking = get_candidate_model("stacking_ensemble", balanced=True)
    assert isinstance(stacking, StackingClassifier)
    assert len(stacking.estimators) == 3
    assert isinstance(stacking.final_estimator, LogisticRegression)


def test_get_candidate_model_knn():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    knn_model = get_candidate_model("knn")
    assert isinstance(knn_model, Pipeline)
    assert isinstance(knn_model.named_steps["scaler"], StandardScaler)
    assert isinstance(knn_model.named_steps["classifier"], KNeighborsClassifier)


def test_benchmark_candidates_knn_unweighted_only():
    from sklearn.datasets import make_classification
    from final_project.models import benchmark_candidates

    X_mat, y_vec = make_classification(
        n_samples=60,
        n_features=4,
        n_informative=2,
        weights=[0.75, 0.25],
        random_state=42,
    )
    X = pd.DataFrame(X_mat, columns=[f"f_{i}" for i in range(4)])
    y = pd.Series(y_vec, name="churn")

    df = benchmark_candidates(X, y, algorithms=["knn"], cv_folds=2, random_state=42)
    assert len(df) == 1
    assert df["model"].iloc[0] == "K-Nearest Neighbors (Scaled)"
    assert df["weighting"].iloc[0] == "Unweighted"


def test_supports_class_weighting():
    from final_project.models import supports_class_weighting

    assert supports_class_weighting("knn") is False
    assert supports_class_weighting("kneighbors") is False
    assert supports_class_weighting("KNN") is False
    assert supports_class_weighting("random_forest") is True
    assert supports_class_weighting("lightgbm") is True
    assert supports_class_weighting("xgboost") is True


def test_get_candidate_model_invalid_name():
    with pytest.raises(ValueError, match="Unsupported algorithm") as exc_info:
        get_candidate_model("unknown_model")
    assert "'knn'" in str(exc_info.value)


def test_algorithm_specs_cli_choices_and_benchmark_all():
    from sklearn.base import BaseEstimator
    from final_project.models import (
        BENCHMARK_ALL_ALGORITHMS,
        CANDIDATE_ALGORITHM_SPECS,
        CLI_ALGORITHM_CHOICES,
        normalize_algorithm_name,
        supports_class_weighting,
    )

    canonical_names = [spec.canonical_name for spec in CANDIDATE_ALGORITHM_SPECS]
    assert BENCHMARK_ALL_ALGORITHMS == canonical_names
    assert "knn" in BENCHMARK_ALL_ALGORITHMS

    expected_cli_entries = {
        "random_forest",
        "lightgbm",
        "xgboost",
        "rf",
        "lgb",
        "xgb",
        "logistic_regression",
        "logistic",
        "lr",
        "bagged_logistic",
        "bagged_lr",
        "mlp",
        "knn",
        "kneighbors",
        "voting_ensemble",
        "stacking_ensemble",
    }
    assert expected_cli_entries.issubset(set(CLI_ALGORITHM_CHOICES))

    for choice in CLI_ALGORITHM_CHOICES:
        model = get_candidate_model(choice)
        assert isinstance(model, BaseEstimator)

    for algo in BENCHMARK_ALL_ALGORITHMS:
        model = get_candidate_model(algo)
        assert isinstance(model, BaseEstimator)
        expected_weighted = algo != "knn"
        assert supports_class_weighting(algo) is expected_weighted

    assert supports_class_weighting("knn") is False
    assert supports_class_weighting("kneighbors") is False
    assert normalize_algorithm_name("rf") == "random_forest"
    assert normalize_algorithm_name("lgbm") == "lightgbm"
    assert normalize_algorithm_name("lgb") == "lightgbm"
    assert normalize_algorithm_name("xgb") == "xgboost"
    assert normalize_algorithm_name("lr") == "logistic_regression"
    assert normalize_algorithm_name("logistic") == "logistic_regression"
    assert normalize_algorithm_name("bagged_lr") == "bagged_logistic"
    assert normalize_algorithm_name("neural_network") == "mlp"
    assert normalize_algorithm_name("mlp_classifier") == "mlp"
    assert normalize_algorithm_name("kneighbors") == "knn"
    assert normalize_algorithm_name("voting") == "voting_ensemble"
    assert normalize_algorithm_name("stacking") == "stacking_ensemble"
