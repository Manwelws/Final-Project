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


def test_get_candidate_model_invalid_name():
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        get_candidate_model("unknown_model")
