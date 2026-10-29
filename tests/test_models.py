import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from final_project.models import train_baseline_model, TrainingResult


def test_train_baseline_model_contract_and_cv():
    # Synthetic dataset with 100 samples, 10 features, 30% positive class
    X_mat, y_vec = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        weights=[0.7, 0.3],
        random_state=42,
    )
    feature_cols = [f"feat_{i}" for i in range(10)]
    X = pd.DataFrame(X_mat, columns=feature_cols)
    y = pd.Series(y_vec, name="target")

    result = train_baseline_model(X, y, cv_folds=3, random_state=42)

    assert isinstance(result, TrainingResult)
    assert isinstance(result.model, RandomForestClassifier)
    assert len(result.cv_fold_scores) == 3
    assert 0.5 <= result.cv_auc_mean <= 1.0
    assert result.cv_auc_std >= 0.0
    assert result.feature_names == feature_cols

    # Verify the model is fitted and can predict probabilities
    probs = result.model.predict_proba(X)
    assert probs.shape == (100, 2)
