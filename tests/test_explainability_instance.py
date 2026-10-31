import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from final_project.explainability import (
    explain_customer_instance,
    CustomerExplanation,
)


def test_explain_customer_instance_signed_parent_grouping():
    # Setup dataset with dummy encoded features
    feature_cols = [
        "Tenure Months",
        "Monthly Charges",
        "Contract_Month-to-month",
        "Contract_One year",
        "Contract_Two year",
        "Internet Service_DSL",
        "Internet Service_Fiber optic",
        "Internet Service_No",
    ]
    np.random.seed(42)
    X = np.random.randn(50, len(feature_cols))
    y = np.random.choice([0, 1], size=50)

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    customer_df = pd.DataFrame([X[0]], columns=feature_cols)

    explanation = explain_customer_instance(clf, customer_df)

    assert isinstance(explanation, CustomerExplanation)
    assert 0.0 <= explanation.churn_probability <= 1.0
    assert isinstance(explanation.base_value, float)
    assert isinstance(explanation.risk_drivers, list)
    assert isinstance(explanation.protective_factors, list)

    # Check that risk drivers are positive and protective factors are negative
    for parent, val in explanation.risk_drivers:
        assert val > 0, f"Risk driver {parent} should have positive SHAP impact"
    for parent, val in explanation.protective_factors:
        assert val < 0, f"Protective factor {parent} should have negative SHAP impact"

    # Verify all feature names appear in raw_shap_values
    assert set(explanation.raw_shap_values.keys()) == set(feature_cols)


def test_explain_customer_instance_series_input():
    feature_cols = ["f1", "f2"]
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 0.5]])
    y = np.array([0, 1, 0])
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)

    series = pd.Series([1.5, 2.5], index=feature_cols)
    explanation = explain_customer_instance(clf, series)

    assert isinstance(explanation, CustomerExplanation)
    assert len(explanation.raw_shap_values) == 2
