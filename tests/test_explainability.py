import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import joblib

from final_project.explainability import (
    group_shap_by_parent_features,
    run_explanation,
    ExplanationResult,
)


def test_group_shap_by_parent_features():
    feature_names = [
        "Tenure Months",
        "Monthly Charges",
        "Contract_Month-to-month",
        "Contract_One year",
        "Contract_Two year",
        "Internet Service_DSL",
        "Internet Service_Fiber optic",
        "Internet Service_No",
    ]
    # Sample SHAP matrix: 2 samples x 8 features
    shap_vals = np.array([
        [0.5, 0.2, 0.3, -0.1, 0.0, 0.1, 0.2, -0.05],
        [0.1, 0.4, 0.1, 0.0, -0.2, -0.1, 0.3, 0.0],
    ])

    grouped_df = group_shap_by_parent_features(shap_vals, feature_names)

    assert isinstance(grouped_df, pd.DataFrame)
    assert set(grouped_df["parent_feature"]).issubset({
        "Tenure Months",
        "Monthly Charges",
        "Contract",
        "Internet Service",
    })
    assert "mean_abs_shap" in grouped_df.columns
    # Check descending sort
    vals = grouped_df["mean_abs_shap"].tolist()
    assert vals == sorted(vals, reverse=True)


def test_run_explanation_generates_all_visualizations(tmp_path):
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    figures_dir = tmp_path / "reports/figures"
    data_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    # Synthetic dataset with dummy column names matching Telco schema
    feature_cols = [
        "Tenure Months",
        "Monthly Charges",
        "Total Charges",
        "Contract_Month-to-month",
        "Contract_One year",
        "Internet Service_DSL",
        "Internet Service_Fiber optic",
    ]
    X_mat, y_vec = make_classification(
        n_samples=50,
        n_features=len(feature_cols),
        n_informative=4,
        weights=[0.7, 0.3],
        random_state=42,
    )
    pd.DataFrame(X_mat, columns=feature_cols).to_csv(data_dir / "X_features.csv", index=False)
    pd.DataFrame({"Churn Value": y_vec}).to_csv(data_dir / "y_target.csv", index=False)

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_mat, y_vec)
    joblib.dump(clf, models_dir / "model.joblib")

    result = run_explanation(
        data_dir=data_dir,
        model_path=models_dir / "model.joblib",
        output_dir=figures_dir,
        max_samples=30,
        random_state=42,
    )

    assert isinstance(result, ExplanationResult)
    assert (figures_dir / "shap_summary_grouped.png").exists()
    assert (figures_dir / "shap_summary_detailed.png").exists()
    assert (figures_dir / "shap_waterfall_churner.png").exists()
    assert (figures_dir / "shap_waterfall_retained.png").exists()
