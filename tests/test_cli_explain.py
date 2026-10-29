import pytest
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import joblib

from final_project.cli import main


def test_cli_explain_command(tmp_path, capsys):
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    figures_dir = tmp_path / "reports/figures"
    data_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    feature_cols = [
        "Tenure Months",
        "Monthly Charges",
        "Contract_Month-to-month",
        "Contract_Two year",
        "Internet Service_DSL",
        "Internet Service_Fiber optic",
    ]
    X_mat, y_vec = make_classification(
        n_samples=40,
        n_features=len(feature_cols),
        n_informative=3,
        weights=[0.7, 0.3],
        random_state=42,
    )
    pd.DataFrame(X_mat, columns=feature_cols).to_csv(data_dir / "X_features.csv", index=False)
    pd.DataFrame({"Churn Value": y_vec}).to_csv(data_dir / "y_target.csv", index=False)

    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X_mat, y_vec)
    joblib.dump(clf, models_dir / "model.joblib")

    exit_code = main([
        "explain",
        "--data-dir", str(data_dir),
        "--model-path", str(models_dir / "model.joblib"),
        "--output-dir", str(figures_dir),
        "--max-samples", "25",
    ])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Model Explainability Completed Successfully" in captured.out
    assert "Top Parent Feature Importances" in captured.out
    assert (figures_dir / "shap_summary_grouped.png").exists()
    assert (figures_dir / "shap_summary_detailed.png").exists()
