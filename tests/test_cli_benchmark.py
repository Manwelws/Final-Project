import pytest
import pandas as pd
from sklearn.datasets import make_classification
from final_project.cli import main


def test_cli_benchmark_command(tmp_path, capsys):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    X_mat, y_vec = make_classification(
        n_samples=60,
        n_features=4,
        n_informative=2,
        weights=[0.75, 0.25],
        random_state=42,
    )
    feature_cols = [f"feat_{i}" for i in range(4)]
    pd.DataFrame(X_mat, columns=feature_cols).to_csv(data_dir / "X_features.csv", index=False)
    pd.DataFrame({"Churn Value": y_vec}).to_csv(data_dir / "y_target.csv", index=False)

    exit_code = main([
        "benchmark",
        "--data-dir", str(data_dir),
        "--cv-folds", "2",
    ])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Candidate Benchmark Results" in captured.out
    assert "Random Forest" in captured.out
    assert "LightGBM" in captured.out
    assert "XGBoost" in captured.out
    assert "Unweighted" in captured.out
    assert "Balanced" in captured.out


def test_cli_train_with_algorithm_and_balanced(tmp_path, capsys):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "models"
    data_dir.mkdir()

    X_mat, y_vec = make_classification(
        n_samples=40,
        n_features=4,
        n_informative=2,
        weights=[0.7, 0.3],
        random_state=42,
    )
    feature_cols = [f"f_{i}" for i in range(4)]
    pd.DataFrame(X_mat, columns=feature_cols).to_csv(data_dir / "X_features.csv", index=False)
    pd.DataFrame({"Churn Value": y_vec}).to_csv(data_dir / "y_target.csv", index=False)

    exit_code = main([
        "train",
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
        "--algorithm", "lightgbm",
        "--balanced",
        "--cv-folds", "2",
    ])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "LGBMClassifier" in captured.out or "LightGBM" in captured.out
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "metadata.json").exists()
