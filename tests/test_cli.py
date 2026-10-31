import pytest
import pandas as pd
from sklearn.datasets import make_classification
from final_project.cli import main


def test_cli_train_command(tmp_path, capsys):
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
        "--cv-folds", "2",
    ])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Training completed successfully" in captured.out
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "metadata.json").exists()


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_cli_dashboard_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["dashboard", "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--port" in captured.out
    assert "--host" in captured.out


def test_cli_dashboard_launch_mock(monkeypatch):
    import subprocess

    called_cmds = []

    def fake_call(cmd):
        called_cmds.append(cmd)
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)
    exit_code = main(["dashboard", "--port", "9000", "--host", "127.0.0.1", "--headless"])
    assert exit_code == 0
    assert len(called_cmds) == 1
    assert "--server.port" in called_cmds[0]
    assert "9000" in called_cmds[0]
    assert "--server.address" in called_cmds[0]
    assert "127.0.0.1" in called_cmds[0]

