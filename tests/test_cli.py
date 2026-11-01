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
    assert "--server.headless" in called_cmds[0]
    assert "true" in called_cmds[0]

    # Test --no-headless toggle
    exit_code2 = main(["dashboard", "--no-headless"])
    assert exit_code2 == 0
    assert len(called_cmds) == 2
    assert "--server.headless" in called_cmds[1]
    assert "false" in called_cmds[1]


def test_cli_simulate_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["simulate", "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--n-weeks" in captured.out
    assert "--no-save" in captured.out


def test_cli_simulate_command_mock(monkeypatch, capsys):
    from final_project import cli
    import pandas as pd

    class FakeResults:
        drift_metrics = pd.DataFrame([{"week": 1}])
        retrained_weeks = [3, 7]
        final_retrain_auc_recent = 0.88

    def fake_run_simulation(*args, **kwargs):
        return FakeResults()

    monkeypatch.setattr(cli, "run_simulation", fake_run_simulation)
    exit_code = main(["simulate", "--n-weeks", "4", "--no-save"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Starting weekly streaming simulation (4 weeks)" in captured.out
    assert "Simulation Completed Successfully" in captured.out


def test_cli_serve_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--port" in captured.out
    assert "--host" in captured.out
    assert "--reload" in captured.out


def test_cli_serve_launch_mock(monkeypatch):
    import uvicorn

    called_kwargs = {}

    def fake_run(app_target, **kwargs):
        called_kwargs["app"] = app_target
        called_kwargs.update(kwargs)
        return None

    monkeypatch.setattr(uvicorn, "run", fake_run)
    exit_code = main(["serve", "--port", "8080", "--host", "127.0.0.1", "--reload"])
    assert exit_code == 0
    assert called_kwargs["port"] == 8080
    assert called_kwargs["host"] == "127.0.0.1"
    assert called_kwargs["reload"] is True



