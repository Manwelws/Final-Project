import ast
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pytest
from fastapi.testclient import TestClient

from final_project.registry import ModelMetadata, ModelBundle
from final_project.serving.api import app, lifespan


def test_zero_bare_print_in_src():
    """Verify that no bare print() statements exist under src/final_project."""
    src_dir = Path("src/final_project")
    print_calls = []

    for py_file in src_dir.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "print":
                    print_calls.append(f"{py_file}:{node.lineno}")

    assert not print_calls, f"Bare print() calls found in src/: {print_calls}"


def test_metadata_corrupted_json_handled_gracefully(tmp_path, caplog):
    """Corrupted metadata.json is caught and logged as a warning without crashing."""
    corrupted_file = tmp_path / "metadata.json"
    corrupted_file.write_text("{this is not valid json", encoding="utf-8")

    # Create dummy model
    import joblib
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit([[0], [1]], [0, 1])
    joblib.dump(clf, tmp_path / "model.joblib")

    with caplog.at_level(logging.WARNING):
        bundle = ModelBundle.load(tmp_path)
    assert bundle.metadata is not None
    assert bundle.metadata.algorithm == "LogisticRegression"
    assert any("Failed to load model metadata" in r.message for r in caplog.records)


def test_startup_load_failure_logs_error(tmp_path, caplog, monkeypatch):
    """Missing model directory during startup logs ERROR level."""
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()
    monkeypatch.setenv("CHURN_MODELS_DIR", str(empty_dir))

    # Reset bundle
    app.state.bundle = None

    with caplog.at_level(logging.ERROR):
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["algorithm"] == "unloaded"

    assert any(r.levelno == logging.ERROR and "Failed to load ModelBundle on startup" in r.message for r in caplog.records)


def test_reporting_figures_are_properly_closed(tmp_path):
    """Reporting plot generation leaves zero open matplotlib figures."""
    from final_project.reporting import render_evaluation_plots
    from final_project.evaluation import EvaluationReport

    # Close any existing figures before test
    plt.close("all")
    assert len(plt.get_fignums()) == 0

    y_true = [0, 1, 0, 1, 0]
    y_probs = [0.1, 0.9, 0.2, 0.8, 0.3]
    rep = EvaluationReport.compute(y_true=y_true, y_probs=y_probs)
    fig_dir = tmp_path / "figures"
    plots = rep.render_plots(figures_dir=fig_dir)

    assert len(plots) == 5
    assert len(plt.get_fignums()) == 0
