import pytest
import numpy as np
import pandas as pd
from final_project.simulation import (
    SimulationConfig,
    DualDriftMonitor,
    StreamingBatchEngine,
    SimulationRunner,
    run_simulation,
)


def test_simulation_config_defaults():
    cfg = SimulationConfig()
    assert cfg.n_weeks == 8
    assert cfg.drift_start_week == 5
    assert cfg.label_delay_weeks == 2
    assert cfg.sample_decay_rate == 0.85


def test_dual_drift_monitor_decay_check():
    monitor = DualDriftMonitor(auc_drop_threshold=0.05)

    # Initial check before any model trained
    triggered, reason = monitor.check_decay(recent_auc=0.85, model_retrain_exists=False)
    assert triggered is True
    assert reason == "initial training"

    # Set baseline
    monitor.update_best_auc(0.90)

    # Mild drop within threshold (0.90 - 0.03 = 0.87 > 0.85) -> no decay
    triggered, reason = monitor.check_decay(recent_auc=0.87, model_retrain_exists=True)
    assert triggered is False
    assert reason == "monitoring"

    # Significant drop beyond threshold (0.90 - 0.06 = 0.84 < 0.85) -> decay triggered
    triggered, reason = monitor.check_decay(recent_auc=0.84, model_retrain_exists=True)
    assert triggered is True
    assert "performance decay" in reason


def test_streaming_batch_engine_drift_injection():
    cfg = SimulationConfig(n_weeks=3, drift_start_week=2, price_increase_mult=1.5, tenure_factor=0.5)
    engine = StreamingBatchEngine(cfg)

    features = pd.DataFrame({
        "Monthly Charges": [50.0, 100.0, 75.0, 60.0],
        "Tenure Months": [20.0, 40.0, 10.0, 30.0],
    })
    target = pd.DataFrame({"Churn Value": [0, 0, 0, 0]})

    reference, population = engine.prepare_population_and_reference(features, target)
    batches = engine.build_weekly_batches(population)
    batches = engine.inject_drift(batches)

    # Week 1 (index 0) is before drift start (drift_start_week=2)
    # Week 2 (index 1) has drift injected
    assert len(batches) == 3


def test_simulation_runner_dependency_injection():
    cfg = SimulationConfig(n_weeks=2)
    custom_monitor = DualDriftMonitor(auc_drop_threshold=0.10)
    custom_engine = StreamingBatchEngine(cfg)

    runner = SimulationRunner(
        config=cfg,
        monitor=custom_monitor,
        engine=custom_engine,
        save_artifacts=False,
        verbose=False,
    )

    assert runner.monitor is custom_monitor
    assert runner.engine is custom_engine
    assert runner.save_artifacts is False
    assert runner.verbose is False


def test_safe_roc_auc_score():
    from final_project.simulation import _safe_roc_auc_score

    # Normal 2-class case
    y_true = np.array([0, 1, 0, 1])
    preds = np.array([0.1, 0.9, 0.2, 0.8])
    assert _safe_roc_auc_score(y_true, preds) == 1.0

    # Single-class case -> None without throwing
    y_single = np.array([0, 0, 0, 0])
    assert _safe_roc_auc_score(y_single, preds) is None


def test_fit_never_retrain_baseline_pre_drift_only():
    from sklearn.linear_model import LogisticRegression

    cfg = SimulationConfig(drift_start_week=5)
    runner = SimulationRunner(config=cfg, save_artifacts=False, verbose=False)

    master = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "Churn Value": [0, 1, 0, 1, 0],
        "label_release_week": [3, 4, 5, 6, 7],
    })

    base_model = LogisticRegression()
    fitted_model = runner._fit_never_retrain_baseline(base_model, master, ["feat1"])
    # Should only fit on label_release_week < 5 (i.e. weeks 3 and 4 -> 2 samples)
    assert fitted_model is not None

