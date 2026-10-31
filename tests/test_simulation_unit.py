import pytest
import numpy as np
import pandas as pd
from final_project.simulation import (
    SimulationConfig,
    DualDriftMonitor,
    StreamingBatchEngine,
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
