import pytest
from final_project.domain import CostMatrix, DriftPolicy, RetrainingPolicy


def test_cost_matrix_defaults_and_loss_calculation():
    matrix = CostMatrix()
    assert matrix.cost_fn == 200.0
    assert matrix.cost_fp == 20.0

    # 3 false negatives ($200 each = $600) + 5 false positives ($20 each = $100) -> $700
    loss = matrix.compute_loss(fn=3, fp=5)
    assert loss == 700.0


def test_drift_policy_classification():
    policy = DriftPolicy()
    assert policy.classify_psi(0.04) == "Stable"
    assert policy.classify_psi(0.15) == "Moderate Shift"
    assert policy.classify_psi(0.35) == "Significant Shift (Drift Detected)"


def test_retraining_policy_defaults():
    policy = RetrainingPolicy()
    assert policy.auc_drop_threshold == 0.05
    assert policy.min_labels_for_retrain == 200
    assert policy.recent_window_weeks == 2
    assert policy.sample_decay_rate == 0.85
