import pytest
import numpy as np
import pandas as pd
from final_project.domain import CostMatrix, DriftPolicy, RetrainingPolicy


def test_cost_matrix_defaults_and_loss_calculation():
    matrix = CostMatrix()
    assert matrix.cost_fn == 200.0
    assert matrix.cost_fp == 20.0

    # 3 false negatives ($200 each = $600) + 5 false positives ($20 each = $100) -> $700
    loss = matrix.compute_loss(fn=3, fp=5)
    assert loss == 700.0
    assert matrix.total_cost(fn=3, fp=5) == 700.0


def test_cost_matrix_financial_savings_and_roi():
    matrix = CostMatrix()
    default_loss = 36740.0
    optimal_loss = 14880.0

    savings = matrix.compute_net_savings(default_loss, optimal_loss)
    assert savings == 21860.0

    pct = matrix.compute_loss_reduction_pct(default_loss, optimal_loss)
    assert round(pct, 1) == 59.5

    # Zero or negative default loss boundary
    assert matrix.compute_loss_reduction_pct(0.0, 100.0) == 0.0

    # Campaign ROI
    roi = matrix.compute_campaign_roi(gross_revenue_saved=400000.0, campaign_cost=20000.0)
    assert roi == 20.0
    assert matrix.compute_campaign_roi(100.0, 0.0) == 0.0


def test_drift_policy_classification():
    policy = DriftPolicy()
    assert policy.classify_psi(0.04) == "Stable"
    assert policy.classify_psi(0.15) == "Moderate Shift"
    assert policy.classify_psi(0.35) == "Significant Shift (Drift Detected)"

    assert policy.is_drift_detected(0.15) is False
    assert policy.is_drift_detected(0.25) is True
    assert policy.is_drift_detected(0.40) is True


def test_retraining_policy_defaults():
    policy = RetrainingPolicy()
    assert policy.auc_drop_threshold == 0.05
    assert policy.min_labels_for_retrain == 200
    assert policy.recent_window_weeks == 2
    assert policy.sample_decay_rate == 0.85


def test_retraining_policy_decision_rules():
    policy = RetrainingPolicy(auc_drop_threshold=0.05, min_labels_for_retrain=200)

    # 1. Uninitialized model -> initial training
    trigger, reason = policy.should_retrain(recent_auc=0.85, best_recent_auc=None, model_exists=False)
    assert trigger is True
    assert reason == "initial training"

    # 2. Insufficient labels -> do not retrain
    trigger, reason = policy.should_retrain(
        recent_auc=0.75, best_recent_auc=0.88, model_exists=True, n_labeled=150
    )
    assert trigger is False
    assert "insufficient labeled samples" in reason

    # 3. Stable performance within threshold (0.88 - 0.03 = 0.85 >= 0.83)
    trigger, reason = policy.should_retrain(
        recent_auc=0.85, best_recent_auc=0.88, model_exists=True, n_labeled=300
    )
    assert trigger is False
    assert reason == "monitoring"

    # 4. Decayed performance (0.88 - 0.06 = 0.82 < 0.83) -> trigger
    trigger, reason = policy.should_retrain(
        recent_auc=0.82, best_recent_auc=0.88, model_exists=True, n_labeled=300
    )
    assert trigger is True
    assert "performance decay" in reason


def test_retraining_policy_sample_weights():
    policy = RetrainingPolicy(sample_decay_rate=0.85)

    current_week = 5
    available_weeks = [5, 4, 3, 2, 1]
    weights = policy.compute_sample_weights(current_week, available_weeks)

    assert len(weights) == 5
    # Week 5: age 0 -> 0.85^0 = 1.0
    assert weights[0] == 1.0
    # Week 4: age 1 -> 0.85^1 = 0.85
    assert round(weights[1], 4) == 0.85
    # Week 3: age 2 -> 0.85^2 = 0.7225
    assert round(weights[2], 4) == 0.7225
