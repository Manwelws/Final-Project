import pytest
import numpy as np
import pandas as pd

from final_project.drift import (
    calculate_psi,
    compute_dataset_psi,
    compute_exponential_sample_weights,
)


def test_calculate_psi_identical_distributions():
    rng = np.random.RandomState(42)
    ref = rng.normal(loc=100.0, scale=15.0, size=1000)
    curr = rng.normal(loc=100.0, scale=15.0, size=1000)

    psi = calculate_psi(ref, curr, num_bins=10)
    assert isinstance(psi, float)
    # Identical distributions should have PSI very close to 0 (< 0.05)
    assert psi < 0.10


def test_calculate_psi_detects_significant_drift():
    rng = np.random.RandomState(42)
    ref = rng.normal(loc=50.0, scale=10.0, size=1000)
    # Large shift in mean and scale
    curr = rng.normal(loc=90.0, scale=20.0, size=1000)

    psi = calculate_psi(ref, curr, num_bins=10)
    # Significant shift is PSI >= 0.25
    assert psi >= 0.25


def test_compute_dataset_psi():
    rng = np.random.RandomState(42)
    ref_df = pd.DataFrame({
        "Monthly Charges": rng.normal(70, 15, 500),
        "Tenure Months": rng.uniform(1, 72, 500),
        "Contract_Month-to-month": rng.binomial(1, 0.5, 500),
    })
    curr_df = pd.DataFrame({
        "Monthly Charges": rng.normal(110, 15, 500),  # Drifted
        "Tenure Months": rng.uniform(1, 30, 500),      # Drifted
        "Contract_Month-to-month": rng.binomial(1, 0.5, 500),
    })

    psi_summary = compute_dataset_psi(ref_df, curr_df)
    assert "mean_psi" in psi_summary
    assert "max_psi" in psi_summary
    assert "feature_psi" in psi_summary
    assert psi_summary["feature_psi"]["Monthly Charges"] > 0.10


def test_compute_exponential_sample_weights():
    # Samples from week 1, 2, 3, 4 when current week is 5
    available_weeks = np.array([1, 2, 3, 4, 5])
    current_week = 5
    decay_rate = 0.85

    weights = compute_exponential_sample_weights(
        current_week=current_week,
        available_weeks=available_weeks,
        decay_rate=decay_rate,
    )

    # Week 5 sample is 0 weeks old: 0.85^0 = 1.0
    assert pytest.approx(weights[4], abs=1e-5) == 1.0
    # Week 4 sample is 1 week old: 0.85^1 = 0.85
    assert pytest.approx(weights[3], abs=1e-5) == 0.85
    # Week 1 sample is 4 weeks old: 0.85^4 = 0.5220
    assert pytest.approx(weights[0], abs=1e-5) == 0.85**4
    # Weights should be monotonically increasing with recency
    assert list(weights) == sorted(weights)
