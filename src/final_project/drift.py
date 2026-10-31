"""Drift detection and adaptive sample weighting routines."""

from typing import Any
import numpy as np
import pandas as pd


def calculate_psi(
    reference: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between reference and current samples.

    Threshold interpretation:
        PSI < 0.10: No significant distribution change (stable)
        0.10 <= PSI < 0.25: Moderate distribution shift
        PSI >= 0.25: Significant distribution shift (drift detected)
    """
    ref_arr = np.asarray(reference, dtype=float)
    curr_arr = np.asarray(current, dtype=float)

    ref_arr = ref_arr[~np.isnan(ref_arr)]
    curr_arr = curr_arr[~np.isnan(curr_arr)]

    if len(ref_arr) == 0 or len(curr_arr) == 0:
        return 0.0

    # Determine quantile bins from reference distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref_arr, percentiles)
    bin_edges = np.unique(bin_edges)

    # Fallback if quantiles collapse due to low feature cardinality
    if len(bin_edges) < 2:
        bin_edges = np.linspace(np.min(ref_arr), np.max(ref_arr), num_bins + 1)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            # Single distinct value in reference
            val = ref_arr[0]
            bin_edges = np.array([val - 1e-5, val + 1e-5])

    # Extend outer boundaries to cover extreme or unseen values
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_arr, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_arr, bins=bin_edges)

    # Apply Laplace smoothing to avoid division by zero and log of zero
    n_bins = len(ref_counts)
    p_ref = (ref_counts + epsilon) / (len(ref_arr) + epsilon * n_bins)
    q_curr = (curr_counts + epsilon) / (len(curr_arr) + epsilon * n_bins)

    psi_val = np.sum((q_curr - p_ref) * np.log(q_curr / p_ref))
    return float(max(0.0, psi_val))


def compute_dataset_psi(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str] | None = None,
    num_bins: int = 10,
) -> dict[str, Any]:
    """Compute Population Stability Index across features in a dataset."""
    if features is None:
        features = [c for c in reference_df.columns if c in current_df.columns]

    feature_psi: dict[str, float] = {}
    for feat in features:
        ref_vals = reference_df[feat].to_numpy()
        curr_vals = current_df[feat].to_numpy()
        feature_psi[feat] = calculate_psi(ref_vals, curr_vals, num_bins=num_bins)

    vals = list(feature_psi.values())
    mean_psi = float(np.mean(vals)) if vals else 0.0
    max_psi = float(np.max(vals)) if vals else 0.0

    return {
        "mean_psi": mean_psi,
        "max_psi": max_psi,
        "feature_psi": feature_psi,
    }


def compute_exponential_sample_weights(
    current_week: int,
    available_weeks: pd.Series | np.ndarray,
    decay_rate: float = 0.85,
) -> np.ndarray:
    """Compute exponential sample weights for retraining: decay_rate ** weeks_old."""
    weeks_arr = np.asarray(available_weeks, dtype=float)
    weeks_old = np.maximum(0.0, float(current_week) - weeks_arr)
    weights = decay_rate ** weeks_old
    return np.asarray(weights, dtype=float)
