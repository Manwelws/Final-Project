"""Domain policies for drift detection and automated retraining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriftPolicy:
    """Thresholds and categorization rules for Population Stability Index (PSI) drift monitoring."""

    psi_threshold_warning: float = 0.10
    psi_threshold_alert: float = 0.25

    def classify_psi(self, max_psi: float) -> str:
        """Categorize drift status based on maximum feature PSI."""
        if max_psi >= self.psi_threshold_alert:
            return "Significant Shift (Drift Detected)"
        elif max_psi >= self.psi_threshold_warning:
            return "Moderate Shift"
        return "Stable"

    def is_drift_detected(self, max_psi: float) -> bool:
        """Return True if PSI breaches the alert threshold requiring operational attention."""
        return bool(max_psi >= self.psi_threshold_alert)


@dataclass(frozen=True)
class RetrainingPolicy:
    """Policy rules governing when and how to retrain production models under ADR 0001."""

    auc_drop_threshold: float = 0.05
    min_labels_for_retrain: int = 200
    recent_window_weeks: int = 2
    sample_decay_rate: float = 0.85

    def should_retrain(
        self,
        recent_auc: float | None,
        best_recent_auc: float | None,
        model_exists: bool = True,
        n_labeled: int | None = None,
    ) -> tuple[bool, str]:
        """Evaluate if model performance has degraded beyond tolerance or requires bootstrap.

        Returns:
            tuple[bool, str]: (should_trigger_retraining, decision_reason_explanation)
        """
        if not model_exists:
            return True, "initial training"

        if n_labeled is not None and n_labeled < self.min_labels_for_retrain:
            return False, f"insufficient labeled samples ({n_labeled} < {self.min_labels_for_retrain})"

        if recent_auc is not None and best_recent_auc is not None:
            if recent_auc < best_recent_auc - self.auc_drop_threshold:
                return (
                    True,
                    f"performance decay: recent AUC {recent_auc:.4f} < "
                    f"best {best_recent_auc:.4f} - {self.auc_drop_threshold}",
                )

        return False, "monitoring"

    def compute_sample_weights(
        self,
        current_week: int,
        available_weeks: pd.Series | np.ndarray | list[int],
    ) -> np.ndarray:
        """Compute exponential sample weights for retraining: decay_rate ** weeks_old."""
        weeks_arr = np.asarray(available_weeks, dtype=float)
        weeks_old = np.maximum(0.0, float(current_week) - weeks_arr)
        weights = self.sample_decay_rate ** weeks_old
        return np.asarray(weights, dtype=float)
