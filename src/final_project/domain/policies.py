"""Domain policies for drift detection and automated retraining."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DriftPolicy:
    """Thresholds for Population Stability Index (PSI) drift monitoring."""

    psi_threshold_warning: float = 0.10
    psi_threshold_alert: float = 0.25

    def classify_psi(self, max_psi: float) -> str:
        """Categorize drift status based on maximum feature PSI."""
        if max_psi >= self.psi_threshold_alert:
            return "Significant Shift (Drift Detected)"
        elif max_psi >= self.psi_threshold_warning:
            return "Moderate Shift"
        return "Stable"


@dataclass(frozen=True)
class RetrainingPolicy:
    """Policy rules governing when and how to retrain production models."""

    auc_drop_threshold: float = 0.05
    min_labels_for_retrain: int = 200
    recent_window_weeks: int = 2
    sample_decay_rate: float = 0.85
