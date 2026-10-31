"""Model registry, artifact bundling, and typed metadata schemas."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
from typing import Any
import numpy as np
import pandas as pd
import joblib


@dataclass(frozen=True)
class ModelMetadata:
    """Typed schema representing training and evaluation metadata for a serialized model."""

    timestamp: str = ""
    algorithm: str = ""
    balanced: bool = False
    optimal_threshold: float = 0.50
    cost_savings: float = 0.0
    cost_at_optimal: float = 0.0
    cost_at_default: float = 0.0
    holdout_roc_auc: float = 0.0
    holdout_pr_auc: float = 0.0
    holdout_brier_score: float = 0.0
    features: list[str] = field(default_factory=list)
    random_state: int = 42
    tuning_enabled: bool = False
    best_params: dict[str, Any] | None = None
    cv_auc_mean: float = 0.0
    cv_auc_std: float = 0.0
    cv_fold_scores: list[float] = field(default_factory=list)
    train_samples: int = 0
    test_samples: int = 0
    feature_count: int = 0
    model_artifact: str = "model.joblib"
    n_trials: int | None = None
    mlflow_experiment_id: str | None = None
    threshold_comparison: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        d = asdict(self)
        # Prune None values for optional trial fields if not set
        if self.best_params is None:
            d.pop("best_params", None)
        if self.n_trials is None:
            d.pop("n_trials", None)
        if self.mlflow_experiment_id is None:
            d.pop("mlflow_experiment_id", None)
        if self.threshold_comparison is None:
            d.pop("threshold_comparison", None)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Construct ModelMetadata from dictionary, ignoring unknown extra keys."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def save(self, path: Path | str) -> None:
        """Write metadata JSON file to target path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "ModelMetadata":
        """Read and parse metadata JSON from file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Metadata file not found at {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass(frozen=True)
class RiskPrediction:
    """Encapsulates churn probabilities, binary decisions, and operational risk tiers."""

    probabilities: np.ndarray
    predictions: np.ndarray
    risk_tiers: list[str]
    threshold_applied: float

    @property
    def is_single_instance(self) -> bool:
        return len(self.probabilities) == 1


class ModelBundle:
    """Unified repository bundle packaging a trained model estimator with its typed metadata."""

    def __init__(self, model: Any, metadata: ModelMetadata):
        self.model = model
        self.metadata = metadata

    @classmethod
    def load(
        cls,
        model_dir: Path | str = "models",
        model_filename: str | None = None,
        metadata_filename: str = "metadata.json",
    ) -> "ModelBundle":
        """Load trained model estimator and accompanying metadata from directory."""
        dir_path = Path(model_dir)

        if model_filename is not None:
            candidate_paths = [dir_path / model_filename]
        else:
            candidate_paths = [
                dir_path / "model.joblib",
                dir_path / "best_model.joblib",
            ]

        model_path = next((p for p in candidate_paths if p.exists()), None)
        if model_path is None:
            raise FileNotFoundError(
                f"Model artifact not found in {dir_path}. Checked: {[str(p) for p in candidate_paths]}"
            )

        model = joblib.load(model_path)

        meta_path = dir_path / metadata_filename
        if meta_path.exists():
            try:
                metadata = ModelMetadata.load(meta_path)
            except Exception:
                metadata = ModelMetadata(
                    algorithm=model.__class__.__name__,
                    model_artifact=model_path.name,
                )
        else:
            metadata = ModelMetadata(
                algorithm=model.__class__.__name__,
                model_artifact=model_path.name,
            )

        return cls(model=model, metadata=metadata)

    def save(
        self,
        model_dir: Path | str = "models",
        model_filename: str = "model.joblib",
        metadata_filename: str = "metadata.json",
    ) -> None:
        """Persist model artifact and metadata to target directory."""
        dir_path = Path(model_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, dir_path / model_filename)
        self.metadata.save(dir_path / metadata_filename)

    def predict_risk(self, X: pd.DataFrame | np.ndarray) -> RiskPrediction:
        """Predict churn risk using optimal decision threshold and assign operational risk tiers.

        Risk Tiers:
            * Low: probability < optimal_threshold (or < 0.20 if threshold is higher)
            * Medium: optimal_threshold <= probability < 0.50
            * High: probability >= 0.50
        """
        if isinstance(X, pd.DataFrame) and self.metadata.features:
            missing_cols = set(self.metadata.features) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Input DataFrame is missing required features: {missing_cols}")
            # Ensure consistent feature column ordering
            X_eval = X[self.metadata.features]
        else:
            X_eval = X

        probs = self.model.predict_proba(X_eval)[:, 1]
        threshold = (
            self.metadata.optimal_threshold
            if self.metadata.optimal_threshold > 0.0
            else 0.50
        )
        preds = (probs >= threshold).astype(int)

        tiers = []
        for p in probs:
            if p >= 0.50:
                tiers.append("High")
            elif p >= threshold:
                tiers.append("Medium")
            else:
                tiers.append("Low")

        return RiskPrediction(
            probabilities=probs,
            predictions=preds,
            risk_tiers=tiers,
            threshold_applied=threshold,
        )

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return 1D array of positive class churn probabilities."""
        return self.predict_risk(X).probabilities

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return binary predictions applying calibrated decision threshold."""
        return self.predict_risk(X).predictions

