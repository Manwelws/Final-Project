"""Retraining simulation engine modeling streaming data, drift, and retraining."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from final_project.drift import (
    compute_dataset_psi,
    compute_exponential_sample_weights,
)
from final_project.domain.policies import DriftPolicy, RetrainingPolicy
from final_project.data import load_features_and_target



@dataclass
class SimulationConfig:
    """Configuration parameters for weekly retraining simulation."""

    n_weeks: int = 8
    drift_start_week: int = 5
    label_delay_weeks: int = 2
    price_increase_mult: float = 1.25
    tenure_factor: float = 0.70
    concept_flip_prob: float = 0.30
    returning_frac: float = 0.20
    auc_drop_threshold: float = 0.05
    min_labels_for_retrain: int = 200
    reference_frac: float = 0.15
    recent_window_weeks: int = 2
    sample_decay_rate: float = 0.85
    random_state: int = 42


@dataclass(frozen=True)
class SimulationRunResults:
    """Summary and artifacts resulting from a simulation execution."""

    drift_metrics: pd.DataFrame
    retrained_weeks: list[int]
    final_never_retrain_auc: float | None
    final_retrain_auc_recent: float | None
    artifacts_saved: bool


class DualDriftMonitor:
    """Dual-trigger monitor tracking unsupervised PSI and supervised performance decay."""

    def __init__(
        self,
        auc_drop_threshold: float = 0.05,
        drift_policy: DriftPolicy | None = None,
        retraining_policy: RetrainingPolicy | None = None,
    ):
        self.auc_drop_threshold = auc_drop_threshold
        self.drift_policy = drift_policy or DriftPolicy()
        self.retraining_policy = retraining_policy or RetrainingPolicy(
            auc_drop_threshold=auc_drop_threshold
        )
        self.best_recent_auc: float | None = None

    def evaluate_unsupervised_drift(
        self,
        reference_features: pd.DataFrame,
        current_features: pd.DataFrame,
    ) -> dict[str, Any]:
        """Compute PSI across features and classify drift status."""
        psi_results = compute_dataset_psi(reference_features, current_features)
        psi_mean = float(psi_results["mean_psi"])
        psi_max = float(psi_results["max_psi"])
        psi_status = self.drift_policy.classify_psi(psi_max)
        return {
            "psi_mean": psi_mean,
            "psi_max": psi_max,
            "psi_status": psi_status,
        }

    def check_decay(
        self,
        recent_auc: float,
        model_retrain_exists: bool,
        n_labeled: int | None = None,
    ) -> tuple[bool, str]:
        """Evaluate if model performance has degraded beyond tolerance."""
        return self.retraining_policy.should_retrain(
            recent_auc=recent_auc,
            best_recent_auc=self.best_recent_auc,
            model_exists=model_retrain_exists,
            n_labeled=n_labeled,
        )

    def update_best_auc(self, new_auc: float) -> None:
        """Update historical peak performance benchmark."""
        self.best_recent_auc = new_auc


class StreamingBatchEngine:
    """Generates weekly streaming batches and injects covariate and concept drift."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.RandomState(config.random_state)

    def prepare_population_and_reference(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Combine features and target, assign synthetic customer IDs, and extract reference holdout."""
        data = pd.concat([features, target], axis=1).reset_index(drop=True)
        data["customer_id"] = data.index.astype(str)
        data = data.sample(frac=1, random_state=self.config.random_state).reset_index(drop=True)

        n_ref = int(len(data) * self.config.reference_frac)
        reference = data.iloc[:n_ref].copy()
        population = data.iloc[n_ref:].copy()
        return reference, population

    def build_weekly_batches(self, population: pd.DataFrame) -> list[pd.DataFrame]:
        """Partition population into weekly streams with hybrid returning customer overlap."""
        split_indices = np.array_split(np.arange(len(population)), self.config.n_weeks)
        new_batches = [population.iloc[idx].reset_index(drop=True) for idx in split_indices]

        weekly_batches: list[pd.DataFrame] = []
        for i in range(self.config.n_weeks):
            new_batch = new_batches[i].copy()

            if i == 0:
                weekly_batch = new_batch.copy()
            else:
                prev_batch = weekly_batches[i - 1]
                n_returning = int(len(new_batch) * self.config.returning_frac)

                if n_returning > 0 and len(prev_batch) > 0:
                    returning = prev_batch.sample(
                        n=n_returning, random_state=self.config.random_state + i
                    ).copy()
                    returning["Tenure Months"] = (returning["Tenure Months"] + 1).clip(upper=72)
                    returning["Monthly Charges"] = returning["Monthly Charges"] * 1.02
                    weekly_batch = pd.concat([new_batch, returning], ignore_index=True)
                else:
                    weekly_batch = new_batch.copy()

            weekly_batch["feature_available_week"] = i + 1
            weekly_batch["label_release_week"] = i + 1 + self.config.label_delay_weeks
            weekly_batches.append(weekly_batch)

        return weekly_batches

    def inject_drift(self, weekly_batches: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """Inject covariate drift and concept drift from drift_start_week onwards."""
        for i in range(self.config.drift_start_week - 1, self.config.n_weeks):
            batch = weekly_batches[i]

            # Covariate drift
            batch["Monthly Charges"] = batch["Monthly Charges"] * self.config.price_increase_mult
            batch["Tenure Months"] = (batch["Tenure Months"] * self.config.tenure_factor).round().clip(lower=0)

            # Concept drift (flip 30% of high-charge retained to churned)
            median_charge = batch["Monthly Charges"].median()
            high_charge = batch["Monthly Charges"] > median_charge
            was_retained = batch["Churn Value"] == 0
            flip_mask = high_charge & was_retained & (self.rng.rand(len(batch)) < self.config.concept_flip_prob)
            batch.loc[flip_mask, "Churn Value"] = 1
            print(f"Week {i+1}: {int(flip_mask.sum())} labels flipped due to concept drift")

        return weekly_batches


def _safe_roc_auc_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float | None:
    """Compute ROC-AUC safely, returning None if y_true contains fewer than 2 classes."""
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_pred_proba))
    except (ValueError, TypeError):
        return None


def _safe_fit_with_sample_weights(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> None:
    """Fit model with sample weights if supported, gracefully falling back for pipelines or unweighted estimators."""
    if sample_weights is None:
        model.fit(X, y)
        return

    # Check for Pipeline
    if hasattr(model, "named_steps"):
        step_names = list(model.named_steps.keys())
        clf_name = step_names[-1]
        try:
            model.fit(X, y, **{f"{clf_name}__sample_weight": sample_weights})
            return
        except (TypeError, ValueError):
            pass

    # Direct fit with sample_weight
    try:
        model.fit(X, y, sample_weight=sample_weights)
        return
    except (TypeError, ValueError):
        pass

    # Unweighted fallback
    model.fit(X, y)


class SimulationRunner:
    """Orchestrates end-to-end streaming retraining simulation."""

    def __init__(
        self,
        config: SimulationConfig | None = None,
        base_model: Any | None = None,
        data_dir: Path | str = "data/processed",
        models_dir: Path | str = "models",
        live_dir: Path | str = "data/live",
        reference_dir: Path | str = "data/reference",
        retraining_dir: Path | str = "models/retrained",
        drift_dir: Path | str = "data/drift",
        save_artifacts: bool = True,
        monitor: DualDriftMonitor | None = None,
        engine: StreamingBatchEngine | None = None,
        verbose: bool = True,
    ):
        self.config = config or SimulationConfig()
        self.base_model = base_model
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.live_dir = Path(live_dir)
        self.reference_dir = Path(reference_dir)
        self.retraining_dir = Path(retraining_dir)
        self.drift_dir = Path(drift_dir)
        self.save_artifacts = save_artifacts
        self.verbose = verbose

        retrain_policy = RetrainingPolicy(
            auc_drop_threshold=self.config.auc_drop_threshold,
            min_labels_for_retrain=self.config.min_labels_for_retrain,
            recent_window_weeks=self.config.recent_window_weeks,
            sample_decay_rate=self.config.sample_decay_rate,
        )
        self.monitor = monitor or DualDriftMonitor(
            auc_drop_threshold=self.config.auc_drop_threshold,
            retraining_policy=retrain_policy,
        )
        self.engine = engine or StreamingBatchEngine(self.config)

    def _log(self, message: str) -> None:
        """Internal logging respecting verbose flag."""
        if self.verbose:
            print(message)

    def _resolve_base_model(self) -> Any:
        if self.base_model is not None:
            return self.base_model
        for name in ["model.joblib", "best_model.joblib"]:
            p = self.models_dir / name
            if p.exists():
                self._log(f"Loading production model artifact from {p}")
                return joblib.load(p)
        raise FileNotFoundError(
            f"Production model artifact not found in {self.models_dir}. Run 'churn train' first."
        )

    def _setup_directories(self) -> tuple[Path, Path, Path]:
        live_feat_dir = self.live_dir / "features"
        live_label_dir = self.live_dir / "labels"
        live_pend_dir = self.live_dir / "pending"
        if self.save_artifacts:
            for d in [
                live_feat_dir,
                live_label_dir,
                live_pend_dir,
                self.reference_dir,
                self.retraining_dir,
                self.drift_dir,
            ]:
                d.mkdir(parents=True, exist_ok=True)
        return live_feat_dir, live_label_dir, live_pend_dir

    def _prepare_stream(self) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """Load features, partition reference/population, and inject weekly drift."""
        X, y_series = load_features_and_target(self.data_dir)
        y = pd.DataFrame({"Churn Value": y_series})

        reference, population = self.engine.prepare_population_and_reference(X, y)

        if self.save_artifacts:
            reference.to_csv(self.reference_dir / "X_reference.csv", index=False)
            reference[["customer_id", "Churn Value"]].to_csv(
                self.reference_dir / "y_reference.csv", index=False
            )

        self._log(f"Population size: {len(population)}")
        self._log(f"Reference set size: {len(reference)}")

        weekly_batches = self.engine.build_weekly_batches(population)
        weekly_batches = self.engine.inject_drift(weekly_batches)

        master = pd.concat(weekly_batches, ignore_index=True)
        self._log(f"Total weekly records (with returning duplicates): {len(master)}")
        self._log(f"Unique customers: {master['customer_id'].nunique()}")

        meta_cols = ["customer_id", "Churn Value", "feature_available_week", "label_release_week"]
        feature_cols = [c for c in master.columns if c not in meta_cols]
        return reference, master, feature_cols

    def _fit_never_retrain_baseline(
        self,
        base_production_model: Any,
        master: pd.DataFrame,
        feature_cols: list[str],
    ) -> Any:
        """Fit static never-retrain baseline on pre-drift released data."""
        never_retrain_data = master[
            master["label_release_week"] < self.config.drift_start_week
        ].copy()

        if len(never_retrain_data) > 0:
            X_nr = never_retrain_data[feature_cols]
            y_nr = never_retrain_data["Churn Value"].values
            model_never = clone(base_production_model)
            model_never.fit(X_nr, y_nr)
            self._log(
                f"\nNever-retrain baseline fitted on {len(never_retrain_data)} samples "
                f"(weeks 1-{self.config.drift_start_week - 1})"
            )
        else:
            model_never = clone(base_production_model)
            self._log("WARNING: Using base production model directly for never-retrain baseline")
        return model_never

    def run(self) -> SimulationRunResults:
        """Run the multi-week streaming simulation."""
        base_production_model = self._resolve_base_model()
        live_feat_dir, live_label_dir, live_pend_dir = self._setup_directories()
        reference, master, feature_cols = self._prepare_stream()

        X_ref = reference[feature_cols].copy()
        y_ref = reference["Churn Value"].values

        model_never = self._fit_never_retrain_baseline(base_production_model, master, feature_cols)
        model_retrain: Any | None = None
        auc_never: float | None = None

        drift_metrics: list[dict[str, Any]] = []
        retrained_weeks: list[int] = []

        for week in range(1, self.config.n_weeks + 1):
            self._log(f"\n{'='*60}\nSIMULATION WEEK {week}\n{'='*60}")

            # 1. Release features for week
            week_features = master[
                master["feature_available_week"] == week
            ][feature_cols + ["customer_id"]]

            if self.save_artifacts:
                week_features.to_csv(live_feat_dir / f"week_{week}_features.csv", index=False)
                week_features.copy().to_csv(live_pend_dir / f"week_{week}_pending.csv", index=False)

            # 2. Unsupervised PSI Drift Monitor
            psi_info = self.monitor.evaluate_unsupervised_drift(
                reference[feature_cols], week_features[feature_cols]
            )
            psi_mean = psi_info["psi_mean"]
            psi_max = psi_info["psi_max"]
            psi_status = psi_info["psi_status"]
            self._log(f"  Unsupervised Drift Monitor (PSI): mean={psi_mean:.4f}, max={psi_max:.4f} [{psi_status}]")

            # 3. Release delayed labels
            release_week = week - self.config.label_delay_weeks
            if release_week >= 1:
                week_labels = master[
                    master["feature_available_week"] == release_week
                ][["customer_id", "Churn Value"]].copy()
                if self.save_artifacts:
                    week_labels.to_csv(live_label_dir / f"week_{release_week}_labels.csv", index=False)
                    pending_file = live_pend_dir / f"week_{release_week}_pending.csv"
                    if pending_file.exists():
                        pending_file.unlink()
                self._log(f"  Labels released for week {release_week}")

            # 4. Check available labeled data
            labeled_data = master[master["label_release_week"] <= week].copy()
            n_labeled = len(labeled_data)
            self._log(f"  Labeled samples available: {n_labeled}")

            if n_labeled < self.config.min_labels_for_retrain:
                self._log("  Not enough labeled data to train/evaluate yet.")
                drift_metrics.append({
                    "week": week,
                    "n_labeled": n_labeled,
                    "psi_mean": round(psi_mean, 4),
                    "psi_max": round(psi_max, 4),
                    "psi_status": psi_status,
                    "never_retrain_auc_reference": None,
                    "retrain_model_auc_recent": None,
                    "retrained": False,
                    "trigger_reason": "awaiting label release",
                })
                continue

            X_avail = labeled_data[feature_cols]
            y_avail = labeled_data["Churn Value"].values

            # 5. Build recent live window
            release_cutoff = week - self.config.recent_window_weeks + 1
            recent_live = master[
                (master["label_release_week"] >= release_cutoff) &
                (master["label_release_week"] <= week)
            ].copy()

            if len(recent_live) > 0:
                X_recent = recent_live[feature_cols]
                y_recent = recent_live["Churn Value"].values
            else:
                X_recent = X_avail
                y_recent = y_avail

            # 6. Never-retrain baseline evaluation on static reference
            auc_never = None
            if model_never is not None:
                y_never_pred = model_never.predict_proba(X_ref)[:, 1]
                auc_never = _safe_roc_auc_score(y_ref, y_never_pred)
                if auc_never is not None:
                    self._log(f"  Never-retrain AUC on static reference: {auc_never:.4f}")

            # 7. Supervised performance decay check (R1-1: strictly out-of-sample)
            should_retrain = False
            trigger_reason = ""

            # Score the incoming live window using the active model frozen BEFORE this week
            active_model = model_retrain if model_retrain is not None else base_production_model
            y_ret_pred = active_model.predict_proba(X_recent)[:, 1]
            auc_ret_recent = _safe_roc_auc_score(y_recent, y_ret_pred)

            if auc_ret_recent is not None:
                self._log(f"  Live out-of-sample AUC on recent data: {auc_ret_recent:.4f}")

            if model_retrain is None:
                should_retrain = True
                trigger_reason = "initial training"
                if auc_ret_recent is not None:
                    self.monitor.update_best_auc(auc_ret_recent)
            else:
                if auc_ret_recent is not None:
                    decay_triggered, reason = self.monitor.check_decay(
                        recent_auc=auc_ret_recent,
                        model_retrain_exists=True,
                    )
                    if decay_triggered:
                        should_retrain = True
                        trigger_reason = reason
                    else:
                        if self.monitor.best_recent_auc is not None:
                            self.monitor.update_best_auc(max(self.monitor.best_recent_auc, auc_ret_recent))
                        else:
                            self.monitor.update_best_auc(auc_ret_recent)
                        self._log(f"  No retraining triggered (best recent AUC: {self.monitor.best_recent_auc:.4f})")

            # 8. Retraining with Exponential Sample Decay
            if should_retrain and n_labeled >= self.config.min_labels_for_retrain:
                self._log(f"  RETRAINING triggered: {trigger_reason}")

                sample_weights = compute_exponential_sample_weights(
                    current_week=week,
                    available_weeks=labeled_data["feature_available_week"],
                    decay_rate=self.config.sample_decay_rate,
                )

                model_retrain = clone(base_production_model)
                _safe_fit_with_sample_weights(model_retrain, X_avail, y_avail, sample_weights)

                # Evaluated on static reference (R1-1: do NOT evaluate on X_recent in-sample)
                y_ret_pred_ref = model_retrain.predict_proba(X_ref)[:, 1]
                auc_ret_ref = _safe_roc_auc_score(y_ref, y_ret_pred_ref)

                retrained_weeks.append(week)
                if auc_ret_ref is not None:
                    self._log(f"  Retrained model AUC on static reference: {auc_ret_ref:.4f}")

                if self.save_artifacts:
                    retrain_dir = self.retraining_dir / f"week_{week}"
                    retrain_dir.mkdir(parents=True, exist_ok=True)
                    metrics = {
                        "week": week,
                        "trigger": trigger_reason,
                        "n_labeled": n_labeled,
                        "psi_mean": round(psi_mean, 4),
                        "psi_max": round(psi_max, 4),
                        "psi_status": psi_status,
                        "sample_decay_rate": self.config.sample_decay_rate,
                        "auc_on_recent_live": round(auc_ret_recent, 4) if auc_ret_recent is not None else None,
                        "auc_on_static_reference": round(auc_ret_ref, 4) if auc_ret_ref is not None else None,
                        "never_retrain_auc_reference": round(auc_never, 4) if auc_never is not None else None,
                    }
                    with open(retrain_dir / "metrics.json", "w") as f:
                        json.dump(metrics, f, indent=2)
                    joblib.dump(model_retrain, retrain_dir / "model.joblib")

            drift_metrics.append({
                "week": week,
                "n_labeled": n_labeled,
                "psi_mean": round(psi_mean, 4),
                "psi_max": round(psi_max, 4),
                "psi_status": psi_status,
                "never_retrain_auc_reference": round(auc_never, 4) if auc_never is not None else None,
                "retrain_model_auc_recent": (
                    round(auc_ret_recent, 4)
                    if auc_ret_recent is not None
                    else None
                ),
                "best_retrain_auc": (
                    round(self.monitor.best_recent_auc, 4)
                    if self.monitor.best_recent_auc is not None
                    else None
                ),
                "retrained": week in retrained_weeks,
                "trigger_reason": trigger_reason if should_retrain else "monitoring",
            })

        drift_metrics_df = pd.DataFrame(drift_metrics)
        if self.save_artifacts:
            drift_metrics_df.to_csv(self.drift_dir / "drift_metrics.csv", index=False)

        self._log(f"\n{'='*60}\nSIMULATION COMPLETE\n{'='*60}")
        self._log(f"Retraining triggered at weeks: {retrained_weeks}")
        if self.save_artifacts:
            self._log(f"Metrics saved to {self.drift_dir / 'drift_metrics.csv'}")
            self._log(f"Retraining artifacts in {self.retraining_dir}")

        if not drift_metrics_df.empty:
            self._log("\nDrift Metrics Summary:")
            self._log(drift_metrics_df.to_string(index=False))

        return SimulationRunResults(
            drift_metrics=drift_metrics_df,
            retrained_weeks=retrained_weeks,
            final_never_retrain_auc=auc_never,
            final_retrain_auc_recent=self.monitor.best_recent_auc,
            artifacts_saved=self.save_artifacts,
        )


def run_simulation(
    config: SimulationConfig | None = None,
    base_model: Any | None = None,
    data_dir: Path | str = "data/processed",
    models_dir: Path | str = "models",
    save_artifacts: bool = True,
    verbose: bool = True,
) -> SimulationRunResults:
    """Convenience function running streaming retraining simulation with standard conventions."""
    runner = SimulationRunner(
        config=config,
        base_model=base_model,
        data_dir=data_dir,
        models_dir=models_dir,
        save_artifacts=save_artifacts,
        verbose=verbose,
    )
    return runner.run()
