from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import joblib
import pandas as pd

from final_project.data import load_and_split_data
from final_project.models import train_candidate_model, train_baseline_model
from final_project.tuning import run_tuning_study
from final_project.registry import ModelMetadata


@dataclass(frozen=True)
class PipelineRunSummary:
    timestamp: str
    algorithm: str
    balanced: bool
    cv_auc_mean: float
    cv_auc_std: float
    cv_fold_scores: list[float]
    train_samples: int
    test_samples: int
    feature_count: int
    features: list[str]
    random_state: int
    model_path: str
    metadata_path: str


def run_training(
    data_dir: Path | str = "data/processed",
    output_dir: Path | str = "models",
    algorithm: str = "random_forest",
    balanced: bool = False,
    tune: bool = False,
    n_trials: int = 50,
    experiment_name: str = "churn_retraining_optimization",
    tracking_uri: str = "sqlite:///mlflow.db",
    test_size: float = 0.20,
    val_size: float = 0.10,
    cv_folds: int = 5,
    random_state: int = 42,
    algorithms: list[str] | None = None,
) -> PipelineRunSummary:
    """Orchestrate training of candidate models or run Bayesian tuning, persisting model and metadata."""
    data_path = Path(data_dir)
    features_path = data_path / "X_features.csv"
    target_path = data_path / "y_target.csv"

    # Step 1: Load and split data with validation partition for threshold calibration
    split = load_and_split_data(
        features_path=features_path,
        target_path=target_path,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    feature_names = list(split.X_train.columns)
    timestamp = datetime.now(timezone.utc).isoformat()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_file = out_path / "model.joblib"

    if tune:
        # Step 2a: Run Optuna study with MLflow experiment tracking on train split
        tuning_res = run_tuning_study(
            X_train=split.X_train,
            y_train=split.y_train,
            n_trials=n_trials,
            cv_folds=cv_folds,
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            random_state=random_state,
            algorithms=algorithms,
        )

        final_model = tuning_res.best_model
        algo_name = final_model.__class__.__name__
        cv_auc_mean = tuning_res.best_roc_auc
        cv_auc_std = 0.0
        cv_fold_scores = [cv_auc_mean]
        is_balanced = tuning_res.best_params.get("balanced", False)

    else:
        # Step 2b: Standard Candidate Training with Stratified CV on train split
        result = train_candidate_model(
            X_train=split.X_train,
            y_train=split.y_train,
            algorithm=algorithm,
            balanced=balanced,
            cv_folds=cv_folds,
            random_state=random_state,
        )

        final_model = result.model
        algo_name = final_model.__class__.__name__
        cv_auc_mean = result.cv_auc_mean
        cv_auc_std = result.cv_auc_std
        cv_fold_scores = result.cv_fold_scores
        is_balanced = balanced

    # Step 3: Threshold calibration on validation split + refit on train+val (D8)
    optimal_threshold = 0.50
    if split.X_val is not None and split.y_val is not None:
        from final_project.evaluation import optimize_threshold
        y_val_probs = final_model.predict_proba(split.X_val)[:, 1]
        val_opt = optimize_threshold(
            y_true=split.y_val.to_numpy(),
            y_probs=y_val_probs,
        )
        optimal_threshold = val_opt.cost_optimal_threshold

        # Refit winning model on train + validation before deployment
        X_train_val = pd.concat([split.X_train, split.X_val], ignore_index=True)
        y_train_val = pd.concat([split.y_train, split.y_val], ignore_index=True)
        final_model.fit(X_train_val, y_train_val)
        train_samples_count = len(X_train_val)
    else:
        train_samples_count = len(split.X_train)

    joblib.dump(final_model, model_file)

    metadata = ModelMetadata(
        timestamp=timestamp,
        algorithm=algo_name,
        balanced=is_balanced,
        tuning_enabled=tune,
        n_trials=n_trials if tune else None,
        best_params=tuning_res.best_params if tune else None,
        cv_auc_mean=cv_auc_mean,
        cv_auc_std=cv_auc_std,
        cv_fold_scores=cv_fold_scores,
        mlflow_experiment_id=tuning_res.experiment_id if tune else None,
        optimal_threshold=optimal_threshold,
        train_samples=train_samples_count,
        test_samples=len(split.X_test),
        feature_count=len(feature_names),
        features=feature_names,
        random_state=random_state,
        model_artifact=str(model_file.name),
    )

    metadata_file = out_path / "metadata.json"
    metadata.save(metadata_file)

    return PipelineRunSummary(
        timestamp=timestamp,
        algorithm=algo_name,
        balanced=is_balanced,
        cv_auc_mean=cv_auc_mean,
        cv_auc_std=cv_auc_std,
        cv_fold_scores=cv_fold_scores,
        train_samples=len(split.X_train),
        test_samples=len(split.X_test),
        feature_count=len(feature_names),
        features=feature_names,
        random_state=random_state,
        model_path=str(model_file),
        metadata_path=str(metadata_file),
    )
