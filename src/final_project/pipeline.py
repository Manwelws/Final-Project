from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import joblib

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
    tracking_uri: str = "file:./mlruns",
    test_size: float = 0.20,
    cv_folds: int = 5,
    random_state: int = 42,
) -> PipelineRunSummary:
    """Orchestrate training of candidate models or run Bayesian tuning, persisting model and metadata."""
    data_path = Path(data_dir)
    features_path = data_path / "X_features.csv"
    target_path = data_path / "y_target.csv"

    # Step 1: Load and split data
    split = load_and_split_data(
        features_path=features_path,
        target_path=target_path,
        test_size=test_size,
        random_state=random_state,
    )

    feature_names = list(split.X_train.columns)
    timestamp = datetime.now(timezone.utc).isoformat()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_file = out_path / "model.joblib"

    if tune:
        # Step 2a: Run Optuna study with MLflow experiment tracking
        tuning_res = run_tuning_study(
            X_train=split.X_train,
            y_train=split.y_train,
            n_trials=n_trials,
            cv_folds=cv_folds,
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            random_state=random_state,
        )

        final_model = tuning_res.best_model
        algo_name = final_model.__class__.__name__
        cv_auc_mean = tuning_res.best_roc_auc
        cv_auc_std = 0.0
        cv_fold_scores = [cv_auc_mean]
        is_balanced = tuning_res.best_params.get("balanced", False)

        joblib.dump(final_model, model_file)

        metadata = ModelMetadata(
            timestamp=timestamp,
            algorithm=algo_name,
            balanced=is_balanced,
            tuning_enabled=True,
            n_trials=n_trials,
            best_params=tuning_res.best_params,
            cv_auc_mean=cv_auc_mean,
            cv_auc_std=cv_auc_std,
            cv_fold_scores=cv_fold_scores,
            mlflow_experiment_id=tuning_res.experiment_id,
            train_samples=len(split.X_train),
            test_samples=len(split.X_test),
            feature_count=len(feature_names),
            features=feature_names,
            random_state=random_state,
            model_artifact=str(model_file.name),
        )

    else:
        # Step 2b: Standard Candidate Training with Stratified CV
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

        joblib.dump(final_model, model_file)

        metadata = ModelMetadata(
            timestamp=timestamp,
            algorithm=algo_name,
            balanced=is_balanced,
            tuning_enabled=False,
            cv_auc_mean=cv_auc_mean,
            cv_auc_std=cv_auc_std,
            cv_fold_scores=cv_fold_scores,
            train_samples=len(split.X_train),
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
