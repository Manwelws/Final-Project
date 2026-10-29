from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import joblib

from final_project.data import load_and_split_data
from final_project.models import train_baseline_model


@dataclass(frozen=True)
class PipelineRunSummary:
    timestamp: str
    algorithm: str
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
    test_size: float = 0.20,
    cv_folds: int = 5,
    random_state: int = 42,
) -> PipelineRunSummary:
    """Orchestrate the full baseline training pipeline, persisting model and metadata."""
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

    # Step 2: Train baseline model with Stratified CV
    result = train_baseline_model(
        X_train=split.X_train,
        y_train=split.y_train,
        cv_folds=cv_folds,
        random_state=random_state,
    )

    # Step 3: Serialize artifacts to output_dir
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    model_file = out_path / "model.joblib"
    joblib.dump(result.model, model_file)

    meta_dict = {
        "timestamp": timestamp,
        "algorithm": "RandomForestClassifier",
        "cv_auc_mean": result.cv_auc_mean,
        "cv_auc_std": result.cv_auc_std,
        "cv_fold_scores": result.cv_fold_scores,
        "train_samples": len(split.X_train),
        "test_samples": len(split.X_test),
        "feature_count": len(result.feature_names),
        "features": result.feature_names,
        "random_state": random_state,
        "model_artifact": str(model_file.name),
    }

    metadata_file = out_path / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    return PipelineRunSummary(
        timestamp=timestamp,
        algorithm="RandomForestClassifier",
        cv_auc_mean=result.cv_auc_mean,
        cv_auc_std=result.cv_auc_std,
        cv_fold_scores=result.cv_fold_scores,
        train_samples=len(split.X_train),
        test_samples=len(split.X_test),
        feature_count=len(result.feature_names),
        features=result.feature_names,
        random_state=random_state,
        model_path=str(model_file),
        metadata_path=str(metadata_file),
    )
