from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


@dataclass(frozen=True)
class TrainingResult:
    model: BaseEstimator
    cv_auc_mean: float
    cv_auc_std: float
    cv_fold_scores: list[float]
    feature_names: list[str]


@dataclass(frozen=True)
class ModelCVMetrics:
    model_name: str
    balanced: bool
    roc_auc_mean: float
    roc_auc_std: float
    pr_auc_mean: float
    pr_auc_std: float


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Compute negative / positive ratio for XGBoost scale_pos_weight balancing."""
    counts = y.value_counts()
    neg_count = counts.get(0, 0)
    pos_count = counts.get(1, 0)
    if pos_count == 0:
        return 1.0
    return float(neg_count / pos_count)


def get_candidate_model(
    name: str,
    balanced: bool = False,
    scale_pos_weight: float | None = None,
    random_state: int = 42,
    **kwargs: Any,
) -> BaseEstimator:
    """Instantiate a tree-based candidate classifier with native class weighting."""
    normalized_name = name.lower().replace("-", "_").replace(" ", "_")

    if normalized_name in ("random_forest", "rf"):
        class_weight = "balanced" if balanced else None
        return RandomForestClassifier(
            random_state=random_state,
            class_weight=class_weight,
            **kwargs,
        )

    elif normalized_name in ("lightgbm", "lgbm", "lgb"):
        class_weight = "balanced" if balanced else None
        return LGBMClassifier(
            random_state=random_state,
            class_weight=class_weight,
            verbose=-1,
            **kwargs,
        )

    elif normalized_name in ("xgboost", "xgb"):
        weight = 1.0
        if balanced:
            weight = scale_pos_weight if scale_pos_weight is not None else 1.0
        elif scale_pos_weight is not None:
            weight = scale_pos_weight
        return XGBClassifier(
            random_state=random_state,
            scale_pos_weight=weight,
            eval_metric="logloss",
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unsupported algorithm '{name}'. Choose from 'random_forest', 'lightgbm', 'xgboost'."
        )


def train_candidate_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    algorithm: str = "random_forest",
    balanced: bool = False,
    cv_folds: int = 5,
    random_state: int = 42,
) -> TrainingResult:
    """Train a specified candidate classifier with optional class weighting and Stratified CV."""
    feature_names = list(X_train.columns)
    scale_pos_weight = compute_scale_pos_weight(y_train) if balanced else 1.0

    model = get_candidate_model(
        algorithm,
        balanced=balanced,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    fold_scores = [float(s) for s in scores]
    cv_auc_mean = float(np.mean(fold_scores))
    cv_auc_std = float(np.std(fold_scores))

    model.fit(X_train, y_train)

    return TrainingResult(
        model=model,
        cv_auc_mean=cv_auc_mean,
        cv_auc_std=cv_auc_std,
        cv_fold_scores=fold_scores,
        feature_names=feature_names,
    )


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42,
) -> TrainingResult:
    """Train a baseline Random Forest classifier and evaluate 5-fold CV ROC-AUC."""
    return train_candidate_model(
        X_train=X_train,
        y_train=y_train,
        algorithm="random_forest",
        balanced=False,
        cv_folds=cv_folds,
        random_state=random_state,
    )


def evaluate_model_cv(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    model_name: str = "Model",
    balanced: bool = False,
    random_state: int = 42,
) -> ModelCVMetrics:
    """Evaluate an estimator using Stratified K-Fold CV on ROC-AUC and PR-AUC."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    
    roc_scores = scores["test_roc_auc"]
    pr_scores = scores["test_pr_auc"]

    return ModelCVMetrics(
        model_name=model_name,
        balanced=balanced,
        roc_auc_mean=float(np.mean(roc_scores)),
        roc_auc_std=float(np.std(roc_scores)),
        pr_auc_mean=float(np.mean(pr_scores)),
        pr_auc_std=float(np.std(pr_scores)),
    )


def benchmark_candidates(
    X: pd.DataFrame,
    y: pd.Series,
    algorithms: list[str] | None = None,
    cv_folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Benchmark candidate models across 5-fold stratified CV, comparing unweighted and class-weighted modes."""
    if algorithms is None:
        algorithms = ["random_forest", "lightgbm", "xgboost"]

    scale_pos_weight = compute_scale_pos_weight(y)
    display_names = {
        "random_forest": "Random Forest",
        "rf": "Random Forest",
        "lightgbm": "LightGBM",
        "lgbm": "LightGBM",
        "lgb": "LightGBM",
        "xgboost": "XGBoost",
        "xgb": "XGBoost",
    }

    records = []
    for algo in algorithms:
        d_name = display_names.get(algo.lower(), algo)
        for balanced in [False, True]:
            model = get_candidate_model(
                algo,
                balanced=balanced,
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
            )
            metrics = evaluate_model_cv(
                model=model,
                X=X,
                y=y,
                cv_folds=cv_folds,
                model_name=d_name,
                balanced=balanced,
                random_state=random_state,
            )
            records.append({
                "model": d_name,
                "weighting": "Balanced" if balanced else "Unweighted",
                "roc_auc_mean": metrics.roc_auc_mean,
                "roc_auc_std": metrics.roc_auc_std,
                "pr_auc_mean": metrics.pr_auc_mean,
                "pr_auc_std": metrics.pr_auc_std,
            })

    df = pd.DataFrame(records)
    df = df.sort_values(by="roc_auc_mean", ascending=False).reset_index(drop=True)
    return df
