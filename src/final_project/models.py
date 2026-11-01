from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
    """Instantiate a tree-based, linear, neural, or ensemble candidate classifier with class weighting."""
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
        weight = scale_pos_weight if (balanced and scale_pos_weight is not None) else 1.0
        return XGBClassifier(
            random_state=random_state,
            scale_pos_weight=weight,
            eval_metric="logloss",
            **kwargs,
        )

    elif normalized_name in ("logistic_regression", "lr", "logistic"):
        class_weight = "balanced" if balanced else None
        max_iter = kwargs.pop("max_iter", 1000)
        c_val = kwargs.pop("C", 1.0)
        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_val,
                    max_iter=max_iter,
                    class_weight=class_weight,
                    random_state=random_state,
                    **kwargs,
                ),
            ),
        ])

    elif normalized_name in ("bagged_logistic", "bagged_lr"):
        class_weight = "balanced" if balanced else None
        n_estimators = kwargs.pop("n_estimators", 30)
        base_lr = Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight=class_weight,
                    random_state=random_state,
                    **kwargs,
                ),
            ),
        ])
        return BaggingClassifier(
            estimator=base_lr,
            n_estimators=n_estimators,
            random_state=random_state,
        )

    elif normalized_name in ("mlp", "neural_network", "mlp_classifier"):
        hidden_layer_sizes = kwargs.pop("hidden_layer_sizes", (64, 32))
        max_iter = kwargs.pop("max_iter", 500)
        alpha = kwargs.pop("alpha", 0.001)
        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    max_iter=max_iter,
                    alpha=alpha,
                    early_stopping=True,
                    random_state=random_state,
                    **kwargs,
                ),
            ),
        ])

    elif normalized_name in ("knn", "kneighbors"):
        n_neighbors = kwargs.pop("n_neighbors", 15)
        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    **kwargs,
                ),
            ),
        ])

    elif normalized_name in ("voting_ensemble", "voting"):
        class_weight = "balanced" if balanced else None
        lgb_est = LGBMClassifier(
            random_state=random_state,
            class_weight=class_weight,
            verbose=-1,
        )
        lr_est = Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight=class_weight,
                    random_state=random_state,
                ),
            ),
        ])
        rf_est = RandomForestClassifier(
            random_state=random_state,
            class_weight=class_weight,
        )
        return VotingClassifier(
            estimators=[
                ("lgb", lgb_est),
                ("lr", lr_est),
                ("rf", rf_est),
            ],
            voting="soft",
            **kwargs,
        )

    elif normalized_name in ("stacking_ensemble", "stacking"):
        class_weight = "balanced" if balanced else None
        lgb_est = LGBMClassifier(
            random_state=random_state,
            class_weight=class_weight,
            verbose=-1,
        )
        lr_est = Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight=class_weight,
                    random_state=random_state,
                ),
            ),
        ])
        rf_est = RandomForestClassifier(
            random_state=random_state,
            class_weight=class_weight,
        )
        final_est = LogisticRegression(random_state=random_state)
        cv_splits = kwargs.pop("cv", 3)
        return StackingClassifier(
            estimators=[
                ("lgb", lgb_est),
                ("lr", lr_est),
                ("rf", rf_est),
            ],
            final_estimator=final_est,
            cv=cv_splits,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unsupported algorithm '{name}'. Choose from 'random_forest', 'lightgbm', 'xgboost', "
            f"'logistic_regression', 'bagged_logistic', 'mlp', 'voting_ensemble', 'stacking_ensemble'."
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
        "logistic_regression": "Logistic Regression (Scaled)",
        "logistic": "Logistic Regression (Scaled)",
        "lr": "Logistic Regression (Scaled)",
        "bagged_logistic": "Bagged Logistic Regression",
        "bagged_lr": "Bagged Logistic Regression",
        "mlp": "MLP Neural Network (Scaled)",
        "neural_network": "MLP Neural Network (Scaled)",
        "knn": "K-Nearest Neighbors (Scaled)",
        "voting_ensemble": "Soft Voting Ensemble (Tree + Linear)",
        "voting": "Soft Voting Ensemble (Tree + Linear)",
        "stacking_ensemble": "Stacking Ensemble (Tree + Linear -> LR)",
        "stacking": "Stacking Ensemble (Tree + Linear -> LR)",
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
