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


@dataclass(frozen=True)
class AlgorithmSpec:
    canonical_name: str
    aliases: tuple[str, ...]
    display_name: str
    supports_class_weighting: bool = True


CANDIDATE_ALGORITHM_SPECS: tuple[AlgorithmSpec, ...] = (
    AlgorithmSpec(
        canonical_name="random_forest",
        aliases=("rf",),
        display_name="Random Forest",
    ),
    AlgorithmSpec(
        canonical_name="lightgbm",
        aliases=("lgbm", "lgb"),
        display_name="LightGBM",
    ),
    AlgorithmSpec(
        canonical_name="xgboost",
        aliases=("xgb",),
        display_name="XGBoost",
    ),
    AlgorithmSpec(
        canonical_name="logistic_regression",
        aliases=("lr", "logistic"),
        display_name="Logistic Regression (Scaled)",
    ),
    AlgorithmSpec(
        canonical_name="bagged_logistic",
        aliases=("bagged_lr",),
        display_name="Bagged Logistic Regression",
    ),
    AlgorithmSpec(
        canonical_name="mlp",
        aliases=("neural_network", "mlp_classifier"),
        display_name="MLP Neural Network (Scaled)",
    ),
    AlgorithmSpec(
        canonical_name="knn",
        aliases=("kneighbors",),
        display_name="K-Nearest Neighbors (Scaled)",
        supports_class_weighting=False,
    ),
    AlgorithmSpec(
        canonical_name="voting_ensemble",
        aliases=("voting",),
        display_name="Soft Voting Ensemble (Tree + Linear)",
    ),
    AlgorithmSpec(
        canonical_name="stacking_ensemble",
        aliases=("stacking",),
        display_name="Stacking Ensemble (Tree + Linear -> LR)",
    ),
)

_SPEC_BY_NAME: dict[str, AlgorithmSpec] = {
    alias: spec
    for spec in CANDIDATE_ALGORITHM_SPECS
    for alias in (spec.canonical_name, *spec.aliases)
}

CLI_ALGORITHM_CHOICES: list[str] = [
    name
    for spec in CANDIDATE_ALGORITHM_SPECS
    for name in (spec.canonical_name, *spec.aliases)
]

BENCHMARK_ALL_ALGORITHMS: list[str] = [
    spec.canonical_name for spec in CANDIDATE_ALGORITHM_SPECS
]

SUPPORTED_ALGORITHMS_ERROR_LIST: str = ", ".join(
    f"'{spec.canonical_name}'" for spec in CANDIDATE_ALGORITHM_SPECS
)


def _normalize_key(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def normalize_algorithm_name(name: str) -> str:
    """Normalize an algorithm name or alias to its canonical identifier."""
    key = _normalize_key(name)
    spec = _SPEC_BY_NAME.get(key)
    return spec.canonical_name if spec is not None else key


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Compute negative / positive ratio for XGBoost scale_pos_weight balancing."""
    counts = y.value_counts()
    neg_count = counts.get(0, 0)
    pos_count = counts.get(1, 0)
    if pos_count == 0:
        return 1.0
    return float(neg_count / pos_count)


def supports_class_weighting(name: str) -> bool:
    """Return whether the given candidate algorithm accepts class weighting."""
    canonical_name = normalize_algorithm_name(name)
    spec = _SPEC_BY_NAME.get(canonical_name)
    return spec.supports_class_weighting if spec is not None else True


def get_candidate_model(
    name: str,
    balanced: bool = False,
    scale_pos_weight: float | None = None,
    random_state: int = 42,
    **kwargs: Any,
) -> BaseEstimator:
    """Instantiate a tree-based, linear, neural, or ensemble candidate classifier with class weighting."""
    canonical_name = normalize_algorithm_name(name)

    if canonical_name == "random_forest":
        class_weight = "balanced" if balanced else None
        return RandomForestClassifier(
            random_state=random_state,
            class_weight=class_weight,
            **kwargs,
        )

    elif canonical_name == "lightgbm":
        class_weight = "balanced" if balanced else None
        return LGBMClassifier(
            random_state=random_state,
            class_weight=class_weight,
            verbose=-1,
            **kwargs,
        )

    elif canonical_name == "xgboost":
        weight = scale_pos_weight if (balanced and scale_pos_weight is not None) else 1.0
        return XGBClassifier(
            random_state=random_state,
            scale_pos_weight=weight,
            eval_metric="logloss",
            **kwargs,
        )

    elif canonical_name == "logistic_regression":
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

    elif canonical_name == "bagged_logistic":
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

    elif canonical_name == "mlp":
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

    elif canonical_name == "knn":
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

    elif canonical_name == "voting_ensemble":
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

    elif canonical_name == "stacking_ensemble":
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
            f"Unsupported algorithm '{name}'. Choose from {SUPPORTED_ALGORITHMS_ERROR_LIST}."
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
        spec.canonical_name: spec.display_name
        for spec in CANDIDATE_ALGORITHM_SPECS
    }

    records = []
    for algo in algorithms:
        canonical_name = normalize_algorithm_name(algo)
        d_name = display_names.get(canonical_name, algo)
        weighting_modes = [False, True] if supports_class_weighting(canonical_name) else [False]
        for balanced in weighting_modes:
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
