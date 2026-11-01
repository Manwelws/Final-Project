import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import mlflow

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

from final_project.models import get_candidate_model, compute_scale_pos_weight


@dataclass(frozen=True)
class TuningResult:
    best_model: BaseEstimator
    best_algorithm: str
    best_params: dict[str, Any]
    best_roc_auc: float
    study: optuna.Study
    experiment_id: str


def _build_model_from_trial(
    trial: optuna.Trial,
    scale_pos_weight: float,
    random_state: int,
    algorithms: list[str] | None = None,
) -> tuple[str, bool, dict[str, Any], BaseEstimator]:
    """Sample hyperparameters for a candidate architecture from Optuna trial."""
    candidate_algos = algorithms or ["random_forest", "lightgbm", "xgboost"]
    algorithm = trial.suggest_categorical("algorithm", candidate_algos)
    balanced = trial.suggest_categorical("balanced", [True, False])

    params: dict[str, Any] = {"algorithm": algorithm, "balanced": balanced}

    if algorithm == "random_forest":
        n_estimators = trial.suggest_int("rf_n_estimators", 50, 300, step=50)
        max_depth = trial.suggest_int("rf_max_depth", 4, 20)
        min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 1, 6)
        params.update({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
        })
        model = get_candidate_model(
            "random_forest",
            balanced=balanced,
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )

    elif algorithm == "lightgbm":
        n_estimators = trial.suggest_int("lgb_n_estimators", 50, 300, step=50)
        learning_rate = trial.suggest_float("lgb_learning_rate", 0.01, 0.2, log=True)
        num_leaves = trial.suggest_int("lgb_num_leaves", 15, 63)
        max_depth = trial.suggest_int("lgb_max_depth", 3, 12)
        reg_alpha = trial.suggest_float("lgb_reg_alpha", 1e-3, 10.0, log=True)
        reg_lambda = trial.suggest_float("lgb_reg_lambda", 1e-3, 10.0, log=True)
        params.update({
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
        })
        model = get_candidate_model(
            "lightgbm",
            balanced=balanced,
            random_state=random_state,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
        )

    elif algorithm == "xgboost":
        n_estimators = trial.suggest_int("xgb_n_estimators", 50, 300, step=50)
        learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.2, log=True)
        max_depth = trial.suggest_int("xgb_max_depth", 3, 10)
        subsample = trial.suggest_float("xgb_subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0)
        reg_alpha = trial.suggest_float("xgb_reg_alpha", 1e-3, 10.0, log=True)
        reg_lambda = trial.suggest_float("xgb_reg_lambda", 1e-3, 10.0, log=True)
        params.update({
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
        })
        model = get_candidate_model(
            "xgboost",
            balanced=balanced,
            scale_pos_weight=scale_pos_weight if balanced else None,
            random_state=random_state,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
        )

    elif algorithm in ("logistic_regression", "lr", "logistic"):
        c_val = trial.suggest_float("lr_c", 1e-3, 10.0, log=True)
        params.update({"C": c_val})
        model = get_candidate_model(
            "logistic_regression",
            balanced=balanced,
            random_state=random_state,
            C=c_val,
        )

    elif algorithm in ("bagged_logistic", "bagged_lr"):
        n_estimators = trial.suggest_int("bagged_n_estimators", 15, 60, step=15)
        params.update({"n_estimators": n_estimators})
        model = get_candidate_model(
            "bagged_logistic",
            balanced=balanced,
            random_state=random_state,
            n_estimators=n_estimators,
        )

    elif algorithm in ("mlp", "neural_network", "mlp_classifier"):
        alpha = trial.suggest_float("mlp_alpha", 1e-4, 1e-1, log=True)
        params.update({"alpha": alpha})
        model = get_candidate_model(
            "mlp",
            balanced=balanced,
            random_state=random_state,
            alpha=alpha,
        )

    elif algorithm in ("voting_ensemble", "voting"):
        params.update({"voting": "soft"})
        model = get_candidate_model(
            "voting_ensemble",
            balanced=balanced,
            random_state=random_state,
        )

    elif algorithm in ("stacking_ensemble", "stacking"):
        params.update({"meta_classifier": "LogisticRegression"})
        model = get_candidate_model(
            "stacking_ensemble",
            balanced=balanced,
            random_state=random_state,
        )

    else:
        model = get_candidate_model(
            algorithm,
            balanced=balanced,
            scale_pos_weight=scale_pos_weight if balanced else None,
            random_state=random_state,
        )

    return algorithm, balanced, params, model


def run_tuning_study(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
    cv_folds: int = 5,
    experiment_name: str = "churn_retraining_optimization",
    tracking_uri: str = "sqlite:///mlflow.db",
    random_state: int = 42,
    algorithms: list[str] | None = None,
) -> TuningResult:
    """Run a unified Optuna Bayesian optimization study across candidate models with MLflow logging."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Set up MLflow
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    exp_id = experiment.experiment_id

    scale_pos_weight = compute_scale_pos_weight(y_train)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        algorithm, balanced, params, model = _build_model_from_trial(
            trial=trial,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            algorithms=algorithms,
        )

        fold_roc_scores = []
        fold_pr_scores = []

        with mlflow.start_run(
            experiment_id=exp_id,
            run_name=f"trial_{trial.number}_{algorithm}",
            nested=True,
        ):
            # Log params
            mlflow.log_params(params)
            mlflow.set_tag("algorithm", algorithm)
            mlflow.set_tag("balanced", str(balanced))
            mlflow.set_tag("trial_number", str(trial.number))

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model.fit(X_tr, y_tr)
                y_val_prob = model.predict_proba(X_val)[:, 1]

                roc_score = float(roc_auc_score(y_val, y_val_prob))
                pr_score = float(average_precision_score(y_val, y_val_prob))

                fold_roc_scores.append(roc_score)
                fold_pr_scores.append(pr_score)

                # Report intermediate mean ROC-AUC for Optuna pruning
                intermediate_mean = float(np.mean(fold_roc_scores))
                trial.report(intermediate_mean, step=fold_idx)

                if trial.should_prune():
                    mlflow.set_tag("status", "PRUNED")
                    mlflow.log_metric("cv_roc_auc", intermediate_mean)
                    raise optuna.TrialPruned()

            mean_roc = float(np.mean(fold_roc_scores))
            std_roc = float(np.std(fold_roc_scores))
            mean_pr = float(np.mean(fold_pr_scores))
            std_pr = float(np.std(fold_pr_scores))

            mlflow.log_metrics({
                "cv_roc_auc": mean_roc,
                "cv_roc_auc_std": std_roc,
                "cv_pr_auc": mean_pr,
                "cv_pr_auc_std": std_pr,
            })
            mlflow.set_tag("status", "COMPLETED")

            return mean_roc

    sampler = TPESampler(seed=random_state)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"study_{experiment_name}_{random_state}",
    )

    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    best_algorithm = best_trial.params["algorithm"]
    best_balanced = best_trial.params["balanced"]

    # Reconstruct winning model and fit on complete training dataset
    _, _, best_clean_params, best_estimator = _build_model_from_trial(
        trial=best_trial,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        algorithms=algorithms,
    )
    best_estimator.fit(X_train, y_train)

    return TuningResult(
        best_model=best_estimator,
        best_algorithm=best_algorithm,
        best_params=best_clean_params,
        best_roc_auc=float(best_trial.value),
        study=study,
        experiment_id=exp_id,
    )
