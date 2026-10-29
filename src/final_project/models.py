from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


@dataclass(frozen=True)
class TrainingResult:
    model: RandomForestClassifier
    cv_auc_mean: float
    cv_auc_std: float
    cv_fold_scores: list[float]
    feature_names: list[str]


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42,
) -> TrainingResult:
    """Train a baseline Random Forest classifier and evaluate 5-fold CV ROC-AUC."""
    feature_names = list(X_train.columns)
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    rf = RandomForestClassifier(random_state=random_state)
    
    # Compute cross-validation ROC-AUC scores
    scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring="roc_auc")
    fold_scores = [float(s) for s in scores]
    cv_auc_mean = float(np.mean(fold_scores))
    cv_auc_std = float(np.std(fold_scores))
    
    # Fit final estimator on all training data
    rf.fit(X_train, y_train)
    
    return TrainingResult(
        model=rf,
        cv_auc_mean=cv_auc_mean,
        cv_auc_std=cv_auc_std,
        cv_fold_scores=fold_scores,
        feature_names=feature_names,
    )
