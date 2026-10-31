from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
import joblib

from final_project.data import load_and_split_data
from final_project.domain.cost import CostMatrix
from final_project.registry import ModelMetadata
from final_project.reporting import render_evaluation_plots, generate_evaluation_report

# Backward compatibility aliases for internal/external references
_generate_plots = render_evaluation_plots
_write_markdown_report = generate_evaluation_report


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    f2: float
    cost: float


@dataclass(frozen=True)
class OptimizationResult:
    default_metrics: ThresholdMetrics
    f1_optimal_metrics: ThresholdMetrics
    cost_optimal_metrics: ThresholdMetrics
    cost_optimal_threshold: float
    f1_optimal_threshold: float
    cost_at_default: float
    cost_at_optimal: float
    cost_savings: float
    thresholds_evaluated: list[ThresholdMetrics]


@dataclass(frozen=True)
class EvaluationSummary:
    roc_auc: float
    pr_auc: float
    brier_score: float
    optimal_threshold: float
    cost_savings: float
    default_metrics: ThresholdMetrics
    cost_optimal_metrics: ThresholdMetrics
    report_path: str
    figures: list[str]


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float,
    cost_fn: float = 200.0,
    cost_fp: float = 20.0,
    cost_matrix: CostMatrix | None = None,
) -> ThresholdMetrics:
    """Compute confusion matrix and business cost metrics at a specific classification threshold."""
    if cost_matrix is not None:
        matrix = cost_matrix
    else:
        matrix = CostMatrix(cost_fn=cost_fn, cost_fp=cost_fp)

    y_pred = (y_probs >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    if (precision + recall) > 0:
        f1 = float(2 * (precision * recall) / (precision + recall))
    else:
        f1 = 0.0

    # F-beta with beta=2 (weights recall 2x more than precision)
    beta_sq = 4.0
    if (beta_sq * precision + recall) > 0:
        f2 = float((1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall))
    else:
        f2 = 0.0

    cost = matrix.compute_loss(fn=fn, fp=fp)

    return ThresholdMetrics(
        threshold=float(threshold),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        f2=f2,
        cost=cost,
    )


def optimize_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    cost_fn: float = 200.0,
    cost_fp: float = 20.0,
    step: float = 0.01,
    cost_matrix: CostMatrix | None = None,
) -> OptimizationResult:
    """Sweep decision thresholds to determine the business cost-optimal and F1-optimal cutoffs."""
    if cost_matrix is None:
        cost_matrix = CostMatrix(cost_fn=cost_fn, cost_fp=cost_fp)

    thresholds = np.arange(0.01, 1.0, step)
    all_metrics = [
        compute_metrics_at_threshold(y_true, y_probs, t, cost_matrix=cost_matrix)
        for t in thresholds
    ]

    default_metrics = compute_metrics_at_threshold(
        y_true, y_probs, 0.50, cost_matrix=cost_matrix
    )

    # Best F1
    f1_optimal = max(all_metrics, key=lambda m: (m.f1, m.recall))

    # Best Cost (minimum expected loss)
    cost_optimal = min(all_metrics, key=lambda m: (m.cost, -m.recall))

    cost_at_default = default_metrics.cost
    cost_at_optimal = cost_optimal.cost
    cost_savings = cost_at_default - cost_at_optimal

    return OptimizationResult(
        default_metrics=default_metrics,
        f1_optimal_metrics=f1_optimal,
        cost_optimal_metrics=cost_optimal,
        cost_optimal_threshold=cost_optimal.threshold,
        f1_optimal_threshold=f1_optimal.threshold,
        cost_at_default=cost_at_default,
        cost_at_optimal=cost_at_optimal,
        cost_savings=cost_savings,
        thresholds_evaluated=all_metrics,
    )


def run_evaluation(
    data_dir: Path | str = "data/processed",
    model_path: Path | str = "models/model.joblib",
    metadata_path: Path | str = "models/metadata.json",
    report_dir: Path | str = "reports",
    cost_fn: float = 200.0,
    cost_fp: float = 20.0,
    test_size: float = 0.20,
    random_state: int = 42,
    cost_matrix: CostMatrix | None = None,
) -> EvaluationSummary:
    """Evaluate trained model on holdout test set, optimizing decision threshold and generating report."""
    if cost_matrix is None:
        cost_matrix = CostMatrix(cost_fn=cost_fn, cost_fp=cost_fp)

    data_path = Path(data_dir)
    features_path = data_path / "X_features.csv"
    target_path = data_path / "y_target.csv"

    # Step 1: Load holdout split
    split = load_and_split_data(
        features_path=features_path,
        target_path=target_path,
        test_size=test_size,
        random_state=random_state,
    )

    # Step 2: Load trained model
    m_path = Path(model_path)
    if not m_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {m_path}")
    model = joblib.load(m_path)

    # Step 3: Generate holdout probability predictions
    y_test_arr = split.y_test.to_numpy()
    y_probs = model.predict_proba(split.X_test)[:, 1]

    # Step 4: Compute global discrimination and calibration metrics
    roc_auc = float(roc_auc_score(y_test_arr, y_probs))
    pr_auc = float(average_precision_score(y_test_arr, y_probs))
    brier_score = float(brier_score_loss(y_test_arr, y_probs))

    # Step 5: Cost-Utility Threshold Optimization
    opt_result = optimize_threshold(
        y_true=y_test_arr,
        y_probs=y_probs,
        cost_matrix=cost_matrix,
    )

    # Step 6: Generate Figures & Markdown Report via presentation adapter
    rep_dir = Path(report_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = rep_dir / "figures"
    plot_files = render_evaluation_plots(
        y_true=y_test_arr,
        y_probs=y_probs,
        opt_result=opt_result,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        brier_score=brier_score,
        figures_dir=figures_dir,
    )

    report_file = rep_dir / "evaluation_report.md"
    summary_data = {
        "test_samples": len(split.X_test),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier_score,
    }
    generate_evaluation_report(summary_data, opt_result, report_file)

    # Step 7: Update models/metadata.json with optimal threshold details
    meta_p = Path(metadata_path)
    if meta_p.exists():
        try:
            with open(meta_p, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    else:
        meta = {}

    meta.update({
        "optimal_threshold": opt_result.cost_optimal_threshold,
        "cost_savings": opt_result.cost_savings,
        "cost_at_optimal": opt_result.cost_at_optimal,
        "cost_at_default": opt_result.cost_at_default,
        "holdout_roc_auc": roc_auc,
        "holdout_pr_auc": pr_auc,
        "holdout_brier_score": brier_score,
        "threshold_comparison": {
            "default_0_50": asdict(opt_result.default_metrics),
            "f1_optimal": asdict(opt_result.f1_optimal_metrics),
            "cost_optimal": asdict(opt_result.cost_optimal_metrics),
        },
    })

    metadata = ModelMetadata.from_dict(meta)
    metadata.save(meta_p)

    return EvaluationSummary(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        brier_score=brier_score,
        optimal_threshold=opt_result.cost_optimal_threshold,
        cost_savings=opt_result.cost_savings,
        default_metrics=opt_result.default_metrics,
        cost_optimal_metrics=opt_result.cost_optimal_metrics,
        report_path=str(report_file),
        figures=plot_files,
    )
