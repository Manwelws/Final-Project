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

    @property
    def fpr(self) -> float:
        """False positive rate: FP / (TN + FP)."""
        total_negatives = self.tn + self.fp
        return float(self.fp / total_negatives) if total_negatives > 0 else 0.0


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
    roc_auc: float | None
    pr_auc: float | None
    brier_score: float | None
    optimal_threshold: float
    cost_savings: float
    default_metrics: ThresholdMetrics
    cost_optimal_metrics: ThresholdMetrics
    report_path: str
    figures: list[str]


@dataclass(frozen=True)
class EvaluationReport:
    """Rich domain module encapsulating holdout evaluation metrics, optimization, and report rendering."""

    y_true: np.ndarray
    y_probs: np.ndarray
    roc_auc: float | None
    pr_auc: float | None
    brier_score: float | None
    optimization: OptimizationResult
    cost_matrix: CostMatrix

    @classmethod
    def compute(
        cls,
        y_true: np.ndarray | pd.Series,
        y_probs: np.ndarray | pd.Series,
        cost_matrix: CostMatrix | None = None,
        cost_fn: float = 200.0,
        cost_fp: float = 20.0,
        step: float = 0.01,
        optimal_threshold: float | None = None,
        f1_optimal_threshold: float | None = None,
    ) -> "EvaluationReport":
        """Pure in-process computation: derives all metrics and optimization without filesystem I/O."""
        if cost_matrix is None:
            cost_matrix = CostMatrix(cost_fn=cost_fn, cost_fp=cost_fp)

        y_true_arr = np.asarray(y_true).ravel()
        y_probs_arr = np.asarray(y_probs).ravel()

        if len(np.unique(y_true_arr)) < 2:
            roc_auc = None
            pr_auc = None
            brier_score = None
        else:
            roc_auc = float(roc_auc_score(y_true_arr, y_probs_arr))
            pr_auc = float(average_precision_score(y_true_arr, y_probs_arr))
            brier_score = float(brier_score_loss(y_true_arr, y_probs_arr))

        optimization = optimize_threshold(
            y_true=y_true_arr,
            y_probs=y_probs_arr,
            cost_matrix=cost_matrix,
            step=step,
            optimal_threshold=optimal_threshold,
            f1_threshold=f1_optimal_threshold,
        )

        return cls(
            y_true=y_true_arr,
            y_probs=y_probs_arr,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            brier_score=brier_score,
            optimization=optimization,
            cost_matrix=cost_matrix,
        )

    def render_plots(self, figures_dir: Path | str) -> list[str]:
        """Render diagnostic matplotlib figures to disk."""
        return render_evaluation_plots(report=self, figures_dir=figures_dir)

    def write_markdown(self, report_file: Path | str) -> None:
        """Write executive markdown comparison report."""
        generate_evaluation_report(report=self, report_file=report_file)

    def to_metadata_dict(self) -> dict[str, Any]:
        """Convert holdout metrics into a dict formatted for models/metadata.json."""
        return {
            "optimal_threshold": self.optimization.cost_optimal_threshold,
            "cost_savings": self.optimization.cost_savings,
            "cost_at_optimal": self.optimization.cost_at_optimal,
            "cost_at_default": self.optimization.cost_at_default,
            "holdout_roc_auc": self.roc_auc,
            "holdout_pr_auc": self.pr_auc,
            "holdout_brier_score": self.brier_score,
            "threshold_comparison": {
                "default_0_50": asdict(self.optimization.default_metrics),
                "f1_optimal": asdict(self.optimization.f1_optimal_metrics),
                "cost_optimal": asdict(self.optimization.cost_optimal_metrics),
            },
        }

    def to_summary(self, report_path: str = "", figures: list[str] | None = None) -> EvaluationSummary:
        """Convert to EvaluationSummary for backward compatibility."""
        return EvaluationSummary(
            roc_auc=self.roc_auc,
            pr_auc=self.pr_auc,
            brier_score=self.brier_score,
            optimal_threshold=self.optimization.cost_optimal_threshold,
            cost_savings=self.optimization.cost_savings,
            default_metrics=self.optimization.default_metrics,
            cost_optimal_metrics=self.optimization.cost_optimal_metrics,
            report_path=report_path,
            figures=figures or [],
        )


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
    optimal_threshold: float | None = None,
    f1_threshold: float | None = None,
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
    if f1_threshold is not None:
        f1_optimal = compute_metrics_at_threshold(y_true, y_probs, f1_threshold, cost_matrix=cost_matrix)
    else:
        f1_optimal = max(all_metrics, key=lambda m: (m.f1, m.recall))

    # Best Cost (minimum expected loss)
    if optimal_threshold is not None:
        cost_optimal = compute_metrics_at_threshold(y_true, y_probs, optimal_threshold, cost_matrix=cost_matrix)
    else:
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
    val_size: float = 0.10,
    random_state: int = 42,
    cost_matrix: CostMatrix | None = None,
) -> EvaluationSummary:
    """Evaluate trained model on holdout test set, optimizing decision threshold on validation and generating report."""
    if cost_matrix is None:
        cost_matrix = CostMatrix(cost_fn=cost_fn, cost_fp=cost_fp)

    data_path = Path(data_dir)
    features_path = data_path / "X_features.csv"
    target_path = data_path / "y_target.csv"

    # Step 1: Load holdout split with validation partition for threshold selection
    split = load_and_split_data(
        features_path=features_path,
        target_path=target_path,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    # Step 2: Load trained model
    m_path = Path(model_path)
    if not m_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {m_path}")
    model = joblib.load(m_path)

    # Step 3: Select decision threshold on validation set (D8 / R1-3 leak-free)
    if split.X_val is not None and split.y_val is not None:
        y_val_probs = model.predict_proba(split.X_val)[:, 1]
        val_opt = optimize_threshold(
            y_true=split.y_val.to_numpy(),
            y_probs=y_val_probs,
            cost_matrix=cost_matrix,
        )
        optimal_threshold = val_opt.cost_optimal_threshold
        f1_threshold = val_opt.f1_optimal_threshold
    else:
        optimal_threshold = None
        f1_threshold = None

    # Step 4: Evaluate holdout test set ONCE with validation-calibrated threshold
    y_test_arr = split.y_test.to_numpy()
    y_probs = model.predict_proba(split.X_test)[:, 1]

    report = EvaluationReport.compute(
        y_true=y_test_arr,
        y_probs=y_probs,
        cost_matrix=cost_matrix,
        optimal_threshold=optimal_threshold,
        f1_optimal_threshold=f1_threshold,
    )

    # Step 5: Generate Figures & Markdown Report via report methods
    rep_dir = Path(report_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = rep_dir / "figures"
    plot_files = report.render_plots(figures_dir=figures_dir)

    report_file = rep_dir / "evaluation_report.md"
    report.write_markdown(report_file=report_file)

    # Step 6: Update models/metadata.json with optimal threshold details
    meta_p = Path(metadata_path)
    if meta_p.exists():
        try:
            with open(meta_p, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    else:
        meta = {}

    meta.update(report.to_metadata_dict())

    metadata = ModelMetadata.from_dict(meta)
    metadata.save(meta_p)

    return report.to_summary(report_path=str(report_file), figures=plot_files)
