from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve
import joblib

from final_project.data import load_and_split_data


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
) -> ThresholdMetrics:
    """Compute confusion matrix and business cost metrics at a specific classification threshold."""
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

    cost = float(cost_fn * fn + cost_fp * fp)

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
) -> OptimizationResult:
    """Sweep decision thresholds to determine the business cost-optimal and F1-optimal cutoffs."""
    thresholds = np.arange(0.01, 1.0, step)
    all_metrics = [
        compute_metrics_at_threshold(y_true, y_probs, t, cost_fn=cost_fn, cost_fp=cost_fp)
        for t in thresholds
    ]

    default_metrics = compute_metrics_at_threshold(
        y_true, y_probs, 0.50, cost_fn=cost_fn, cost_fp=cost_fp
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


def _generate_plots(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    opt_result: OptimizationResult,
    roc_auc: float,
    pr_auc: float,
    brier_score: float,
    figures_dir: Path,
) -> list[str]:
    """Generate and save publication-quality diagnostic plots."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_files = []

    # 1. ROC Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (Holdout Test)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    roc_path = figures_dir / "roc_curve.png"
    fig.tight_layout()
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    plot_files.append(str(roc_path))

    # 2. PR Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_probs)
    baseline_rate = float(np.mean(y_true))
    ax.plot(recall_vals, precision_vals, color="#2ca02c", lw=2, label=f"PR Curve (AUC = {pr_auc:.4f})")
    ax.axhline(baseline_rate, color="gray", lw=1, linestyle="--", label=f"Baseline ({baseline_rate:.2%})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Holdout Test)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    pr_path = figures_dir / "pr_curve.png"
    fig.tight_layout()
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)
    plot_files.append(str(pr_path))

    # 3. Calibration Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
    ax.plot(prob_pred, prob_true, marker="s", color="#d62728", lw=2, label=f"Model (Brier = {brier_score:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Probability Calibration Curve (Reliability Diagram)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    calib_path = figures_dir / "calibration_curve.png"
    fig.tight_layout()
    fig.savefig(calib_path, dpi=150)
    plt.close(fig)
    plot_files.append(str(calib_path))

    # 4. Cost Curve Across Thresholds
    fig, ax = plt.subplots(figsize=(7, 5))
    thresh_vals = [m.threshold for m in opt_result.thresholds_evaluated]
    cost_vals = [m.cost for m in opt_result.thresholds_evaluated]
    ax.plot(thresh_vals, cost_vals, color="#9467bd", lw=2, label="Expected Business Loss ($)")
    ax.axvline(opt_result.cost_optimal_threshold, color="red", linestyle="--", label=f"Optimal ({opt_result.cost_optimal_threshold:.2f}, ${opt_result.cost_at_optimal:,.0f})")
    ax.axvline(0.50, color="gray", linestyle=":", label=f"Default 0.50 (${opt_result.cost_at_default:,.0f})")
    ax.set_xlabel("Classification Decision Threshold")
    ax.set_ylabel("Total Financial Loss ($)")
    ax.set_title("Business Cost-Utility Optimization Curve ($C_{FN}=$200, $C_{FP}=$20)")
    ax.legend()
    ax.grid(alpha=0.3)
    cost_path = figures_dir / "cost_curve.png"
    fig.tight_layout()
    fig.savefig(cost_path, dpi=150)
    plt.close(fig)
    plot_files.append(str(cost_path))

    # 5. Confusion Matrices Comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    configs = [
        ("Default (0.50)", opt_result.default_metrics),
        (f"Max F1 ({opt_result.f1_optimal_threshold:.2f})", opt_result.f1_optimal_metrics),
        (f"Cost-Optimal ({opt_result.cost_optimal_threshold:.2f})", opt_result.cost_optimal_metrics),
    ]
    for ax, (title, m) in zip(axes, configs):
        matrix = np.array([[m.tn, m.fp], [m.fn, m.tp]])
        cax = ax.matshow(matrix, cmap="Blues", alpha=0.8)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", color="black", fontsize=12)
        ax.set_title(title, pad=12, fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1], ["Retain", "Churn"])
        ax.set_yticks([0, 1], ["Retain", "Churn"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.tight_layout()
    cm_path = figures_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    plot_files.append(str(cm_path))

    return plot_files


def _write_markdown_report(
    summary_data: dict[str, Any],
    opt_result: OptimizationResult,
    report_file: Path,
) -> None:
    """Generate executive markdown evaluation report with formatted comparison tables."""
    def_m = opt_result.default_metrics
    f1_m = opt_result.f1_optimal_metrics
    cost_m = opt_result.cost_optimal_metrics

    savings_pct = (
        (opt_result.cost_savings / opt_result.cost_at_default) * 100
        if opt_result.cost_at_default > 0
        else 0.0
    )

    content = f"""# Production Churn Model: Holdout Evaluation Report

## 1. Executive Summary
- **Holdout Test Set Size**: {summary_data['test_samples']:,} customers (20% stratified split)
- **Holdout ROC-AUC**: `{summary_data['roc_auc']:.4f}`
- **Holdout PR-AUC**: `{summary_data['pr_auc']:.4f}`
- **Brier Calibration Score**: `{summary_data['brier_score']:.4f}` (lower is better, <0.15 indicates strong calibration)
- **Business Cost Savings at Optimal Cutoff**: **${opt_result.cost_savings:,.2f} ({savings_pct:.1f}% reduction in expected financial loss)**

---

## 2. Decision Threshold Comparison & Cost-Utility Threshold Optimization

Business Loss Formula: $\\text{{Loss}} = C_{{\\text{{FN}}}} \\times \\text{{FN}} + C_{{\\text{{FP}}}} \\times \\text{{FP}}$  
Assumed Matrix: $C_{{\\text{{FN}}}} = \\$200$ (missed churner customer lifetime loss) vs. $C_{{\\text{{FP}}}} = \\$20$ (cost of customer retention outreach).

| Decision Strategy | Threshold | Precision | Recall | F1 Score | F2 Score | False Negatives | False Positives | Expected Loss ($) |
|---|---|---|---|---|---|---|---|---|
| **Default Threshold** | `0.50` | `{def_m.precision:.3f}` | `{def_m.recall:.3f}` | `{def_m.f1:.3f}` | `{def_m.f2:.3f}` | `{def_m.fn:,}` | `{def_m.fp:,}` | **${def_m.cost:,.0f}** |
| **Max F1 Threshold** | `{f1_m.threshold:.2f}` | `{f1_m.precision:.3f}` | `{f1_m.recall:.3f}` | `{f1_m.f1:.3f}` | `{f1_m.f2:.3f}` | `{f1_m.fn:,}` | `{f1_m.fp:,}` | **${f1_m.cost:,.0f}** |
| **Cost-Optimal Threshold** | **`{cost_m.threshold:.2f}`** | `{cost_m.precision:.3f}` | **`{cost_m.recall:.3f}`** | `{cost_m.f1:.3f}` | **`{cost_m.f2:.3f}`** | **`{cost_m.fn:,}`** | `{cost_m.fp:,}` | **${cost_m.cost:,.0f}** |

---

## 3. Key Observations & Operational Recommendations
1. **Financial Leverage**: Shifting the decision cutoff from `0.50` down to **`{cost_m.threshold:.2f}`** captures **{def_m.fn - cost_m.fn} additional at-risk churners**, reducing missed churn loss from ${def_m.fn * 200:,.0f} to ${cost_m.fn * 200:,.0f}.
2. **Acceptable Outreach Volume**: While false alarms increase by {cost_m.fp - def_m.fp:,} customers, each outreach costs only $20, yielding a net business savings of **${opt_result.cost_savings:,.2f}**.
3. **Serving Recommendation**: Production scoring should apply the decision threshold `threshold={cost_m.threshold:.2f}` when routing customers to the retention team.

---

## 4. Diagnostic Visualizations
- **ROC Curve**: `figures/roc_curve.png`
- **Precision-Recall Curve**: `figures/pr_curve.png`
- **Calibration Diagram**: `figures/calibration_curve.png`
- **Cost Curve**: `figures/cost_curve.png`
- **Confusion Matrices**: `figures/confusion_matrix.png`
"""
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(content)


def run_evaluation(
    data_dir: Path | str = "data/processed",
    model_path: Path | str = "models/model.joblib",
    metadata_path: Path | str = "models/metadata.json",
    report_dir: Path | str = "reports",
    cost_fn: float = 200.0,
    cost_fp: float = 20.0,
    test_size: float = 0.20,
    random_state: int = 42,
) -> EvaluationSummary:
    """Evaluate trained model on holdout test set, optimizing decision threshold and generating report."""
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
        cost_fn=cost_fn,
        cost_fp=cost_fp,
    )

    # Step 6: Generate Figures & Markdown Report
    rep_dir = Path(report_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = rep_dir / "figures"
    plot_files = _generate_plots(
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
    _write_markdown_report(summary_data, opt_result, report_file)

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

    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

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
