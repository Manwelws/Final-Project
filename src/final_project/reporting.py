"""Reporting and visualization adapters for evaluation reports and diagnostic plots."""

from pathlib import Path
from typing import Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve


def render_evaluation_plots(
    y_true: np.ndarray | Any = None,
    y_probs: np.ndarray | None = None,
    opt_result: Any = None,
    roc_auc: float | None = None,
    pr_auc: float | None = None,
    brier_score: float | None = None,
    figures_dir: Path | str | None = None,
    *,
    report: Any = None,
) -> list[str]:
    """Generate and save publication-quality diagnostic plots.

    Accepts either an EvaluationReport directly (via report or first argument),
    or individual metric parameters for backwards compatibility.
    """
    # Check if first arg is an EvaluationReport
    if report is None and hasattr(y_true, "optimization") and hasattr(y_true, "roc_auc"):
        report = y_true
        if figures_dir is None and y_probs is not None:
            figures_dir = y_probs

    if report is not None:
        y_true = report.y_true
        y_probs = report.y_probs
        opt_result = report.optimization
        roc_auc = report.roc_auc
        pr_auc = report.pr_auc
        brier_score = report.brier_score

    if figures_dir is None:
        raise ValueError("figures_dir must be specified")

    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
    roc_path = out_dir / "roc_curve.png"
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
    pr_path = out_dir / "pr_curve.png"
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
    calib_path = out_dir / "calibration_curve.png"
    fig.tight_layout()
    fig.savefig(calib_path, dpi=150)
    plt.close(fig)
    plot_files.append(str(calib_path))

    # 4. Cost Curve Across Thresholds
    fig, ax = plt.subplots(figsize=(7, 5))
    thresh_vals = [m.threshold for m in opt_result.thresholds_evaluated]
    cost_vals = [m.cost for m in opt_result.thresholds_evaluated]
    ax.plot(thresh_vals, cost_vals, color="#9467bd", lw=2, label="Expected Business Loss ($)")
    ax.axvline(
        opt_result.cost_optimal_threshold,
        color="red",
        linestyle="--",
        label=f"Optimal ({opt_result.cost_optimal_threshold:.2f}, ${opt_result.cost_at_optimal:,.0f})",
    )
    ax.axvline(0.50, color="gray", linestyle=":", label=f"Default 0.50 (${opt_result.cost_at_default:,.0f})")
    ax.set_xlabel("Classification Decision Threshold")
    ax.set_ylabel("Total Financial Loss ($)")
    ax.set_title("Business Cost-Utility Optimization Curve ($C_{FN}=$200, $C_{FP}=$20)")
    ax.legend()
    ax.grid(alpha=0.3)
    cost_path = out_dir / "cost_curve.png"
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
    cm_path = out_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    plot_files.append(str(cm_path))

    return plot_files


def generate_evaluation_report(
    summary_data: dict[str, Any] | Any = None,
    opt_result: Any = None,
    report_file: Path | str | None = None,
    *,
    report: Any = None,
) -> None:
    """Generate executive markdown evaluation report with formatted comparison tables.

    Accepts either an EvaluationReport directly (via report or first argument),
    or summary_data dict + opt_result for backwards compatibility.
    """
    if report is None and hasattr(summary_data, "optimization") and hasattr(summary_data, "roc_auc"):
        report = summary_data
        if report_file is None and opt_result is not None:
            report_file = opt_result

    if report is not None:
        opt_result = report.optimization
        summary_data = {
            "test_samples": len(report.y_true),
            "roc_auc": report.roc_auc,
            "pr_auc": report.pr_auc,
            "brier_score": report.brier_score,
        }

    if report_file is None:
        raise ValueError("report_file must be specified")

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
    out_file = Path(report_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(content)
