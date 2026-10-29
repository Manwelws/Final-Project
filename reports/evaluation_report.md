# Production Churn Model: Holdout Evaluation Report

## 1. Executive Summary
- **Holdout Test Set Size**: 1,409 customers (20% stratified split)
- **Holdout ROC-AUC**: `0.8390`
- **Holdout PR-AUC**: `0.6276`
- **Brier Calibration Score**: `0.1392` (lower is better, <0.15 indicates strong calibration)
- **Business Cost Savings at Optimal Cutoff**: **$21,860.00 (59.5% reduction in expected financial loss)**

---

## 2. Decision Threshold Comparison & Cost-Utility Threshold Optimization

Business Loss Formula: $\text{Loss} = C_{\text{FN}} \times \text{FN} + C_{\text{FP}} \times \text{FP}$  
Assumed Matrix: $C_{\text{FN}} = \$200$ (missed churner customer lifetime loss) vs. $C_{\text{FP}} = \$20$ (cost of customer retention outreach).

| Decision Strategy | Threshold | Precision | Recall | F1 Score | F2 Score | False Negatives | False Positives | Expected Loss ($) |
|---|---|---|---|---|---|---|---|---|
| **Default Threshold** | `0.50` | `0.631` | `0.540` | `0.582` | `0.556` | `172` | `118` | **$36,760** |
| **Max F1 Threshold** | `0.38` | `0.581` | `0.693` | `0.632` | `0.667` | `115` | `187` | **$26,740** |
| **Cost-Optimal Threshold** | **`0.12`** | `0.418` | **`0.930`** | `0.577` | **`0.747`** | **`26`** | `485` | **$14,900** |

---

## 3. Key Observations & Operational Recommendations
1. **Financial Leverage**: Shifting the decision cutoff from `0.50` down to **`0.12`** captures **146 additional at-risk churners**, reducing missed churn loss from $34,400 to $5,200.
2. **Acceptable Outreach Volume**: While false alarms increase by 367 customers, each outreach costs only $20, yielding a net business savings of **$21,860.00**.
3. **Serving Recommendation**: Production scoring should apply the decision threshold `threshold=0.12` when routing customers to the retention team.

---

## 4. Diagnostic Visualizations
- **ROC Curve**: `figures/roc_curve.png`
- **Precision-Recall Curve**: `figures/pr_curve.png`
- **Calibration Diagram**: `figures/calibration_curve.png`
- **Cost Curve**: `figures/cost_curve.png`
- **Confusion Matrices**: `figures/confusion_matrix.png`
