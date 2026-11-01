# Production Churn Model: Holdout Evaluation Report

## 1. Executive Summary
- **Holdout Test Set Size**: 1,409 customers (20% stratified split)
- **Holdout ROC-AUC**: `0.8436`
- **Holdout PR-AUC**: `0.6442`
- **Brier Calibration Score**: `0.1373` (lower is better, <0.15 indicates strong calibration)
- **Business Cost Savings at Optimal Cutoff**: **$10,600.00 (29.4% reduction in expected financial loss)**

---

## 2. Decision Threshold Comparison & Cost-Utility Threshold Optimization

Business Loss Formula: $\text{Loss} = C_{\text{FN}} \times \text{FN} + C_{\text{FP}} \times \text{FP}$  
Assumed Matrix: $C_{\text{FN}} = \$200$ (missed churner customer lifetime loss) vs. $C_{\text{FP}} = \$20$ (cost of customer retention outreach).

| Decision Strategy | Threshold | Precision | Recall | F1 Score | F2 Score | False Negatives | False Positives | Expected Loss ($) |
|---|---|---|---|---|---|---|---|---|
| **Default Threshold** | `0.50` | `0.641` | `0.548` | `0.591` | `0.564` | `169` | `115` | **$36,100** |
| **Max F1 Threshold** | `0.36` | `0.577` | `0.711` | `0.637` | `0.680` | `108` | `195` | **$25,500** |
| **Cost-Optimal Threshold** | **`0.36`** | `0.577` | **`0.711`** | `0.637` | **`0.680`** | **`108`** | `195` | **$25,500** |

---

## 3. Key Observations & Operational Recommendations
1. **Financial Leverage**: Shifting the decision cutoff from `0.50` down to **`0.36`** captures **61 additional at-risk churners**, reducing missed churn loss from $33,800 to $21,600.
2. **Acceptable Outreach Volume**: While false alarms increase by 80 customers, each outreach costs only $20, yielding a net business savings of **$10,600.00**.
3. **Serving Recommendation**: Production scoring should apply the decision threshold `threshold=0.36` when routing customers to the retention team.

---

## 4. Diagnostic Visualizations
- **ROC Curve**: `figures/roc_curve.png`
- **Precision-Recall Curve**: `figures/pr_curve.png`
- **Calibration Diagram**: `figures/calibration_curve.png`
- **Cost Curve**: `figures/cost_curve.png`
- **Confusion Matrices**: `figures/confusion_matrix.png`
