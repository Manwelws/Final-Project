# Production Churn Model: Holdout Evaluation Report

## 1. Executive Summary
- **Evaluated Model Algorithm**: `RandomForestClassifier`
- **Imbalance Strategy**: `unweighted`
- **Holdout Test Set Size**: 1,409 customers (20% stratified split)
- **Train / Validation Split Sizes**: 4,929 train (70%) / 705 validation (10%)
- **Holdout ROC-AUC**: `0.8436`
- **Holdout PR-AUC**: `0.6442`
- **Brier Calibration Score**: `0.1373` (lower is better, <0.15 indicates strong calibration)
- **Business Cost Savings at Optimal Cutoff**: **$21,200.00 (58.7% reduction in expected financial loss)**

---

## 2. Evaluation & Calibration Methodology
- **Data Partitioning (70/10/20 Stratified Split)**: Dataset is partitioned using a 70/10/20 stratified split into training (70%), validation (10%), and holdout test (20%) sets to preserve class balance across all evaluations.
- **Validation-Split Threshold Calibration**: Operating thresholds (max-F1 and cost-optimal cutoffs) are calibrated leak-free on the 10% validation split using a model trained exclusively on the 70% training split, never on post-refit in-sample predictions or the holdout test set.
- **Train+Val Refit & Final Scoring**: The final production model is refitted on the combined train+val data (80%) and evaluated once against the untouched 20% holdout test split using the validation-calibrated decision threshold.

---

## 3. Decision Threshold Comparison & Cost-Utility Threshold Optimization

Business Loss Formula: $\text{Loss} = C_{\text{FN}} \times \text{FN} + C_{\text{FP}} \times \text{FP}$  
Assumed Matrix: $C_{\text{FN}} = \$200$ (missed churner customer lifetime loss) vs. $C_{\text{FP}} = \$20$ (cost of customer retention outreach).

| Decision Strategy | Threshold | Precision | Recall | F1 Score | F2 Score | False Negatives | False Positives | Expected Loss ($) |
|---|---|---|---|---|---|---|---|---|
| **Default Threshold** | `0.50` | `0.641` | `0.548` | `0.591` | `0.564` | `169` | `115` | **$36,100** |
| **Max F1 Threshold** | `0.43` | `0.608` | `0.650` | `0.628` | `0.641` | `131` | `157` | **$29,340** |
| **Cost-Optimal Threshold** | **`0.10`** | `0.401` | **`0.941`** | `0.563` | **`0.742`** | **`22`** | `525` | **$14,900** |

---

## 4. Key Observations & Operational Recommendations
1. **Financial Leverage**: Shifting the decision cutoff from `0.50` down to **`0.10`** captures **147 additional at-risk churners**, reducing missed churn loss from $33,800 to $4,400.
2. **Acceptable Outreach Volume**: While false alarms increase by 410 customers, each outreach costs only $20, yielding a net business savings of **$21,200.00**.
3. **Serving Recommendation**: Production scoring should apply the decision threshold `threshold=0.10` when routing customers to the retention team.

---

## 5. Diagnostic Visualizations
- **ROC Curve**: `figures/roc_curve.png`
- **Precision-Recall Curve**: `figures/pr_curve.png`
- **Calibration Diagram**: `figures/calibration_curve.png`
- **Cost Curve**: `figures/cost_curve.png`
- **Confusion Matrices**: `figures/confusion_matrix.png`
- **SHAP Grouped Importance Summary**: `figures/shap_summary_grouped.png`
- **SHAP Detailed Beeswarm Summary**: `figures/shap_summary_detailed.png`
- **SHAP Dependence (Monthly Charges)**: `figures/shap_dependence_monthly_charges.png`
- **SHAP Dependence (Tenure Months)**: `figures/shap_dependence_tenure.png`
- **SHAP Local Waterfall (Churned Customer)**: `figures/shap_waterfall_churner.png`
- **SHAP Local Waterfall (Retained / False Positive Customer)**: `figures/shap_waterfall_retained.png`
