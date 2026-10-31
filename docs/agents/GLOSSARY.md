# Churn Prediction System

A production-grade customer churn prediction and automated retraining system.

## Language

**Churn**:
A customer leaving the service. Binary outcome: churned (1) or retained (0).
_Avoid_: Attrition, turnover, cancellation

**Churner**:
A customer who has churned.
_Avoid_: Lost customer, defector

**Covariate Drift**:
A shift in the distribution of input features (e.g. Monthly Charges rising) without a change in the true relationship between features and the target. Measured via Population Stability Index (PSI).
_Avoid_: Feature drift, data drift (too vague)

**Concept Drift**:
A change in the true relationship between features and the target (e.g. high-charge customers who used to retain now churn).
_Avoid_: Model decay (describes the symptom, not the cause)

**Population Stability Index (PSI)**:
Statistical metric measuring shift in feature distributions between a reference baseline and incoming weekly batches. Values < 0.1 indicate stability; 0.1–0.25 moderate shift; > 0.25 significant shift.
_Avoid_: KS drift score, shift index

**Label Lag**:
The delay between when features are observed and when the true churn outcome is known. In this system, 2 weeks.
_Avoid_: Label delay, feedback delay

**Dual Monitoring & Retraining Architecture**:
A two-tiered monitoring strategy to handle label lag:
1. **Unsupervised Drift Alert (Immediate)**: PSI computed on incoming Week $N$ unlabeled features for early warning.
2. **Supervised Retraining Trigger (Delayed)**: Evaluated on Week $N+2$ ground truth labels when rolling live AUC drops below historical best minus a threshold (0.05).
_Avoid_: Drift trigger without labels, single-tier monitoring

**Exponential Sample Decay**:
The retraining data weighting scheme that includes historical data but downweights older weeks by a discount factor (e.g. $0.85^{\text{weeks\_old}}$), ensuring post-drift data guides model updates without discarding sample volume.
_Avoid_: Expanding window (dilutes drift adaptation), sliding window (drastically reduces sample size)

**Cost-Utility Threshold Optimization**:
Selecting the probability cutoff that minimizes total expected business loss ($C_{\text{FN}} \cdot \text{FN} + C_{\text{FP}} \cdot \text{FP}$), where missed churners carry substantially higher penalty than unnecessary retention offers.
_Avoid_: Symmetric F1 cutoff, arbitrary 0.5 default

**Reference Data**:
A static pre-drift holdout dataset used to evaluate the never-retrain baseline. Not used for training.
_Avoid_: Baseline data, test set (ambiguous)

**Imbalance Strategy**:
The technique used to handle class imbalance in training data. Restricted to algorithmic class weighting (`class_weight='balanced'`, `scale_pos_weight`) to preserve binary one-hot integrity and probability calibration.
_Avoid_: SMOTE (interpolates invalid fractional categories on one-hot features)

**Trial**:
A single Optuna hyperparameter evaluation: one set of hyperparameters, trained and scored via cross-validation.
_Avoid_: Run, iteration, experiment (overloaded with MLflow)

**Experiment**:
An MLflow experiment grouping related training runs (e.g. "churn-tree-tuning").
_Avoid_: Trial (that's an Optuna concept), study (that's also Optuna)
