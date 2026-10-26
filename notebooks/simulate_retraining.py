#!/usr/bin/env python3
"""
Churn Retraining Simulator
==========================

Simulates a live weekly streaming environment where:
  * Features arrive every week (available immediately for scoring)
  * Labels (actual churn outcomes) are released after a fixed delay
  * Covariate + concept drift start at week 5
  * A performance-decay trigger decides when to retrain
  * A never-retrain baseline is run for comparison

Outputs:
  data/live/features/week_N_features.csv   - features available at week N
  data/live/labels/week_N_labels.csv       - labels released at week N+delay
  data/live/pending/                       - features awaiting labels
  data/reference/                          - static pre-drift holdout
  data/retraining/week_N/                  - metrics when retrained
  data/drift/drift_metrics.csv             - AUC per week for both strategies

Every design decision is documented inline:
  DECISION / ALTERNATIVES CONSIDERED / WHY NOT PICKED
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

# =============================================================================
# PATHS
# =============================================================================
PROCESSED_DIR = Path("data/processed")
LIVE_FEATURES_DIR = Path("data/live/features")
LIVE_LABELS_DIR = Path("data/live/labels")
LIVE_PENDING_DIR = Path("data/live/pending")
REFERENCE_DIR = Path("data/reference")
RETRAINING_DIR = Path("data/retraining")
DRIFT_DIR = Path("data/drift")

for d in [LIVE_FEATURES_DIR, LIVE_LABELS_DIR, LIVE_PENDING_DIR,
          REFERENCE_DIR, RETRAINING_DIR, DRIFT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CONFIGURATION — DECISIONS AND ALTERNATIVES
# =============================================================================

# DECISION: 8 weekly batches, shuffled with random_state=42.
# Alternatives: 4 batches (monthly), 12 batches (fortnightly).
# Why not picked: 8 gives enough granularity to observe drift onset at week 5
# while keeping each batch large enough (~880 customers) for stable metrics.
N_WEEKS = 8

# DECISION: Drift starts at week 5 (1-indexed).
# Alternatives: week 3 (earlier), week 7 (later).
# Why not picked: week 5 gives 4 weeks of stable pre-drift data to establish
# a baseline before testing the retraining trigger.
DRIFT_START_WEEK = 5

# DECISION: Label delay = 2 weeks.
# Alternatives: 1 week (too short to feel real), 4 weeks (too little labeled data).
# Why not picked: 2 weeks simulates a realistic monitoring window (customers
# need time to actually churn or be confirmed retained) while keeping enough
# labeled data for retraining decisions.
LABEL_DELAY_WEEKS = 2

# DECISION: Covariate drift — Monthly Charges * 1.25, Tenure Months * 0.7.
# Alternatives: only price increase, only tenure change, gradual drift.
# Why not picked: both together simulate a realistic scenario (price hike +
# newer customer mix). Step drift at week 5 makes it easier to verify the
# trigger fires than gradual drift.
PRICE_INCREASE_MULT = 1.25
TENURE_FACTOR = 0.7

# DECISION: Concept drift — relabel 30% of high-charge retained customers as
# churned in drifted weeks.
# Alternatives: flip all high-charge labels, use a separate concept-drift model,
# keep labels unchanged (pure covariate drift only).
# Why not picked: partial (30%) concept drift models a realistic scenario where
# price-sensitive customers churn more but not everyone reacts the same way.
# Keeping labels unchanged would make this pure covariate drift, not both.
CONCEPT_FLIP_PROB = 0.30

# DECISION: Hybrid flow — 20% returning customers per week (from previous week).
# Alternatives: 0% (disjoint cohorts), 50% (mostly returning), 100% (same customers).
# Why not picked: 20% creates overlap between weeks (hybrid) while still
# introducing enough new customers to test the model on unseen profiles each week.
RETURNING_FRAC = 0.20

# DECISION: Retraining trigger = performance decay (AUC drop > 5% from best).
# Alternatives: fixed schedule (every week), drift detection (PSI/KS), no retraining.
# Why not picked: fixed schedule retrains even when unnecessary. Drift detection
# flags covariate shift but not necessarily model degradation. Performance decay
# directly measures whether the model is becoming less useful, which is what
# retraining should fix.
AUC_DROP_THRESHOLD = 0.05

# DECISION: Minimum 200 labeled samples before first retrain.
# Alternatives: retrain immediately, wait for all labels, wait for 1000 samples.
# Why not picked: 200 is enough to fit a stable logistic regression without
# waiting too long into the simulation.
MIN_LABELS_FOR_RETRAIN = 200

# DECISION: Reference set = 15% of population, held out before weekly split.
# Alternatives: use latest week as reference, use all data as reference.
# Why not picked: latest week has drift (not a fair yardstick). All data leaks
# into training. 15% pre-drift holdout is a static, clean yardstick used for
# final comparison only — NOT for the retraining trigger.
REFERENCE_FRAC = 0.15

# DECISION: Performance-decay trigger evaluates on recent live labeled data
# (rolling window of last 2 released weeks), not on a static reference.
# Alternatives: evaluate on static reference, evaluate on all labeled data.
# Why not picked: a static reference never changes, so a pre-drift model's AUC
# on it stays flat forever and the trigger can never fire. All labeled data
# mixes pre-drift and post-drift samples, diluting the decay signal. A rolling
# window of recent live labels directly measures whether the model is losing
# predictive power on current data.
RECENT_WINDOW_WEEKS = 2

# DECISION: Logistic regression with StandardScaler, implemented in numpy.
# Alternatives: scikit-learn, RandomForest, XGBoost.
# Why not picked: scikit-learn is not a project dependency; adding it changes
# pyproject.toml. RandomForest/XGBoost are more robust to drift and would mask
# the performance-decay effect we are measuring. Logistic regression is fast,
# interpretable, and sensitive to distribution shifts — ideal for this demo.
RANDOM_STATE = 42
RNG = np.random.RandomState(RANDOM_STATE)


# =============================================================================
# LIGHTWEIGHT MODEL (numpy only, no scikit-learn dependency)
# =============================================================================

class SimpleLogisticRegression:
    """L2-regularised logistic regression trained with gradient descent."""

    def __init__(self, lr=0.05, n_iter=2000, random_state=RANDOM_STATE):
        self.lr = lr
        self.n_iter = n_iter
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.mean_ = None
        self.scale_ = None

    def _standardize_fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0

    def _standardize_transform(self, X):
        return (X - self.mean_) / self.scale_

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self._standardize_fit(X)
        Xs = self._standardize_transform(X)

        n_samples, n_features = Xs.shape
        rng = np.random.RandomState(self.random_state)
        self.weights = rng.randn(n_features) * 0.01
        self.bias = 0.0

        for _ in range(self.n_iter):
            linear = Xs.dot(self.weights) + self.bias
            y_pred = self._sigmoid(linear)
            error = y_pred - y
            dw = (1.0 / n_samples) * Xs.T.dot(error)
            db = (1.0 / n_samples) * np.sum(error)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        Xs = self._standardize_transform(X)
        linear = Xs.dot(self.weights) + self.bias
        p = self._sigmoid(linear)
        return np.column_stack([1.0 - p, p])


def roc_auc_score(y_true, y_score):
    """Mann-Whitney U statistic with tie handling (average ranks)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    order = np.argsort(y_score, kind="mergesort")
    y_sorted = y_true[order]
    scores_sorted = y_score[order]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = np.empty(len(y_true), dtype=float)
    i = 0
    n = len(scores_sorted)
    while i < n:
        j = i
        while j + 1 < n and scores_sorted[j + 1] == scores_sorted[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    sum_ranks_pos = np.sum(ranks[y_true == 1])
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# =============================================================================
# LOAD AND PREPARE POPULATION
# =============================================================================

X = pd.read_csv(PROCESSED_DIR / "X_features.csv")
y = pd.read_csv(PROCESSED_DIR / "y_target.csv")

data = pd.concat([X, y], axis=1).reset_index(drop=True)

# DECISION: Synthetic customer_id = original row index.
# Alternatives: read CustomerID from raw Excel and merge, use UUIDs.
# Why not picked: CustomerID was intentionally dropped during cleaning to avoid
# leakage; re-merging adds complexity. A synthetic integer ID is sufficient for
# tracking hybrid returning customers and is not used as a model feature.
data["customer_id"] = data.index.astype(str)

# Shuffle the entire population
data = data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# =============================================================================
# REFERENCE SET (static pre-drift holdout)
# =============================================================================

n_reference = int(len(data) * REFERENCE_FRAC)
reference = data.iloc[:n_reference].copy()
reference.to_csv(REFERENCE_DIR / "X_reference.csv", index=False)
reference_labels = reference[["customer_id", "Churn Value"]].copy()
reference_labels.to_csv(REFERENCE_DIR / "y_reference.csv", index=False)

# Remaining population goes into weekly batches
population = data.iloc[n_reference:].copy()

print(f"Population size: {len(population)}")
print(f"Reference set size: {len(reference)}")

# =============================================================================
# CREATE WEEKLY BATCHES (HYBRID: NEW + RETURNING CUSTOMERS)
# =============================================================================

# DECISION: Split population into 8 disjoint new-customer batches by index.
# Alternatives: random sample with replacement each week, stratified split.
# Why not picked: array_split is simple and ensures every customer appears as a
# new customer exactly once. Stratification is unnecessary for this simulation.
split_indices = np.array_split(np.arange(len(population)), N_WEEKS)
new_batches = [population.iloc[idx].reset_index(drop=True) for idx in split_indices]

weekly_batches = []
seen_customers = set()

for i in range(N_WEEKS):
    new_batch = new_batches[i].copy()
    new_ids = set(new_batch["customer_id"])
    seen_customers.update(new_ids)

    if i == 0:
        # Week 1: 100% new customers
        weekly_batch = new_batch.copy()
    else:
        # DECISION: Hybrid — mix new customers with returning customers from previous week.
        # Alternatives: draw returning from all previously seen customers, not just previous week.
        # Why not picked: drawing from only the previous week keeps the "returning"
        # relationship local and interpretable (customers return the following week).
        prev_batch = weekly_batches[i - 1]
        n_returning = int(len(new_batch) * RETURNING_FRAC)

        if n_returning > 0 and len(prev_batch) > 0:
            returning = prev_batch.sample(
                n=n_returning, random_state=RANDOM_STATE + i
            ).copy()

            # DECISION: Returning customers age by 1 month and get 2% charge inflation.
            # Alternatives: no feature update for returning customers, random feature changes.
            # Why not picked: aging tenure and slight charge inflation reflects
            # real passage of time between weeks.
            returning["Tenure Months"] = (returning["Tenure Months"] + 1).clip(upper=72)
            returning["Monthly Charges"] = returning["Monthly Charges"] * 1.02

            weekly_batch = pd.concat([new_batch, returning], ignore_index=True)
        else:
            weekly_batch = new_batch.copy()

    # Track when features and labels become available
    weekly_batch["feature_available_week"] = i + 1
    weekly_batch["label_release_week"] = i + 1 + LABEL_DELAY_WEEKS

    weekly_batches.append(weekly_batch)

# Combine all weeks into a master dataframe
master = pd.concat(weekly_batches, ignore_index=True)
print(f"Total weekly records (with returning duplicates): {len(master)}")
print(f"Unique customers: {master['customer_id'].nunique()}")

# =============================================================================
# INJECT DRIFT (WEEKS 5+)
# =============================================================================

for i in range(DRIFT_START_WEEK - 1, N_WEEKS):
    batch = weekly_batches[i]

    # --- Covariate drift ---
    # DECISION: Apply price increase and tenure reduction to all features in drifted weeks.
    # Alternatives: apply only to new customers, apply gradually over weeks.
    # Why not picked: applying to all customers simulates a market-wide price hike
    # affecting everyone. Gradual drift is more realistic but harder to verify;
    # step drift makes trigger testing clearer.
    batch["Monthly Charges"] = batch["Monthly Charges"] * PRICE_INCREASE_MULT
    batch["Tenure Months"] = (batch["Tenure Months"] * TENURE_FACTOR).round().clip(lower=0)

    # --- Concept drift ---
    # DECISION: Relabel retained customers with above-median monthly charges as
    # churned with 30% probability in drifted weeks.
    # Alternatives: flip all high-charge labels, use a separate concept-drift model,
    # keep labels unchanged (pure covariate drift only).
    # Why not picked: partial (30%) concept drift models a realistic scenario where
    # price-sensitive customers churn more but not everyone reacts the same way.
    median_charge = batch["Monthly Charges"].median()
    high_charge = batch["Monthly Charges"] > median_charge
    was_retained = batch["Churn Value"] == 0
    flip_mask = high_charge & was_retained & (RNG.rand(len(batch)) < CONCEPT_FLIP_PROB)
    batch.loc[flip_mask, "Churn Value"] = 1

    print(f"Week {i+1}: {int(flip_mask.sum())} labels flipped due to concept drift")

# Rebuild master after drift
master = pd.concat(weekly_batches, ignore_index=True)

# =============================================================================
# SIMULATION LOOP
# =============================================================================

# Feature columns (everything except metadata and target)
META_COLS = ["customer_id", "Churn Value", "feature_available_week", "label_release_week"]
FEATURE_COLS = [c for c in master.columns if c not in META_COLS]

# Reference features and labels for evaluation (final comparison only)
X_ref = reference[FEATURE_COLS].values
y_ref = reference["Churn Value"].values

# Storage
drift_metrics = []
retrained_weeks = []

# --- Never-retrain baseline ---
# DECISION: Never-retrain trains on all labeled data available up to week 4
# (pre-drift). Alternatives: train only on week 1, train on weeks 1-3.
# Why not picked: weeks 1-4 gives the most pre-drift labeled data for a fair
# baseline comparison without including any drifted data.
never_retrain_data = master[
    master["label_release_week"] <= DRIFT_START_WEEK
].copy()

if len(never_retrain_data) > 0:
    X_nr = never_retrain_data[FEATURE_COLS].values
    y_nr = never_retrain_data["Churn Value"].values
    model_never = SimpleLogisticRegression()
    model_never.fit(X_nr, y_nr)
    print(f"\nNever-retrain baseline trained on {len(never_retrain_data)} samples (weeks 1-{DRIFT_START_WEEK})")
else:
    model_never = None
    print("WARNING: No pre-drift labeled data for never-retrain baseline")

# --- Retraining model state ---
model_retrain = None
best_recent_auc = None
retrain_history = []

# Run simulation week by week
for week in range(1, N_WEEKS + 1):
    print(f"\n{'='*60}")
    print(f"SIMULATION WEEK {week}")
    print(f"{'='*60}")

    # --- Release features for current week ---
    week_features = master[
        master["feature_available_week"] == week
    ][FEATURE_COLS + ["customer_id"]]
    week_features.to_csv(
        LIVE_FEATURES_DIR / f"week_{week}_features.csv", index=False
    )

    # --- Copy features to pending (awaiting labels) ---
    week_features.copy().to_csv(
        LIVE_PENDING_DIR / f"week_{week}_pending.csv", index=False
    )

    # --- Release labels for week (week - LABEL_DELAY) ---
    release_week = week - LABEL_DELAY_WEEKS
    if release_week >= 1:
        week_labels = master[
            master["feature_available_week"] == release_week
        ][["customer_id", "Churn Value"]].copy()
        week_labels.to_csv(
            LIVE_LABELS_DIR / f"week_{release_week}_labels.csv", index=False
        )
        # Remove from pending — labels now available
        pending_file = LIVE_PENDING_DIR / f"week_{release_week}_pending.csv"
        if pending_file.exists():
            pending_file.unlink()
        print(f"Labels released for week {release_week}")

    # --- Collect all labeled data available so far ---
    labeled_data = master[
        master["label_release_week"] <= week
    ].copy()

    n_labeled = len(labeled_data)
    print(f"Labeled samples available: {n_labeled}")

    if n_labeled < MIN_LABELS_FOR_RETRAIN:
        print("Not enough labeled data to train/evaluate yet.")
        continue

    X_avail = labeled_data[FEATURE_COLS].values
    y_avail = labeled_data["Churn Value"].values

    # --- Build recent live evaluation window (last RECENT_WINDOW_WEEKS released weeks) ---
    release_cutoff = week - RECENT_WINDOW_WEEKS + 1
    recent_live = master[
        (master["label_release_week"] >= release_cutoff) &
        (master["label_release_week"] <= week)
    ].copy()

    if len(recent_live) > 0:
        X_recent = recent_live[FEATURE_COLS].values
        y_recent = recent_live["Churn Value"].values
    else:
        X_recent = X_avail
        y_recent = y_avail

    # --- Never-retrain baseline evaluation on reference ---
    auc_never = None
    if model_never is not None:
        y_never_pred = model_never.predict_proba(X_ref)[:, 1]
        auc_never = roc_auc_score(y_ref, y_never_pred)
        print(f"  Never-retrain AUC on static reference: {auc_never:.4f}")

    # --- Retraining model with performance-decay trigger ---
    should_retrain = False
    trigger_reason = ""

    if model_retrain is None:
        should_retrain = True
        trigger_reason = "initial training"
    else:
        # DECISION: Evaluate current model on recent live labeled data for the trigger.
        # Alternatives: evaluate on static reference (never changes → trigger never fires),
        # evaluate on all labeled data (dilutes decay signal).
        # Why not picked: recent live data directly measures whether the model is
        # losing predictive power on current (drifted) data.
        y_ret_pred = model_retrain.predict_proba(X_recent)[:, 1]
        auc_ret_recent = roc_auc_score(y_recent, y_ret_pred)
        print(f"  Retrain-model AUC on recent live data: {auc_ret_recent:.4f}")

        if best_recent_auc is not None and auc_ret_recent < best_recent_auc - AUC_DROP_THRESHOLD:
            should_retrain = True
            trigger_reason = (
                f"performance decay: recent AUC {auc_ret_recent:.4f} < best {best_recent_auc:.4f} - {AUC_DROP_THRESHOLD}"
            )
        else:
            print(f"  No retraining triggered (best recent AUC: {best_recent_auc:.4f})")

    if should_retrain and n_labeled >= MIN_LABELS_FOR_RETRAIN:
        print(f"  RETRAINING triggered: {trigger_reason}")
        model_retrain = SimpleLogisticRegression()
        model_retrain.fit(X_avail, y_avail)

        # Evaluate new model on both recent live and static reference
        y_ret_pred_recent = model_retrain.predict_proba(X_recent)[:, 1]
        auc_ret_recent = roc_auc_score(y_recent, y_ret_pred_recent)
        best_recent_auc = auc_ret_recent

        y_ret_pred_ref = model_retrain.predict_proba(X_ref)[:, 1]
        auc_ret_ref = roc_auc_score(y_ref, y_ret_pred_ref)

        retrained_weeks.append(week)
        print(f"  New model AUC on recent live data: {auc_ret_recent:.4f}")
        print(f"  New model AUC on static reference: {auc_ret_ref:.4f}")

        # Save retraining artifacts
        retrain_dir = RETRAINING_DIR / f"week_{week}"
        retrain_dir.mkdir(parents=True, exist_ok=True)

        metrics = {
            "week": week,
            "trigger": trigger_reason,
            "n_labeled": n_labeled,
            "auc_on_recent_live": auc_ret_recent,
            "auc_on_static_reference": auc_ret_ref,
            "never_retrain_auc_reference": auc_never,
        }
        with open(retrain_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
    elif model_retrain is not None:
        # Evaluate without retraining
        y_ret_pred = model_retrain.predict_proba(X_recent)[:, 1]
        auc_ret_recent = roc_auc_score(y_recent, y_ret_pred)
        best_recent_auc = max(best_recent_auc, auc_ret_recent) if best_recent_auc is not None else auc_ret_recent
        print(f"  Retrain-model AUC on recent live data: {auc_ret_recent:.4f}")

    # Record metrics for this week
    drift_metrics.append({
        "week": week,
        "n_labeled": n_labeled,
        "never_retrain_auc_reference": auc_never,
        "retrain_model_auc_recent": best_recent_auc if model_retrain is not None else None,
        "retrained": week in retrained_weeks,
    })

# Save drift metrics
drift_metrics_df = pd.DataFrame(drift_metrics)
drift_metrics_df.to_csv(DRIFT_DIR / "drift_metrics.csv", index=False)

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print(f"\n{'='*60}")
print("SIMULATION COMPLETE")
print(f"{'='*60}")
print(f"Retraining triggered at weeks: {retrained_weeks}")
print(f"Metrics saved to {DRIFT_DIR / 'drift_metrics.csv'}")
print(f"Retraining artifacts in {RETRAINING_DIR}")

if not drift_metrics_df.empty:
    print("\nDrift Metrics Summary:")
    print(drift_metrics_df.to_string(index=False))
