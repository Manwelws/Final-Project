#!/usr/bin/env python3
"""
Churn Retraining Simulator (Driver Script)
=========================================

Simulates a live weekly streaming environment where:
  * Features arrive every week (available immediately for scoring)
  * Unsupervised Data Drift Monitor computes Population Stability Index (PSI)
  * Labels (actual churn outcomes) are released after a fixed 2-week delay
  * Covariate + concept drift start at week 5
  * A Supervised Performance-Decay Trigger decides when to retrain
  * Retraining incorporates Exponential Sample Decay (0.85 ** weeks_old)
  * A never-retrain baseline is run for comparison (ADR 0001)

Outputs:
  data/live/features/week_N_features.csv   - features available at week N
  data/live/labels/week_N_labels.csv       - labels released at week N+delay
  data/live/pending/                       - features awaiting labels
  data/reference/                          - static pre-drift holdout
  models/retrained/week_N/                 - metrics & model when retrained
  data/drift/drift_metrics.csv             - PSI & AUC per week for both strategies

Every design decision is documented inline in final_project.simulation:
  DECISION / ALTERNATIVES CONSIDERED / WHY NOT PICKED
"""

from pathlib import Path
from final_project.simulation import SimulationConfig, SimulationRunner


def main() -> None:
    # DECISION: 8 weekly batches, shuffled with random_state=42.
    # Drift starts at week 5; label delay = 2 weeks; price * 1.25, tenure * 0.7;
    # Concept flip prob = 0.30; returning frac = 0.20; AUC drop threshold = 0.05;
    # Sample decay rate = 0.85; reference frac = 0.15; recent window = 2 weeks.
    config = SimulationConfig(
        n_weeks=8,
        drift_start_week=5,
        label_delay_weeks=2,
        price_increase_mult=1.25,
        tenure_factor=0.70,
        concept_flip_prob=0.30,
        returning_frac=0.20,
        auc_drop_threshold=0.05,
        min_labels_for_retrain=200,
        reference_frac=0.15,
        recent_window_weeks=2,
        sample_decay_rate=0.85,
        random_state=42,
    )

    runner = SimulationRunner(
        config=config,
        data_dir=Path("data/processed"),
        models_dir=Path("models"),
        live_dir=Path("data/live"),
        reference_dir=Path("data/reference"),
        retraining_dir=Path("models/retrained"),
        drift_dir=Path("data/drift"),
        save_artifacts=True,
    )

    runner.run()


if __name__ == "__main__":
    main()
