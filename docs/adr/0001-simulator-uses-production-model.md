# Replace simulator's hand-rolled logistic regression with production model

The retraining simulator (`notebooks/simulate_retraining.py`) originally implemented logistic regression and rank-based AUC from scratch to avoid any dependency on scikit-learn. With Milestone 2 producing a production-trained model (XGBoost/LightGBM via scikit-learn API), the simulator will load the saved joblib artifact instead of using `SimpleLogisticRegression`. This adds scikit-learn, xgboost, and lightgbm as runtime dependencies for the simulator, breaking the original "no scikit-learn" constraint. We chose this over keeping them independent because the simulator should test how the *actual production model* degrades under drift, not a toy proxy.

## Considered Options

- **Keep independent**: Simulator keeps its hand-rolled model. Simpler dependency story, but the drift analysis doesn't reflect real production behavior.
- **Import training code**: Simulator imports from `src/final_project.training` and retrains from scratch each run. Too coupled; the simulator's job is to test a *fixed* model's degradation, not to retrain.
- **Load joblib artifact** (chosen): Simulator loads `models/best_model.joblib`. Clean boundary — the simulator consumes a model the training pipeline produces. The drift analysis reflects real production model behavior.
