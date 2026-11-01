import argparse
import logging
from pathlib import Path
import sys

class _DynamicStdout:
    def write(self, s: str):
        return sys.stdout.write(s)

    def flush(self):
        return sys.stdout.flush()


class _DynamicStderr:
    def write(self, s: str):
        return sys.stderr.write(s)

    def flush(self):
        return sys.stderr.flush()


logger = logging.getLogger("churn.cli")
if not logger.handlers:
    _out_handler = logging.StreamHandler(_DynamicStdout())
    _out_handler.setFormatter(logging.Formatter("%(message)s"))
    _out_handler.addFilter(lambda record: record.levelno < logging.ERROR)

    _err_handler = logging.StreamHandler(_DynamicStderr())
    _err_handler.setLevel(logging.ERROR)
    _err_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(_out_handler)
    logger.addHandler(_err_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

from final_project.data import load_and_split_data
from final_project.models import benchmark_candidates
from final_project.pipeline import run_training
from final_project.evaluation import run_evaluation
from final_project.explainability import run_explanation
from final_project.simulation import SimulationConfig, run_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="churn",
        description="Production Churn Modeling & Retraining Pipeline CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Pipeline commands")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train baseline/candidate churn model")
    train_parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing X_features.csv and y_target.csv",
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained model artifact and metadata",
    )
    train_parser.add_argument(
        "--algorithm",
        type=str,
        default="random_forest",
        choices=[
            "random_forest",
            "lightgbm",
            "xgboost",
            "rf",
            "lgb",
            "xgb",
            "logistic_regression",
            "logistic",
            "lr",
            "bagged_logistic",
            "bagged_lr",
            "mlp",
            "voting_ensemble",
            "stacking_ensemble",
        ],
        help="Model algorithm to train (default: random_forest)",
    )
    train_parser.add_argument(
        "--balanced",
        action="store_true",
        help="Apply native algorithmic class weighting (class_weight='balanced' or scale_pos_weight)",
    )
    train_parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Proportion of data for holdout test split (default: 0.20)",
    )
    train_parser.add_argument(
        "--val-size",
        type=float,
        default=0.10,
        help="Proportion of data for threshold calibration validation split (default: 0.10)",
    )
    train_parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds (default: 5)",
    )
    train_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    train_parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna Bayesian hyperparameter optimization study across candidates",
    )
    train_parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Total Optuna tuning budget across candidates (default: 50)",
    )
    train_parser.add_argument(
        "--experiment-name",
        type=str,
        default="churn_retraining_optimization",
        help="MLflow experiment name for trial tracking (default: churn_retraining_optimization)",
    )
    train_parser.add_argument(
        "--tracking-uri",
        type=str,
        default="sqlite:///mlflow.db",
        help="MLflow tracking URI (default: sqlite:///mlflow.db)",
    )

    # Benchmark subcommand
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark candidate tree ensembles with unweighted vs. balanced modes",
    )
    bench_parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing X_features.csv and y_target.csv",
    )
    bench_parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds (default: 5)",
    )
    bench_parser.add_argument(
        "--all",
        action="store_true",
        help="Benchmark all candidate model families including non-tree (Logistic, MLP) and ensemble combinations",
    )
    bench_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Evaluate subcommand
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model on holdout test set and optimize decision threshold",
    )
    eval_parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing X_features.csv and y_target.csv",
    )
    eval_parser.add_argument(
        "--model-path",
        type=str,
        default="models/model.joblib",
        help="Path to serialized model artifact",
    )
    eval_parser.add_argument(
        "--metadata-path",
        type=str,
        default="models/metadata.json",
        help="Path to model metadata JSON file to update",
    )
    eval_parser.add_argument(
        "--report-dir",
        type=str,
        default="reports",
        help="Directory to save markdown report and diagnostic figures",
    )
    eval_parser.add_argument(
        "--cost-fn",
        type=float,
        default=200.0,
        help="Financial loss of a false negative / missed churner (default: 200.0)",
    )
    eval_parser.add_argument(
        "--cost-fp",
        type=float,
        default=20.0,
        help="Cost of a false positive / retention incentive offer (default: 20.0)",
    )
    eval_parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Holdout test split proportion (default: 0.20)",
    )
    eval_parser.add_argument(
        "--val-size",
        type=float,
        default=0.10,
        help="Validation split proportion for threshold calibration (default: 0.10)",
    )
    eval_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Explain subcommand
    explain_parser = subparsers.add_parser(
        "explain",
        help="Explain model predictions and feature importance with SHAP TreeExplainer",
    )
    explain_parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing X_features.csv and y_target.csv",
    )
    explain_parser.add_argument(
        "--model-path",
        type=str,
        default="models/model.joblib",
        help="Path to serialized model artifact",
    )
    explain_parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/figures",
        help="Directory to save generated SHAP diagnostic plots",
    )
    explain_parser.add_argument(
        "--max-samples",
        type=int,
        default=150,
        help="Maximum sample size to explain with SHAP (default: 150)",
    )
    explain_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Dashboard subcommand
    dash_parser = subparsers.add_parser(
        "dashboard",
        help="Launch the interactive ChurnOps Streamlit Operations Dashboard",
    )
    dash_parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to serve Streamlit web dashboard (default: 8501)",
    )
    dash_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind Streamlit web dashboard (default: 0.0.0.0)",
    )
    dash_parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Streamlit in headless mode (use --no-headless to open browser window)",
    )

    # Serve subcommand
    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch the high-performance FastAPI inference microservice",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to serve FastAPI inference API (default: 8000)",
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind FastAPI API (default: 0.0.0.0)",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload on code modifications",
    )

    # Simulate subcommand
    sim_parser = subparsers.add_parser(
        "simulate",
        help="Run multi-week streaming data, drift injection, and automated retraining simulation",
    )
    sim_parser.add_argument(
        "--n-weeks",
        type=int,
        default=8,
        help="Number of simulated streaming weeks (default: 8)",
    )
    sim_parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing X_features.csv and y_target.csv",
    )
    sim_parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing production model artifact",
    )
    sim_parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Do not save simulation artifacts to data/live, data/drift, and models/retrained",
    )
    sim_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for simulation reproducibility (default: 42)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand is None:
        parser.print_help()
        return 0

    if args.subcommand == "train":
        logger.info(f"Starting training pipeline...")
        logger.info(f"  Data directory: {args.data_dir}")
        logger.info(f"  Output directory: {args.output_dir}")
        if args.tune:
            logger.info(f"  Mode: Bayesian Hyperparameter Optimization (Optuna)")
            logger.info(f"  Tuning Budget: {args.n_trials} trials")
            logger.info(f"  MLflow Experiment: {args.experiment_name}")
        else:
            logger.info(f"  Algorithm: {args.algorithm} (balanced={args.balanced})")
        logger.info(f"  Stratified CV Folds: {args.cv_folds}")

        summary = run_training(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            algorithm=args.algorithm,
            balanced=args.balanced,
            tune=args.tune,
            n_trials=args.n_trials,
            experiment_name=args.experiment_name,
            tracking_uri=args.tracking_uri,
            test_size=args.test_size,
            val_size=getattr(args, "val_size", 0.10),
            cv_folds=args.cv_folds,
            random_state=args.random_state,
        )

        logger.info(f"\nTraining completed successfully!")
        logger.info(f"  Winning Algorithm: {summary.algorithm}")
        logger.info(f"  Balanced: {summary.balanced}")
        logger.info(f"  Train samples: {summary.train_samples}")
        logger.info(f"  Test samples: {summary.test_samples}")
        logger.info(f"  Features: {summary.feature_count}")
        logger.info(f"  CV Best/Mean ROC-AUC: {summary.cv_auc_mean:.4f} (+/- {summary.cv_auc_std:.4f})")
        logger.info(f"  Fold Scores: {[round(s, 4) for s in summary.cv_fold_scores]}")
        logger.info(f"  Model saved to: {summary.model_path}")
        logger.info(f"  Metadata saved to: {summary.metadata_path}")
        return 0

    if args.subcommand == "benchmark":
        logger.info("Starting multi-model candidate benchmark...")
        logger.info(f"  Data directory: {args.data_dir}")
        logger.info(f"  Stratified CV Folds: {args.cv_folds}")

        features_path = Path(args.data_dir) / "X_features.csv"
        target_path = Path(args.data_dir) / "y_target.csv"
        split = load_and_split_data(
            features_path=features_path,
            target_path=target_path,
            random_state=args.random_state,
        )

        algos = (
            [
                "random_forest",
                "lightgbm",
                "xgboost",
                "logistic_regression",
                "bagged_logistic",
                "mlp",
                "voting_ensemble",
                "stacking_ensemble",
            ]
            if getattr(args, "all", False)
            else None
        )

        df = benchmark_candidates(
            X=split.X_train,
            y=split.y_train,
            algorithms=algos,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
        )

        logger.info(f"\nCandidate Benchmark Results ({args.cv_folds}-Fold Stratified CV):")
        logger.info("=" * 72)
        logger.info(f"{'Rank':<5} {'Model':<15} {'Weighting':<12} {'ROC-AUC (mean±std)':<20} {'PR-AUC (mean±std)':<20}")
        logger.info("-" * 72)
        for i, row in df.iterrows():
            roc_str = f"{row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f}"
            pr_str = f"{row['pr_auc_mean']:.4f} ± {row['pr_auc_std']:.4f}"
            logger.info(f"{i+1:<5} {row['model']:<15} {row['weighting']:<12} {roc_str:<20} {pr_str:<20}")
        logger.info("=" * 72)
        return 0

    if args.subcommand == "evaluate":
        logger.info("Starting holdout evaluation and threshold optimization...")
        logger.info(f"  Model artifact: {args.model_path}")
        logger.info(f"  Data directory: {args.data_dir}")
        logger.info(f"  Cost Matrix: C_FN=${args.cost_fn:.0f}, C_FP=${args.cost_fp:.0f}")

        summary = run_evaluation(
            data_dir=args.data_dir,
            model_path=args.model_path,
            metadata_path=args.metadata_path,
            report_dir=args.report_dir,
            cost_fn=args.cost_fn,
            cost_fp=args.cost_fp,
            test_size=args.test_size,
            val_size=getattr(args, "val_size", 0.10),
            random_state=args.random_state,
        )

        def_m = summary.default_metrics
        opt_m = summary.cost_optimal_metrics

        logger.info(f"\nHoldout Evaluation Completed Successfully!")
        logger.info(f"  ROC-AUC: {summary.roc_auc:.4f}")
        logger.info(f"  PR-AUC:  {summary.pr_auc:.4f}")
        logger.info(f"  Brier Score: {summary.brier_score:.4f}")
        logger.info(f"\nThreshold Optimization Summary:")
        logger.info(f"  Default Cutoff (0.50):       Recall: {def_m.recall:.2%}, Loss: ${def_m.cost:,.0f}")
        logger.info(f"  Optimal Cutoff ({summary.optimal_threshold:.2f}):      Recall: {opt_m.recall:.2%}, Loss: ${opt_m.cost:,.0f}")
        logger.info(f"  Net Financial Savings:       ${summary.cost_savings:,.2f}")
        logger.info(f"\nArtifacts Created:")
        logger.info(f"  Markdown Report: {summary.report_path}")
        logger.info(f"  Diagnostic Plots ({len(summary.figures)}): {Path(args.report_dir) / 'figures'}")
        return 0

    if args.subcommand == "explain":
        logger.info("Starting model explainability analysis (SHAP TreeExplainer)...")
        logger.info(f"  Model artifact: {args.model_path}")
        logger.info(f"  Data directory: {args.data_dir}")
        logger.info(f"  Output directory: {args.output_dir}")
        logger.info(f"  Sample size: {args.max_samples}")

        result = run_explanation(
            data_dir=args.data_dir,
            model_path=args.model_path,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            random_state=args.random_state,
        )

        logger.info(f"\nModel Explainability Completed Successfully!")
        logger.info(f"  Samples explained: {result.samples_explained}")
        logger.info(f"\nTop Parent Feature Importances (Aggregated Mean |SHAP|):")
        logger.info("=" * 60)
        logger.info(f"{'Rank':<5} {'Parent Feature':<30} {'Mean |SHAP|':<15}")
        logger.info("-" * 60)
        for i, row in result.grouped_importance.head(10).iterrows():
            logger.info(f"{i+1:<5} {row['parent_feature']:<30} {row['mean_abs_shap']:.4f}")
        logger.info("=" * 60)
        logger.info(f"\nGenerated Diagnostic Figures ({len(result.generated_figures)}):")
        for fig in result.generated_figures:
            logger.info(f"  - {fig}")
        return 0

    if args.subcommand == "dashboard":
        import subprocess

        app_path = Path(__file__).parent / "dashboard" / "app.py"
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port",
            str(args.port),
            "--server.address",
            str(args.host),
            "--server.headless",
            "true" if args.headless else "false",
        ]
        logger.info(f"Launching ChurnOps Streamlit Dashboard on http://{args.host}:{args.port}...")
        try:
            return subprocess.call(cmd)
        except KeyboardInterrupt:
            logger.info("\nDashboard server terminated by user.")
            return 0

    if args.subcommand == "serve":
        import uvicorn
        from final_project.serving.api import app as fastapi_app

        logger.info(f"Starting ChurnOps REST API inference server on http://{args.host}:{args.port}...")
        try:
            uvicorn.run(
                "final_project.serving.api:app" if args.reload else fastapi_app,
                host=args.host,
                port=args.port,
                reload=args.reload,
            )
            return 0
        except KeyboardInterrupt:
            logger.info("\nServing server terminated by user.")
            return 0

    if args.subcommand == "simulate":
        logger.info(f"Starting weekly streaming simulation ({args.n_weeks} weeks)...")
        logger.info(f"  Data directory: {args.data_dir}")
        logger.info(f"  Models directory: {args.models_dir}")
        logger.info(f"  Artifact saving: {'Disabled' if args.no_save else 'Enabled'}")

        config = SimulationConfig(
            n_weeks=args.n_weeks,
            random_state=args.random_state,
        )
        results = run_simulation(
            config=config,
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            save_artifacts=not args.no_save,
            verbose=True,
        )

        logger.info(f"\nSimulation Completed Successfully!")
        logger.info(f"  Weeks executed: {len(results.drift_metrics)}")
        logger.info(f"  Retraining triggered at weeks: {results.retrained_weeks}")
        if results.final_retrain_auc_recent is not None:
            logger.info(f"  Final Retrained AUC (recent live): {results.final_retrain_auc_recent:.4f}")
        return 0

    logger.error(f"Unknown subcommand: {args.subcommand}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
