import argparse
import sys
from pathlib import Path

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
        action="store_true",
        default=True,
        help="Run Streamlit in headless mode without spawning local browser window",
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
        help="Do not save simulation artifacts to data/live, data/drift, and data/retraining",
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
        print(f"Starting training pipeline...")
        print(f"  Data directory: {args.data_dir}")
        print(f"  Output directory: {args.output_dir}")
        if args.tune:
            print(f"  Mode: Bayesian Hyperparameter Optimization (Optuna)")
            print(f"  Tuning Budget: {args.n_trials} trials")
            print(f"  MLflow Experiment: {args.experiment_name}")
        else:
            print(f"  Algorithm: {args.algorithm} (balanced={args.balanced})")
        print(f"  Stratified CV Folds: {args.cv_folds}")

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
            cv_folds=args.cv_folds,
            random_state=args.random_state,
        )

        print(f"\nTraining completed successfully!")
        print(f"  Winning Algorithm: {summary.algorithm}")
        print(f"  Balanced: {summary.balanced}")
        print(f"  Train samples: {summary.train_samples}")
        print(f"  Test samples: {summary.test_samples}")
        print(f"  Features: {summary.feature_count}")
        print(f"  CV Best/Mean ROC-AUC: {summary.cv_auc_mean:.4f} (+/- {summary.cv_auc_std:.4f})")
        print(f"  Fold Scores: {[round(s, 4) for s in summary.cv_fold_scores]}")
        print(f"  Model saved to: {summary.model_path}")
        print(f"  Metadata saved to: {summary.metadata_path}")
        return 0

    if args.subcommand == "benchmark":
        print("Starting multi-model candidate benchmark...")
        print(f"  Data directory: {args.data_dir}")
        print(f"  Stratified CV Folds: {args.cv_folds}")

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

        print(f"\nCandidate Benchmark Results ({args.cv_folds}-Fold Stratified CV):")
        print("=" * 72)
        print(f"{'Rank':<5} {'Model':<15} {'Weighting':<12} {'ROC-AUC (mean±std)':<20} {'PR-AUC (mean±std)':<20}")
        print("-" * 72)
        for i, row in df.iterrows():
            roc_str = f"{row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f}"
            pr_str = f"{row['pr_auc_mean']:.4f} ± {row['pr_auc_std']:.4f}"
            print(f"{i+1:<5} {row['model']:<15} {row['weighting']:<12} {roc_str:<20} {pr_str:<20}")
        print("=" * 72)
        return 0

    if args.subcommand == "evaluate":
        print("Starting holdout evaluation and threshold optimization...")
        print(f"  Model artifact: {args.model_path}")
        print(f"  Data directory: {args.data_dir}")
        print(f"  Cost Matrix: C_FN=${args.cost_fn:.0f}, C_FP=${args.cost_fp:.0f}")

        summary = run_evaluation(
            data_dir=args.data_dir,
            model_path=args.model_path,
            metadata_path=args.metadata_path,
            report_dir=args.report_dir,
            cost_fn=args.cost_fn,
            cost_fp=args.cost_fp,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        def_m = summary.default_metrics
        opt_m = summary.cost_optimal_metrics

        print(f"\nHoldout Evaluation Completed Successfully!")
        print(f"  ROC-AUC: {summary.roc_auc:.4f}")
        print(f"  PR-AUC:  {summary.pr_auc:.4f}")
        print(f"  Brier Score: {summary.brier_score:.4f}")
        print(f"\nThreshold Optimization Summary:")
        print(f"  Default Cutoff (0.50):       Recall: {def_m.recall:.2%}, Loss: ${def_m.cost:,.0f}")
        print(f"  Optimal Cutoff ({summary.optimal_threshold:.2f}):      Recall: {opt_m.recall:.2%}, Loss: ${opt_m.cost:,.0f}")
        print(f"  Net Financial Savings:       ${summary.cost_savings:,.2f}")
        print(f"\nArtifacts Created:")
        print(f"  Markdown Report: {summary.report_path}")
        print(f"  Diagnostic Plots ({len(summary.figures)}): {Path(args.report_dir) / 'figures'}")
        return 0

    if args.subcommand == "explain":
        print("Starting model explainability analysis (SHAP TreeExplainer)...")
        print(f"  Model artifact: {args.model_path}")
        print(f"  Data directory: {args.data_dir}")
        print(f"  Output directory: {args.output_dir}")
        print(f"  Sample size: {args.max_samples}")

        result = run_explanation(
            data_dir=args.data_dir,
            model_path=args.model_path,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            random_state=args.random_state,
        )

        print(f"\nModel Explainability Completed Successfully!")
        print(f"  Samples explained: {result.samples_explained}")
        print(f"\nTop Parent Feature Importances (Aggregated Mean |SHAP|):")
        print("=" * 60)
        print(f"{'Rank':<5} {'Parent Feature':<30} {'Mean |SHAP|':<15}")
        print("-" * 60)
        for i, row in result.grouped_importance.head(10).iterrows():
            print(f"{i+1:<5} {row['parent_feature']:<30} {row['mean_abs_shap']:.4f}")
        print("=" * 60)
        print(f"\nGenerated Diagnostic Figures ({len(result.generated_figures)}):")
        for fig in result.generated_figures:
            print(f"  - {fig}")
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
        print(f"Launching ChurnOps Streamlit Dashboard on http://{args.host}:{args.port}...")
        try:
            return subprocess.call(cmd)
        except KeyboardInterrupt:
            print("\nDashboard server terminated by user.")
            return 0

    if args.subcommand == "serve":
        import uvicorn
        from final_project.serving.api import app as fastapi_app

        print(f"Starting ChurnOps REST API inference server on http://{args.host}:{args.port}...")
        try:
            uvicorn.run(
                "final_project.serving.api:app" if args.reload else fastapi_app,
                host=args.host,
                port=args.port,
                reload=args.reload,
            )
            return 0
        except KeyboardInterrupt:
            print("\nServing server terminated by user.")
            return 0

    if args.subcommand == "simulate":
        print(f"Starting weekly streaming simulation ({args.n_weeks} weeks)...")
        print(f"  Data directory: {args.data_dir}")
        print(f"  Models directory: {args.models_dir}")
        print(f"  Artifact saving: {'Disabled' if args.no_save else 'Enabled'}")

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

        print(f"\nSimulation Completed Successfully!")
        print(f"  Weeks executed: {len(results.drift_metrics)}")
        print(f"  Retraining triggered at weeks: {results.retrained_weeks}")
        if results.final_retrain_auc_recent is not None:
            print(f"  Final Retrained AUC (recent live): {results.final_retrain_auc_recent:.4f}")
        return 0

    print(f"Unknown subcommand: {args.subcommand}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
