import argparse
import sys
from pathlib import Path

from final_project.data import load_and_split_data
from final_project.models import benchmark_candidates
from final_project.pipeline import run_training


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
        choices=["random_forest", "lightgbm", "xgboost", "rf", "lgb", "xgb"],
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
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
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
        print(f"  Algorithm: {args.algorithm} (balanced={args.balanced})")
        print(f"  Stratified CV Folds: {args.cv_folds}")

        summary = run_training(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            algorithm=args.algorithm,
            balanced=args.balanced,
            test_size=args.test_size,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
        )

        print(f"\nTraining completed successfully!")
        print(f"  Algorithm: {summary.algorithm}")
        print(f"  Balanced: {summary.balanced}")
        print(f"  Train samples: {summary.train_samples}")
        print(f"  Test samples: {summary.test_samples}")
        print(f"  Features: {summary.feature_count}")
        print(f"  CV Mean ROC-AUC: {summary.cv_auc_mean:.4f} (+/- {summary.cv_auc_std:.4f})")
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

        df = benchmark_candidates(
            X=split.X_train,
            y=split.y_train,
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

    print(f"Unknown subcommand: {args.subcommand}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
