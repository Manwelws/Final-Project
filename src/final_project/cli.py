import argparse
import sys
from pathlib import Path
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
        print(f"  Stratified CV Folds: {args.cv_folds}")

        summary = run_training(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            test_size=args.test_size,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
        )

        print(f"\nTraining completed successfully!")
        print(f"  Algorithm: {summary.algorithm}")
        print(f"  Train samples: {summary.train_samples}")
        print(f"  Test samples: {summary.test_samples}")
        print(f"  Features: {summary.feature_count}")
        print(f"  CV Mean ROC-AUC: {summary.cv_auc_mean:.4f} (+/- {summary.cv_auc_std:.4f})")
        print(f"  Fold Scores: {[round(s, 4) for s in summary.cv_fold_scores]}")
        print(f"  Model saved to: {summary.model_path}")
        print(f"  Metadata saved to: {summary.metadata_path}")
        return 0

    print(f"Unknown subcommand: {args.subcommand}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
