import argparse
import os
import sys

# get path to ../refract
refract_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(refract_path)

from refract.trainers import QuantileRegressionTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train Quantile Regression model using cross-validation"
    )
    parser.add_argument(
        "--feature_file", type=str, required=True, help="Path to feature file"
    )
    parser.add_argument(
        "--response_file", type=str, required=True, help="Path to response file"
    )
    parser.add_argument(
        "--split_dir", type=str, required=True, help="Directory containing split files"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save results"
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        default=8,
        help="Number of threads (not used for quantile regression)",
    )
    parser.add_argument(
        "--n_splits", type=int, default=10, help="Number of cross-validation splits"
    )

    # Quantile regression hyperparameters
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.1,
        help="Quantile to estimate (default: 0.1 for bottom 10%)",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.0,0.001,0.01,0.1,1.0,10.0",
        help="Comma-separated alpha values for regularization",
    )
    parser.add_argument(
        "--solvers",
        type=str,
        default="highs,interior-point",
        help="Comma-separated solver names to try",
    )

    args = parser.parse_args()

    # Parse alpha values
    alphas = [float(alpha.strip()) for alpha in args.alphas.split(",")]

    # Parse solvers
    solvers = [solver.strip() for solver in args.solvers.split(",")]

    # Initialize trainer
    trainer = QuantileRegressionTrainer(
        output_dir=args.output_dir,
        n_threads=args.n_threads,
        quantile=args.quantile,
        alphas=alphas,
        solvers=solvers,
    )

    # Train models using cross-validation
    _ = trainer.train_cross_validation(
        feature_file=args.feature_file,
        response_file=args.response_file,
        split_dir=args.split_dir,
        n_splits=args.n_splits,
    )

    # save the feature importance
    trainer.save_feature_importance()

    print(
        f"\nQuantile Regression (quantile={args.quantile}) model training completed successfully!"
    )
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
