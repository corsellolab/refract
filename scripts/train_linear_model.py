import argparse
import os
import sys

# get path to ../refract
refract_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(refract_path)

from refract.trainers import LinearTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train Linear model using cross-validation"
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
        help="Number of threads (not used for linear model)",
    )
    parser.add_argument(
        "--n_splits", type=int, default=10, help="Number of cross-validation splits"
    )

    # Linear model hyperparameters
    parser.add_argument(
        "--max_iter", type=int, default=10000, help="Maximum iterations for convergence"
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.001,0.01,0.1,1.0,10.0,100.0",
        help="Comma-separated alpha values for L1 regularization",
    )

    args = parser.parse_args()

    # Parse alpha values
    alphas = [float(alpha.strip()) for alpha in args.alphas.split(",")]

    # Initialize trainer
    trainer = LinearTrainer(
        output_dir=args.output_dir,
        n_threads=args.n_threads,
        alphas=alphas,
        max_iter=args.max_iter,
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

    print("\nLinear model training completed successfully!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
