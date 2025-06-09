import argparse
import os
import sys

# get path to ../refract
refract_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(refract_path)

from refract.trainers import XGBoostTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost model using cross-validation"
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
        "--n_threads", type=int, default=8, help="Number of threads for XGBoost"
    )
    parser.add_argument(
        "--num_rounds", type=int, default=1000, help="Maximum number of training rounds"
    )
    parser.add_argument(
        "--early_stopping_rounds", type=int, default=50, help="Early stopping rounds"
    )
    parser.add_argument(
        "--n_splits", type=int, default=10, help="Number of cross-validation splits"
    )

    # XGBoost hyperparameters
    parser.add_argument("--eta", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--max_depth", type=int, default=6, help="Maximum tree depth")
    parser.add_argument(
        "--subsample", type=float, default=0.8, help="Row subsampling ratio"
    )
    parser.add_argument(
        "--colsample_bytree", type=float, default=0.8, help="Feature subsampling ratio"
    )
    parser.add_argument(
        "--lambda_reg", type=float, default=1.0, help="L2 regularization"
    )
    parser.add_argument(
        "--alpha_reg", type=float, default=0.1, help="L1 regularization"
    )

    args = parser.parse_args()

    # Prepare XGBoost parameters
    xgb_params = {
        "eta": args.eta,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "lambda": args.lambda_reg,
        "alpha": args.alpha_reg,
    }

    # Initialize trainer
    trainer = XGBoostTrainer(
        output_dir=args.output_dir,
        n_threads=args.n_threads,
        num_rounds=args.num_rounds,
        early_stopping_rounds=args.early_stopping_rounds,
        **xgb_params,
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

    print("\nTraining completed successfully!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
