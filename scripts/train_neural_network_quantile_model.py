import argparse
import os
import sys

# get path to ../refract
refract_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(refract_path)

from refract.trainers import NeuralNetworkQuantileTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train Neural Network Quantile Regression model using cross-validation"
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
    parser.add_argument("--n_threads", type=int, default=8, help="Number of threads")
    parser.add_argument(
        "--n_splits", type=int, default=10, help="Number of cross-validation splits"
    )

    # Neural network hyperparameters
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.1,
        help="Quantile to estimate - default 0.1 for bottom 10 percent",
    )
    parser.add_argument(
        "--hidden_sizes",
        type=str,
        default="128,64,32",
        help="Comma-separated hidden layer sizes",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for optimization",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.2,
        help="Dropout rate for regularization",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=200, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="Early stopping patience"
    )

    args = parser.parse_args()

    # Parse hidden layer architecture
    hidden_sizes = [int(size.strip()) for size in args.hidden_sizes.split(",")]

    # Initialize trainer
    trainer = NeuralNetworkQuantileTrainer(
        output_dir=args.output_dir,
        n_threads=args.n_threads,
        quantile=args.quantile,
        hidden_sizes=hidden_sizes,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        dropout_rate=args.dropout_rate,
        n_epochs=args.n_epochs,
        patience=args.patience,
    )

    # Train models using cross-validation
    _ = trainer.train_cross_validation(
        feature_file=args.feature_file,
        response_file=args.response_file,
        split_dir=args.split_dir,
        n_splits=args.n_splits,
    )

    # save the feature importance
    try:
        trainer.save_feature_importance()
        print("Feature importance computation completed")
    except Exception as e:
        print(f"Warning: Feature importance computation failed: {e}")
        print("Continuing without feature importance analysis...")

    print(
        f"\nNeural Network Quantile Regression - quantile={args.quantile} model training completed successfully!"
    )
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
