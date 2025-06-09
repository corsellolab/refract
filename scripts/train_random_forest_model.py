import argparse
import os
import sys

# get path to ../refract
refract_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(refract_path)

from refract.trainers import RandomForestTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train Random Forest model using cross-validation"
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
        "--n_threads", type=int, default=8, help="Number of threads for Random Forest"
    )
    parser.add_argument(
        "--n_splits", type=int, default=10, help="Number of cross-validation splits"
    )

    # Random Forest hyperparameters
    parser.add_argument(
        "--n_estimators",
        type=str,
        default="100,200,500",
        help="Comma-separated n_estimators values to try",
    )
    parser.add_argument(
        "--max_depth",
        type=str,
        default="10,20,None",
        help="Comma-separated max_depth values to try",
    )
    parser.add_argument(
        "--min_samples_split",
        type=str,
        default="2,5,10",
        help="Comma-separated min_samples_split values to try",
    )
    parser.add_argument(
        "--min_samples_leaf",
        type=str,
        default="1,2,4",
        help="Comma-separated min_samples_leaf values to try",
    )
    parser.add_argument(
        "--max_features",
        type=str,
        default="sqrt,log2,None",
        help="Comma-separated max_features values to try",
    )

    args = parser.parse_args()

    # Parse parameter grid
    def parse_param_list(param_str):
        params = []
        for p in param_str.split(","):
            p = p.strip()
            if p == "None":
                params.append(None)
            elif p in ["sqrt", "log2"]:
                params.append(p)
            else:
                try:
                    params.append(int(p))
                except ValueError:
                    try:
                        params.append(float(p))
                    except ValueError:
                        params.append(p)
        return params

    param_grid = {
        "n_estimators": parse_param_list(args.n_estimators),
        "max_depth": parse_param_list(args.max_depth),
        "min_samples_split": parse_param_list(args.min_samples_split),
        "min_samples_leaf": parse_param_list(args.min_samples_leaf),
        "max_features": parse_param_list(args.max_features),
    }

    # Initialize trainer
    trainer = RandomForestTrainer(
        output_dir=args.output_dir, n_threads=args.n_threads, param_grid=param_grid
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

    print("\nRandom Forest model training completed successfully!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
