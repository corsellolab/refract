#!/usr/bin/env python3
"""
Example script demonstrating how to use the refactored trainer classes.

This script shows how to use the XGBoostTrainer class for training models
with cross-validation, feature importance analysis, and prediction.
"""

import os
import sys

# Add the refract package to the path
refract_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(refract_path)

from refract.trainers import XGBoostTrainer


def example_basic_usage():
    """Example of basic XGBoost trainer usage."""
    print("=== Basic XGBoost Trainer Usage ===")
    
    # Initialize the trainer with custom parameters
    trainer = XGBoostTrainer(
        output_dir="./results",
        n_threads=4,
        num_rounds=500,
        early_stopping_rounds=25,
        eta=0.05,  # Custom learning rate
        max_depth=4,  # Custom tree depth
    )
    
    # Train models using cross-validation
    results = trainer.train_cross_validation(
        feature_file="path/to/features.pkl",
        response_file="path/to/responses.csv", 
        split_dir="path/to/splits",
        n_splits=5  # Use 5-fold CV instead of default 10
    )
    
    print(f"Training completed! Results saved to: {trainer.output_dir}")
    print(f"Number of trained models: {len(trainer.models)}")
    
    return results


def example_custom_parameters():
    """Example with custom XGBoost hyperparameters."""
    print("\n=== Custom Hyperparameters Example ===")
    
    # Initialize trainer with custom hyperparameters
    trainer = XGBoostTrainer(
        output_dir="./custom_results",
        n_threads=8,
        num_rounds=2000,
        early_stopping_rounds=100,
        # Custom XGBoost parameters
        eta=0.01,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        lambda_reg=2.0,
        alpha_reg=0.5,
        # Additional XGBoost parameters
        gamma=0.1,
        min_child_weight=3
    )
    
    print("Trainer initialized with custom parameters:")
    print(f"Learning rate: {trainer.default_params['eta']}")
    print(f"Max depth: {trainer.default_params['max_depth']}")
    print(f"L2 regularization: {trainer.default_params['lambda']}")
    
    return trainer


def example_single_model_training():
    """Example of training a single model (not cross-validation)."""
    print("\n=== Single Model Training Example ===")
    
    # This would be used if you have your own train/val split
    # and just want to train a single model
    
    trainer = XGBoostTrainer(output_dir="./single_model_results")
    
    # Assuming you have your data loaded
    # X_train, y_train, X_val, y_val = load_your_data()
    
    # Train a single model
    # model = trainer.train_single_model(X_train, y_train, X_val, y_val)
    
    # Make predictions
    # predictions = trainer.predict(model, X_test)
    
    # Compute feature importance
    # importance = trainer.compute_feature_importance(model, X_test)
    
    print("Single model training workflow demonstrated")


def main():
    """Main function demonstrating different usage patterns."""
    print("XGBoost Trainer Examples")
    print("=" * 50)
    
    # Example 1: Basic usage
    try:
        # Uncomment the following line when you have actual data files
        # results = example_basic_usage()
        print("Basic usage example ready (uncomment when data is available)")
    except Exception as e:
        print(f"Basic usage example would run with real data: {e}")
    
    # Example 2: Custom parameters
    trainer = example_custom_parameters()
    
    # Example 3: Single model training
    example_single_model_training()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo use with your data:")
    print("1. Update the file paths in example_basic_usage()")
    print("2. Ensure your data files are in the correct format")
    print("3. Run the script")


if __name__ == "__main__":
    main() 