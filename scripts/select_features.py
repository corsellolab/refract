import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# get path to ../refract
refract_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(refract_path)

from refract.feature_selection import get_correlated_features, get_top_p_features
from refract.data_split import get_data_splits, get_data_for_split
from refract.utils import load_feature_df, load_response_df, intersect_depmap_ids

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Select features for each fold split')
    parser.add_argument('--feature_file', type=str, required=True, help='Path to feature file')
    parser.add_argument('--response_file', type=str, required=True, help='Path to response file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save fold splits and features')
    parser.add_argument('--n_splits', type=int, default=10, help='Number of splits for cross-validation')
    parser.add_argument('--feature_fraction', type=float, default=0.1, help='Fraction of features to select per class')
    parser.add_argument('--train_val_split', type=float, default=0.8, help='Fraction of data to use for training and validation')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and preprocess data
    feature_df = load_feature_df(args.feature_file)
    response_df = load_response_df(args.response_file)

    response_df, feature_df = intersect_depmap_ids(response_df, feature_df)

    # Add debug prints to check DataFrame sizes
    print(f"Response DataFrame shape before splits: {response_df.shape}")
    print(f"Unique deciles in response_df: {response_df['decile'].unique()}")
    
    # Get data splits
    splits = get_data_splits(response_df, n_splits=args.n_splits)
    
    # For each split
    for split_idx in range(len(splits)):
        # Get train/test data for this split
        train_response, test_response, train_features, test_features = get_data_for_split(
            response_df, feature_df, splits, split_idx
        )
        
        # Further split training data into train and validation sets
        train_response_final, val_response, train_features_final, val_features = train_test_split(
            train_response, 
            train_features,
            train_size=args.train_val_split,
            random_state=42 + split_idx  # Different seed for each fold
        )
        
        # Get correlations using only the final training data
        corr_df = get_correlated_features(train_response_final, train_features_final)
        
        # Select top features
        selected_features = get_top_p_features(corr_df, args.feature_fraction)
        
        # Save fold split assignments
        split_file = os.path.join(args.output_dir, f'{split_idx}.split.txt')
        with open(split_file, 'w') as f:
            f.write('depmap_id,split\n')
            for idx in train_response_final.index:
                f.write(f'{idx},train\n')
            for idx in val_response.index:
                f.write(f'{idx},val\n')
            for idx in test_response.index:
                f.write(f'{idx},test\n')
        
        # Save selected features
        features_file = os.path.join(args.output_dir, f'{split_idx}.features.csv')
        selected_features.to_csv(features_file, index=False)

if __name__ == "__main__":
    main()