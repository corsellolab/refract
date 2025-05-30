import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

# get path to ../refract
refract_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(refract_path)

from refract.feature_selection import get_pearson_correlations, get_top_n_features
from refract.data_split import get_data_splits, get_data_for_split
from refract.utils import load_feature_df, load_response_df, intersect_depmap_ids

def process_split(split_idx, train_response, val_response, test_response, train_features, args):
    """Process a single split"""
    # Get correlations using only the final training data
    corr_df = get_pearson_correlations(train_response, train_features)
    
    # Select top features
    selected_features = get_top_n_features(corr_df, args.n_features)
    
    # Save fold split assignments
    split_file = os.path.join(args.output_dir, f'{split_idx}.split.txt')
    with open(split_file, 'w') as f:
        f.write('depmap_id,split\n')
        for idx in train_response.index:
            f.write(f'{idx},train\n')
        for idx in val_response.index:
            f.write(f'{idx},val\n')
        for idx in test_response.index:
            f.write(f'{idx},test\n')
    
    # Save selected features
    features_file = os.path.join(args.output_dir, f'{split_idx}.features.csv')
    selected_features.to_csv(features_file, index=False)
    
    return split_idx

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Select features for each fold split')
    parser.add_argument('--feature_file', type=str, required=True, help='Path to feature file')
    parser.add_argument('--response_file', type=str, required=True, help='Path to response file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save fold splits and features')
    parser.add_argument('--n_splits', type=int, default=10, help='Number of splits for cross-validation')
    parser.add_argument('--n_features', type=int, default=50, help='Number of features to select per class')
    parser.add_argument('--train_val_split', type=float, default=0.8, help='Fraction of data to use for training and validation')
    #parser.add_argument('--n_jobs', type=int, default=1, help='Number of jobs to run in parallel')
    
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
    
    # Process splits in parallel
    # Create a list of all split data upfront
    """
    split_data = []
    
    for split_idx in range(len(splits)):
        train_response, test_response, train_features, _ = get_data_for_split(
            response_df, feature_df, splits, split_idx
        )
        train_response_final, val_response, train_features_final, val_features = train_test_split(
            train_response, 
            train_features,
            train_size=args.train_val_split,
            random_state=42 + split_idx
        )
        split_data.append((split_idx, train_response_final, val_response, test_response, train_features_final))

    # Process all splits in parallel using map
    with ThreadPoolExecutor(max_workers=args.n_jobs) as executor:
        results = list(
            executor.map(
                lambda x: process_split(*x, args),
                split_data
            ) 
        )
    
    # Print completion messages
    for split_idx in results:
        print(f"Completed split {split_idx}")
    """
    # Process splits sequentially
    for split_idx in range(len(splits)):
        train_response, test_response, train_features, _ = get_data_for_split(
            response_df, feature_df, splits, split_idx
        )
        train_response_final, val_response, train_features_final, val_features = train_test_split(
            train_response, 
            train_features,
            train_size=args.train_val_split,
            random_state=42 + split_idx
        )
        process_split(split_idx, train_response_final, val_response, test_response, train_features_final, args)


if __name__ == "__main__":
    main()