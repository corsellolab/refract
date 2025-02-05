import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


def get_data_splits(response_df, n_splits=5):
    # Add input validation
    if response_df.empty:
        raise ValueError("Response DataFrame is empty")
    
    # Ensure we have at least one sample per decile
    value_counts = response_df['decile'].value_counts()
    if (value_counts < 1).any():
        raise ValueError(f"Some deciles have no samples. Value counts:\n{value_counts}")
    
    # Create stratified k-fold split
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Get unique combinations of depmap_id and decile
    response_df_unique = response_df.reset_index().drop_duplicates(['depmap_id', 'decile'])
    
    splits = []
    for train_index, test_index in skf.split(response_df_unique, response_df_unique.decile):
        # Store the depmap_ids for train and test
        train_ids = response_df_unique.iloc[train_index]['depmap_id'].values
        test_ids = response_df_unique.iloc[test_index]['depmap_id'].values
        splits.append((train_ids, test_ids))
    
    return splits

def get_data_for_split(response_df, feature_df, data_splits, split_index):
    train_index, test_index = data_splits[split_index]
    train_response_df = response_df.loc[train_index]
    test_response_df = response_df.loc[test_index]
    train_feature_df = feature_df.loc[train_index]
    test_feature_df = feature_df.loc[test_index]
    return train_response_df, test_response_df, train_feature_df, test_feature_df
