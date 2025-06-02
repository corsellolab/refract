import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from glob import glob

def load_feature_df(feature_path):
    feature_df = pd.read_pickle(feature_path)
    # rename index to depmap_id
    feature_df.index.name = 'depmap_id'
    # drop columns where all values are NaN
    feature_df = feature_df.dropna(axis=1, how='all')
    # drop columns where all values are the same
    feature_df = feature_df.loc[:, (feature_df != feature_df.iloc[0]).any()]
    return feature_df

def load_response_df(response_path):
    response_df = pd.read_csv(response_path)
    # drop where depmap_id, LFC, name, or dose is missing
    if "dose" in response_df.columns:
        response_df = response_df.dropna(subset=['depmap_id', 'LFC', 'name', 'dose'])
    else:
        response_df = response_df.dropna(subset=['depmap_id', 'LFC', 'name'])
    # drop duplicates of depmap_id, name, and dose
    if "dose" in response_df.columns:
        response_df = response_df.drop_duplicates(subset=['depmap_id', 'name', 'dose'])
    else:
        response_df = response_df.drop_duplicates(subset=['depmap_id', 'name'])
    # compute decile of LFC
    response_df["decile"] = pd.qcut(response_df.LFC, 10, labels=False)
    # set depmap_id as index
    response_df.set_index('depmap_id', inplace=True)
    return response_df

def intersect_depmap_ids(response_df, feature_df):
    response_df = response_df.copy()
    # Add debug prints
    print(f"Initial response_df shape: {response_df.shape}")
    print(f"Initial feature_df shape: {feature_df.shape}")

    # drop where LFC is NaN in response_df
    response_df = response_df.dropna(subset=['LFC'])
    
    depmap_ids = set(response_df.index.values)
    feature_depmap_ids = set(feature_df.index.values)
    intersecting_depmap_ids = depmap_ids.intersection(feature_depmap_ids)
    
    # Add validation
    if len(intersecting_depmap_ids) == 0:
        raise ValueError("No overlapping DepMap IDs found between response and feature data")
    
    # Convert the set to a list before using it as an indexer
    intersecting_depmap_ids = list(intersecting_depmap_ids)
    
    response_df = response_df.loc[intersecting_depmap_ids]
    feature_df = feature_df.loc[intersecting_depmap_ids]
    
    # Add debug prints
    print(f"Final response_df shape: {response_df.shape}")
    print(f"Final feature_df shape: {feature_df.shape}")
    
    return response_df, feature_df

def load_split(split_file):
    """Load train/val/test split assignments from a split file."""
    split_df = pd.read_csv(split_file)
    train_ids = split_df[split_df['split'] == 'train']['depmap_id'].values
    val_ids = split_df[split_df['split'] == 'val']['depmap_id'].values
    test_ids = split_df[split_df['split'] == 'test']['depmap_id'].values
    return train_ids, val_ids, test_ids

def load_selected_features(features_file):
    """Load selected features from features file."""
    features_df = pd.read_csv(features_file)
    return features_df['feature_name'].values

def evaluate_predictions(y_true, y_pred, set_name):
    """Calculate and print evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    pearson = np.corrcoef(y_true, y_pred)[0,1]
    print(f"{set_name} RMSE: {rmse:.4f}")
    print(f"{set_name} R2: {r2:.4f}")
    print(f"{set_name} Pearson: {pearson:.4f}")
    return {'rmse': rmse, 'r2': r2, 'pearson': pearson}