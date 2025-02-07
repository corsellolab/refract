import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    response_df = response_df.dropna(subset=['depmap_id', 'LFC', 'name', 'dose'])
    # drop duplicates of depmap_id, name, and dose
    response_df = response_df.drop_duplicates(subset=['depmap_id', 'name', 'dose'])
    # compute decile of LFC
    response_df["decile"] = pd.qcut(response_df.LFC, 10, labels=False)
    # set depmap_id as index
    response_df.set_index('depmap_id', inplace=True)
    return response_df

def intersect_depmap_ids(response_df, feature_df):
    # Add debug prints
    print(f"Initial response_df shape: {response_df.shape}")
    print(f"Initial feature_df shape: {feature_df.shape}")
    
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
