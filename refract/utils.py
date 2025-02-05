import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def load_feature_df(feature_path):
    feature_df = pd.read_pickle(feature_path)
    # rename first column as depmap_id
    feature_df.rename(columns={feature_df.columns[0]: 'depmap_id'}, inplace=True)
    # set depmap_id as index
    feature_df.set_index('depmap_id', inplace=True)

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
    depmap_ids = set(response_df.depmap_id.values)
    feature_depmap_ids = set(feature_df.index.values)
    intersecting_depmap_ids = depmap_ids.intersection(feature_depmap_ids)

    # subset both
    response_df = response_df.loc[intersecting_depmap_ids]
    feature_df = feature_df.loc[intersecting_depmap_ids]

    return response_df, feature_df
