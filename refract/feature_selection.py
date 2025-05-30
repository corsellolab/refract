import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import pandas as pd
import numpy as np

def get_kendall_tau_correlations(response_df, feature_df, executor=None):
    # Ensure both dataframes have the same indices and are aligned
    common_ids = response_df.index.intersection(feature_df.index)
    response_df = response_df.loc[common_ids]
    feature_df = feature_df.loc[common_ids]
    
    y = response_df["LFC"].values
    feature_names = feature_df.columns.to_numpy()
    X = feature_df.values
    n_features = X.shape[1]

    def compute_kendall(idx):
        x_col = X[:, idx]
        feature_name = feature_names[idx]
        not_missing = ~np.isnan(x_col)
        x_clean = x_col[not_missing]
        y_clean = y[not_missing]
        corr, p = kendalltau(y_clean, x_clean, nan_policy='omit')
        return {
            "feature_name": feature_name,
            "correlation": corr,
            "p_value": p
        }

    if executor is None:
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(compute_kendall, range(n_features)),
                total=n_features
            ))
    else:
        results = list(tqdm(
            executor.map(compute_kendall, range(n_features)),
            total=n_features
        ))

    return pd.DataFrame(results)

def get_pearson_correlations(response_df, feature_df):
    # Ensure both dataframes have the same indices and are aligned
    common_ids = response_df.index.intersection(feature_df.index)
    response_df = response_df.loc[common_ids]
    feature_df = feature_df.loc[common_ids]
    
    y = response_df["LFC"].values
    feature_names = feature_df.columns.to_numpy()
    X = feature_df.values
    n_features = X.shape[1]

    def compute_pearson(idx):
        x_col = X[:, idx]
        feature_name = feature_names[idx]
        not_missing = ~np.isnan(x_col)
        x_clean = x_col[not_missing]
        y_clean = y[not_missing]

        if len(x_clean) < 10:
            return {
                "feature_name": feature_name,
                "correlation": np.nan,
                "p_value": np.nan
            }
        
        corr, p = pearsonr(y_clean, x_clean)
        return {
            "feature_name": feature_name,
            "correlation": corr,
            "p_value": p
        }
    
    results = list(
        map(compute_pearson, range(n_features)),
        total=n_features
    )
    
    corrs = pd.DataFrame(results)
    corrs = corrs.dropna()
    return corrs

def get_top_n_features(feat_corr_df, n):
    # drop NaN features
    feat_corr_df = feat_corr_df.dropna()
    # drop insignificant features
    feat_corr_df = feat_corr_df[feat_corr_df.p_value < .05]
    # assign feature class
    feat_corr_df['feature_class'] = feat_corr_df.feature_name.str.split('_').str[0]
    # per feature class, select top n of features
    top_n_features = []
    for feature_class, group in feat_corr_df.groupby('feature_class'):
        if n < 1:
            n = 1
        if "OMIC" in feature_class or "ONC" in feature_class:
            # use all features
            top_n_features.append(group)
        else:
            # get the n smallest p values
            top_n_features.append(group.nsmallest(n, 'p_value'))
    return pd.concat(top_n_features)