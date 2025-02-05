import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import pandas as pd
import numpy as np

def get_correlated_features(response_df, feature_df, executor=None):
    y = response_df["LFC"].values
    ids = response_df["depmap_id"].values

    # ensure the feature df index aligns with ids
    feature_sub_df = feature_df.loc[ids]
    feature_names = feature_sub_df.columns.to_numpy()
    X = feature_sub_df.values

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

def get_top_p_features(feat_corr_df, p):
    # drop NaN features
    feat_corr_df = feat_corr_df.dropna()
    # drop insignificant features
    feat_corr_df = feat_corr_df[feat_corr_df.p_value < .05]
    # assign feature class
    feat_corr_df['feature_class'] = feat_corr_df.feature_name.str.split('_').str[0]
    # absolute value correlation
    feat_corr_df['correlation'] = feat_corr_df.correlation.abs()
    # per feature class, select top p percent of features
    top_p_features = []
    for feature_class, group in feat_corr_df.groupby('feature_class'):
        n = int(p * len(group))
        top_p_features.append(group.nlargest(n, 'correlation'))
    return pd.concat(top_p_features)