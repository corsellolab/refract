import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr
from scipy import stats
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

def get_pearson_correlations(response_df: pd.DataFrame | pd.Series,
                             feature_df: pd.DataFrame,
                             *,
                             target_col: str = "LFC",
                             min_non_nan: int = 10) -> pd.DataFrame:
    """
    Vectorised Pearson r (and two-sided p-value) between one response vector
    and every column in `feature_df`.

    Parameters
    ----------
    response_df : (n, 1) DataFrame **or** Series
        Must contain the response column `target_col` if a DataFrame.
    feature_df  : (n, p) DataFrame
        Each column is a feature.
    target_col  : str, default "LFC"
        Name of the response column if `response_df` is a DataFrame.
    min_non_nan : int, default 10
        Skip features with < `min_non_nan` paired, non-missing observations.

    Returns
    -------
    pd.DataFrame with columns
        feature_name | correlation | p_value | n
    """
    # ── 1  Align rows ──────────────────────────────────────────────────────────
    if isinstance(response_df, pd.Series):
        y = response_df
    else:
        y = response_df[target_col]

    idx = y.index.intersection(feature_df.index)
    y = y.loc[idx]
    X = feature_df.loc[idx]

    # ── 2  Vectorised correlation coefficients (Pandas uses C code under the hood)
    r = X.corrwith(y)                                # Series (length p)

    # ── 3  Effective N per feature (pairwise deletion for NaNs), vectorised
    valid_y = ~y.isna().values                       # (n,)
    n = (X.notna().values & valid_y[:, None]).sum(0) # (p,)

    # ── 4  t statistic and two‑sided p‑value, fully vectorised
    df = n - 2                                       # degrees of freedom
    with np.errstate(divide="ignore", invalid="ignore"):
        t = r.values * np.sqrt(df / (1.0 - r.values**2))
    p = 2.0 * stats.t.sf(np.abs(t), df)

    # ── 5  Assemble result and apply minimum‑N filter
    out = pd.DataFrame({
        "feature_name": X.columns,
        "correlation" : r.values,
        "p_value"     : p,
        "n"           : n
    })
    out.loc[out["n"] < min_non_nan, ["correlation", "p_value"]] = np.nan

    # Optional: drop rows that are all‑NaN after filtering
    return out.dropna(subset=["correlation"])

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