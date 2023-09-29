import logging
import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from pydantic import BaseModel
from scipy.stats import pearsonr
from tqdm import tqdm

from refract.metrics import (
    get_merged_shap_values_and_features,
    get_stringdb_network_interactions,
    get_test_predictions,
    get_top_k_features,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            value = self[name]
            if isinstance(value, dict):
                return AttrDict(value)
            else:
                return value
        except KeyError:
            raise AttributeError(f"'{name}' is not a valid attribute")


class RandomForestNestedCVConfig(BaseModel):
    # only use all features
    feature_name_list: List[str] = ["all"]
    # CV Grid search params
    param_grid: Dict[str, List] = {
        "n_estimators": [50],
        "max_depth": [3],
        "max_features": [
            1.0,
        ],
    }
    # CV details
    n_splits: int = 5
    random_state: int = 42
    n_jobs: int = 15
    cv_n_jobs: int = 1


class RandomForestCVConfig(BaseModel):
    # only use all features
    feature_name_list: List[str] = ["all"]
    # CV details
    n_estimators: int = 50
    max_depth: int = 3
    max_features: float = 1.0
    n_splits: int = 5
    random_state: int = 42
    n_jobs: int = 15
    cv_n_jobs: int = 1


class WeightedRandomForestNestedCVConfig(RandomForestNestedCVConfig):
    param_grid: Dict[str, List] = {
        "n_estimators": [50],
        "max_depth": [3],
        "alpha": [
            1.0,
        ],
        "max_features": [
            1.0,
        ],
    }


class WeightedRandomForestCVConfig(RandomForestCVConfig):
    alpha: float = 1.0


class XGBoostCVConfig(RandomForestNestedCVConfig):
    n_jobs: int = 10
    n_estimators = 200
    max_depth = 10
    random_state = 42
    n_jobs = 12
    objective = "rank:pairwise"


class LGBMCVConfig(RandomForestNestedCVConfig):
    param_grid: Dict[str, List] = {
        "num_leaves": [4, 32, 64],
        "max_depth": [2, 4, 8],
    }
    n_jobs: int = 10


def get_top_features(response_df, feature_df, response_col, p):
    df = response_df.merge(feature_df, on="ccle_name")
    df_features = df.loc[:, feature_df.columns]
    series_response = df.loc[:, response_col]

    corrs = []
    print("Correlating features with response...")
    for col in tqdm(df_features.columns):
        c = pearsonr(df_features[col], series_response)[0]
        # get absolute value of c
        c = abs(c)
        feature_type = col.split("_")[0]
        corrs.append({"corr": c, "feature_type": feature_type, "feature": col})

    corr_df = pd.DataFrame(corrs)

    # get all LIN features
    lin_df = corr_df.loc[corr_df.feature_type == "LIN", :]

    top_correlated = (
        corr_df.groupby("feature_type")
        .apply(lambda group: group.nlargest(int(len(group) * p), "corr"))
        .reset_index(level=0, drop=True)
    )

    top_features = top_correlated.feature.tolist()
    top_features.extend(lin_df.feature.tolist())
    top_features = list(set(top_features))

    return top_features


def dataset_to_group_df(ds, num_epochs):
    features = []
    labels = []
    groups = []
    for _ in range(num_epochs):
        for ex in ds:
            _, feat, label = ex
            # convert to numpy
            feat = feat.numpy()
            label = label.numpy()
            features.append(feat)
            labels.append(label)
            groups.append(label.shape)
    group_train_features = np.concatenate(features, axis=0)
    group_train_labels = np.concatenate(labels, axis=0)
    groups = np.array(groups)

    return group_train_features, group_train_labels, groups


def dataset_to_individual_df(ds):
    features = []
    labels = []
    ccle_names = []
    for ex in ds:
        ccle_name, feat, label = ex
        feat = feat.numpy()
        label = label.numpy()
        features.append(feat[0, :].reshape(1, -1))
        labels.append(label[0])
        ccle_names.append(ccle_name)
    individual_train_features = np.concatenate(features, axis=0)
    individual_train_labels = np.array(labels)
    return individual_train_features, individual_train_labels, ccle_names


def save_output(trainers, output_dir):
    # compute SHAP values and predictions across the full dataset
    logger.info("Aggregating SHAP values and predictions...")
    shap_values, features, feature_names = get_merged_shap_values_and_features(trainers)
    test_df = get_test_predictions(trainers)

    # save test_df to file
    logger.info("Saving training results to train_results.csv...")
    test_df.to_csv(os.path.join(output_dir, "train_results.csv"), index=False)

    # plot a scatter plot of predictions vs actual
    logger.info("Plotting scatterplot to train_results.png...")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(test_df["label"], test_df["pred"], alpha=0.5)
    ax.set_xlabel("LFC")
    ax.set_ylabel("Ranking Score")
    fig.savefig(os.path.join(output_dir, "train_results.png"))
    plt.close()

    # compute pearson correlation between pred and true
    logger.info("Computing pearson correlation...")
    train_corr = np.corrcoef(test_df["label"], test_df["pred"])[0, 1]
    logger.info(f"Train correlation: {train_corr}")
    with open(os.path.join(output_dir, "train_corr.txt"), "w") as f:
        f.write(str(train_corr))

    # save SHAP summary plot
    logger.info("Saving SHAP summary plot to shap_summary_plot.png...")
    shap.summary_plot(shap_values, features, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"))
    plt.close()

    # get the gene name of top features
    logger.info("Getting top features...")
    top_feature_names = get_top_k_features(shap_values, feature_names, k=20)
    top_feature_genes = [i.split("_")[1] for i in top_feature_names]

    # get connectivity of top features
    logger.info("Getting network interactions...")
    network_interactions, _ = get_stringdb_network_interactions(top_feature_genes)
    network_interactions.to_csv(
        os.path.join(output_dir, "network_interactions.csv"), index=False
    )

    # save trainers
    logger.info("Saving trainers to trainers.pkl...")
    with open(os.path.join(output_dir, "trainers.pkl"), "wb") as f:
        pickle.dump(trainers, f)


def moving_window_average(lst, window_size=3):
    if window_size % 2 == 0:
        raise ValueError("Please provide an odd window size.")

    half_window = window_size // 2
    extended_list = [lst[0]] * half_window + lst + [lst[-1]] * half_window

    averages = []
    for i in range(half_window, len(extended_list) - half_window):
        avg = sum(extended_list[i - half_window : i + half_window + 1]) / window_size
        averages.append(avg)

    return averages
