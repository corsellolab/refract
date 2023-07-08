# Run training script for a single compound
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import logging

import xgboost as xgb

from refract.datasets import FeatureSet, PrismDataset, ResponseSet
from refract.metrics import (
    get_merged_shap_values_and_features,
    get_stringdb_network_interactions,
    get_test_predictions,
    get_top_k_features,
)
from refract.trainers import XGBoostRankingTrainer
from refract.utils import torch_dataset_to_numpy_array

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

SLATE_LENGTH = 10
NUM_EPOCHS = 1
NUM_TREES = 100


def run(response_path, feature_path, output_dir):
    # create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # update logger to write to file
    fh = logging.FileHandler(os.path.join(output_dir, "train.log"))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # load response data
    logger.info("Loading respose data...")
    response_set = ResponseSet(response_path)
    response_set.load_response_table()
    response_df = response_set.get_response_df(dose=2.5)

    # load feature data
    feature_set = FeatureSet(feature_path)
    feature_set.load_concatenated_feature_tables()
    feature_df = feature_set.get_feature_df("all")

    trainers = []
    outer_splitter = KFold(n_splits=10, shuffle=True, random_state=42)

    for idx, (train_val_index, test_index) in enumerate(
        outer_splitter.split(response_df)
    ):
        logger.info("Training fold: {}".format(idx))
        train_val = response_df.iloc[train_val_index, :].copy()
        train_index, val_index = next(
            KFold(n_splits=4, shuffle=True, random_state=42).split(train_val)
        )

        train = train_val.iloc[train_index].copy()
        val = train_val.iloc[val_index].copy()
        test = response_df.iloc[test_index].copy()

        # check no overlap
        assert len(set(train.index).intersection(set(val.index))) == 0
        assert len(set(train.index).intersection(set(test.index))) == 0
        assert len(set(val.index).intersection(set(test.index))) == 0

        ds_train = PrismDataset(train, feature_df, 10)
        ds_val = PrismDataset(val, feature_df, 10)
        ds_test = PrismDataset(test, feature_df, 10)

        trainer = XGBoostRankingTrainer(ds_train, ds_val, ds_test, num_epochs=1)

        trainer.train()
        trainer.compute_stats()
        trainer.log_stats()

        trainers.append(trainer)

    # compute SHAP values and predictions across full dataset
    shap_values, features, feature_names = get_merged_shap_values_and_features(trainers)
    test_df = get_test_predictions(trainers)

    # save test_df to file
    test_df.to_csv(os.path.join(output_dir, "train_results.csv"), index=False)

    # plot a scatter plot of predictions vs actual
    plt.figure(figsize=(5, 5))
    plt.scatter(test_df["label"], test_df["pred"], alpha=0.5)
    plt.xlabel("LFC")
    plt.ylabel("Ranking Score")
    plt.savefig(os.path.join(output_dir, "train_results.png"))
    plt.close()

    # compute pearson correlation between pred and true
    train_corr = np.corrcoef(test_df["label"], test_df["pred"])[0, 1]
    logger.info(f"Train correlation: {train_corr}")
    with open(os.path.join(output_dir, "train_corr.txt"), "w") as f:
        f.write(str(train_corr))

    # save SHAP summary plot
    shap.summary_plot(shap_values, features, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"))
    plt.close()
    plt.figure()

    # get the gene name of top features
    top_feature_names = get_top_k_features(shap_values, feature_names, k=20)
    top_feature_genes = [i.split("_")[-1] for i in top_feature_names]

    # get connectivity of top features
    network_interactions, _ = get_stringdb_network_interactions(top_feature_genes)
    network_interactions.to_csv(
        os.path.join(output_dir, "network_interactions.csv"), index=False
    )

    # save trainers
    with open(os.path.join(output_dir, "trainers.pkl"), "wb") as f:
        pickle.dump(trainers, f)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--response_path", type=str, required=True)
    argparser.add_argument("--feature_path", type=str, required=True)
    argparser.add_argument("--output_dir", type=str, required=True)
    args = argparser.parse_args()
    run(args.response_path, args.feature_path, args.output_dir)
