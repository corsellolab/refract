# Run training script for a single compound
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import logging

from refract.datasets import FeatureSet, PrismDataset, ResponseSet
from refract.metrics import (
    get_merged_shap_values_and_features,
    get_stringdb_network_interactions,
    get_test_predictions,
    get_top_k_features,
)
from refract.trainers import XGBoostRankingTrainer

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

NUM_EPOCHS = 1
CV_FOLDS = 10
SLATE_LENGTH = 10
NUM_TREES = 50


def run(response_path, feature_path, output_dir):
    # create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # update logger to write to file
    fh = logging.FileHandler(os.path.join(output_dir, "train.log"))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # load response data
    logger.info("Loading response data...")
    response_set = ResponseSet(response_path)
    response_set.load_response_table()
    response_df = response_set.get_response_df(dose=2.5)

    # load feature data
    logger.info("Loading feature data...")
    feature_set = FeatureSet(feature_path)
    feature_set.load_concatenated_feature_tables()
    feature_df = feature_set.get_feature_df("all")

    trainers = []
    outer_splitter = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    logger.info("Starting CV training...")
    for idx, (train_val_index, test_index) in enumerate(
        outer_splitter.split(response_df)
    ):
        logger.info("Training fold: {}".format(idx))
        train_val = response_df.iloc[train_val_index, :].copy()
        train_index, val_index = next(
            KFold(n_splits=CV_FOLDS - 1, shuffle=True, random_state=42).split(train_val)
        )

        train = train_val.iloc[train_index].copy()
        val = train_val.iloc[val_index].copy()
        test = response_df.iloc[test_index].copy()

        # check no overlap
        assert len(set(train.index).intersection(set(val.index))) == 0
        assert len(set(train.index).intersection(set(test.index))) == 0
        assert len(set(val.index).intersection(set(test.index))) == 0

        ds_train = PrismDataset(train, feature_df, SLATE_LENGTH)
        ds_val = PrismDataset(val, feature_df, SLATE_LENGTH)
        ds_test = PrismDataset(test, feature_df, SLATE_LENGTH)

        trainer = XGBoostRankingTrainer(
            ds_train, ds_val, ds_test, num_trees=NUM_TREES, num_epochs=NUM_EPOCHS
        )

        trainer.train()
        trainer.compute_stats()
        trainer.log_stats()

        trainers.append(trainer)

    # compute SHAP values and predictions across full dataset
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
    top_feature_genes = [i.split("_")[-1] for i in top_feature_names]

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
    logger.info("done")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--response_path", type=str, required=True)
    argparser.add_argument("--feature_path", type=str, required=True)
    argparser.add_argument("--output_dir", type=str, required=True)
    args = argparser.parse_args()
    run(args.response_path, args.feature_path, args.output_dir)
