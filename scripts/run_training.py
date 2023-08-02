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

from refract.datasets import PrismDataset
from refract.metrics import (
    get_merged_shap_values_and_features,
    get_stringdb_network_interactions,
    get_test_predictions,
    get_top_k_features,
)
from refract.utils import get_top_features
from refract.trainers import XGBoostRankingTrainer

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

SLATE_LENGTH = 10
NUM_TREES = 200 
NUM_EPOCHS = 100
CV_FOLDS = 5

def dataset_to_group_df(ds):
    features = []
    labels = []
    groups = []
    for i in range(NUM_EPOCHS):
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

def run(response_path, feature_path, output_dir):
    # create output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # update logger to write to file
    fh = logging.FileHandler(os.path.join(output_dir, "train.log"))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # load data
    logger.info("Loading feature data...")
    with open(feature_path, "rb") as f:
        feature_df = pickle.load(f)
    feature_df = feature_df.rename_axis("ccle_name")
    feature_df = feature_df.fillna(-1)

    logger.info("Loading response data...")
    response_df = pd.read_csv(response_path)

    # get top 1% of features
    top_features = get_top_features(response_df, feature_df, "LFC.cb", 0.01)

    # START CV TRAIN
    splitter = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    trainers = []
    for i, (train_val_index, test_index) in enumerate(splitter.split(response_df)):
        logger.info(f"Training fold {i}")
        response_train_val = response_df.iloc[train_val_index, :].copy()
        response_test = response_df.iloc[test_index, :].copy()
        train_val_splitter = KFold(n_splits=CV_FOLDS - 1, shuffle=True, random_state=42)
        train_index, val_index = next(train_val_splitter.split(response_train_val))
        response_train = response_train_val.iloc[train_index, :].copy()
        response_val = response_train_val.iloc[val_index, :].copy()

        # load datasets
        ds_train = PrismDataset(
            response_train.copy(),
            feature_df,
            slate_length=SLATE_LENGTH,
            feature_cols=top_features
        )
        ds_val = PrismDataset(
            response_val.copy(),
            feature_df,
            slate_length=SLATE_LENGTH,
            feature_cols=top_features
        )
        ds_test = PrismDataset(
            response_test.copy(),
            feature_df,
            slate_length=SLATE_LENGTH,
            feature_cols=top_features
        )

        # train one fold
        trainer = XGBoostRankingTrainer(
            train_ds=ds_train,
            val_ds=ds_val,
            test_ds=ds_test,
            num_trees=NUM_TREES,
            num_epochs=NUM_EPOCHS,
        )
        trainer.train()
        trainers.append(trainer)
    ### END CV TRAIN

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
