# Run training script for a single compound
import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.model_selection import KFold, train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import logging

import xgboost as xgb

from refract.datasets import FeatureSet, PrismDataset, ResponseSet
from refract.losses import lambdaLoss
from refract.metrics import get_stringdb_network_interactions
from refract.models import FeedForwardNet
from refract.ranking_trainers import NNRankerTrainer

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

SLATE_LENGTH = 10
NUM_EPOCHS = 1
NUM_TREES = 100


def dataset_to_group_df(ds):
    features = []
    labels = []
    groups = []
    for i in range(NUM_EPOCHS):
        for ex in ds:
            ccle_name, feat, label = ex
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


def train_one_cv_fold(ds_train, ds_val):
    """Train a single fold of cross-validation"""
    # get training slates
    group_train_features, group_train_labels, train_groups = dataset_to_group_df(
        ds_train
    )
    group_val_features, group_val_labels, val_groups = dataset_to_group_df(ds_val)

    # get dmatrix objects
    group_train_dmatrix = xgb.DMatrix(
        group_train_features, label=group_train_labels, group=train_groups
    )
    group_val_dmatrix = xgb.DMatrix(
        group_val_features, label=group_val_labels, group=val_groups
    )

    # train model
    params = {
        "objective": "rank:pairwise",
        "colsample_bytree": 0.5,
        "colsample_bylevel": 0.5,
        "colsample_bynode": 0.5,
        "seed": 42,
    }

    # train for a minimum of 100 rounds
    xgb_model = xgb.train(params, group_train_dmatrix, num_boost_round=100)
    # train until validation score doesnt improve for 20 rounds
    xgb_model = xgb.train(
        params,
        group_train_dmatrix,
        num_boost_round=NUM_TREES,
        evals=[(group_val_dmatrix, "validation")],
        early_stopping_rounds=20,
        verbose_eval=10,
        xgb_model=xgb_model,
    )

    return xgb_model


def get_stats_one_cv_fold(ds_train, ds_val, ds_test, xgb_model):
    """Get stats for a single fold of the xgb_model"""
    train_features, train_labels, train_ccle_names = dataset_to_individual_df(ds_train)
    val_features, val_labels, val_ccle_names = dataset_to_individual_df(ds_val)
    test_features, test_labels, test_ccle_names = dataset_to_individual_df(ds_test)

    # create dmatrix objects
    train_dmatrix = xgb.DMatrix(train_features, label=train_labels)
    val_dmatrix = xgb.DMatrix(val_features, label=val_labels)
    test_dmatrix = xgb.DMatrix(test_features, label=test_labels)

    # get predictions
    train_preds = xgb_model.predict(train_dmatrix)
    val_preds = xgb_model.predict(val_dmatrix)
    test_preds = xgb_model.predict(test_dmatrix)

    # get correlations
    train_corr = np.corrcoef(train_preds, train_labels)[0, 1]
    val_corr = np.corrcoef(val_preds, val_labels)[0, 1]
    test_corr = np.corrcoef(test_preds, test_labels)[0, 1]

    # return stats
    return {
        "train_ccle_names": train_ccle_names,
        "train_preds": train_preds,
        "train_labels": train_labels,
        "train_corr": train_corr,
        "val_ccle_names": val_ccle_names,
        "val_preds": val_preds,
        "val_labels": val_labels,
        "val_corr": val_corr,
        "test_ccle_names": test_ccle_names,
        "test_preds": test_preds,
        "test_labels": test_labels,
        "test_corr": test_corr,
        "test_df": ds_test.joined_df,
        "model": xgb_model,
    }


def run(response_path, feature_path, output_dir):
    # create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # update logger to write to file
    fh = logging.FileHandler(os.path.join(output_dir, "train.log"))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # load data
    logger.info("Loading response data...")
    response_set = ResponseSet(response_path)
    response_set.load_response_table()
    response_df = response_set.get_response_df(dose=2.5)

    logger.info("Loading feature data...")
    feature_set = FeatureSet(feature_path)
    feature_set.load_concatenated_feature_tables()
    feature_df = feature_set.get_feature_df("all")

    ### START CV TRAIN ###
    splitter = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_stats = []
    for fold, (train_val_index, test_index) in enumerate(splitter.split(response_df)):
        logger.info(f"Training fold: {fold}")
        # get train/val/test indexes
        response_train_val = response_df.iloc[train_val_index, :].copy()
        train_val_splitter = KFold(n_splits=9, shuffle=True, random_state=42)
        train_index, val_index = next(train_val_splitter.split(response_train_val))

        # get response dataframes
        response_train = response_train_val.iloc[train_index, :].copy()
        response_val = response_train_val.iloc[val_index, :].copy()
        response_test = response_df.iloc[test_index, :].copy()

        # create datasets
        ds_train = PrismDataset(response_train, feature_df, SLATE_LENGTH)
        ds_val = PrismDataset(
            response_val,
            feature_df,
            SLATE_LENGTH,
            label_transformer=ds_train.label_transformer,
        )
        ds_test = PrismDataset(
            response_test,
            feature_df,
            SLATE_LENGTH,
            label_transformer=ds_train.label_transformer,
        )

        # train model
        xgb_model = train_one_cv_fold(ds_train, ds_val)

        # get predictions
        stats = get_stats_one_cv_fold(ds_train, ds_val, ds_test, xgb_model)
        fold_stats.append(stats)
    ### END CV TRAIN ###

    # Save predictions
    ccle_names = []
    preds = []
    labels = []
    for fold in fold_stats:
        ccle_names.extend(fold["test_ccle_names"])
        preds.extend(fold["test_preds"])
        labels.extend(fold["test_labels"])
    pred_df = pd.DataFrame(
        {
            "ccle_name": ccle_names,
            "pred": preds,
            "label": labels,
        }
    )
    pred_df.to_csv(os.path.join(output_dir, "train_results.csv"), index=False)

    # plot a scatterplot of pred vs label
    plt.figure(figsize=(5, 5))
    plt.scatter(pred_df["label"], pred_df["pred"], alpha=0.5)
    plt.xlabel("-1 * (Scaled LFC)")
    plt.ylabel("Ranking Score")
    plt.savefig(os.path.join(output_dir, "train_results.png"))
    plt.close()

    # compute pearson correlation between pred and true
    train_corr = np.corrcoef(pred_df["label"], pred_df["pred"])[0, 1]
    logger.info(f"Train correlation: {train_corr}")
    with open(os.path.join(output_dir, "train_corr.txt"), "w") as f:
        f.write(str(train_corr))

    # Compute SHAP values
    logger.info("Computing SHAP values...")
    all_x_test = []
    all_shap_values = []
    for fold in fold_stats:
        x_test = fold["test_df"].iloc[:, :-1].values
        xgb_model = fold["model"]
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(x_test)
        all_x_test.append(x_test)
        all_shap_values.append(shap_values)
    all_x_test = np.concatenate(all_x_test, axis=0)
    all_shap_values = np.concatenate(all_shap_values, axis=0)
    logger.info("Saving SHAP summary plot...")
    shap.summary_plot(
        all_shap_values, all_x_test, feature_names=feature_df.columns[:-1], show=False
    )
    plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"))
    plt.close()
    plt.figure()

    # get the top features
    logger.info("Getting top features...")
    top_features = np.argsort(np.abs(all_shap_values).mean(0))[-20:]
    top_features_df = pd.DataFrame(
        {
            "feature": feature_df.columns[top_features],
            "shap_value": np.abs(all_shap_values).mean(0)[top_features],
        }
    )
    top_features_df.to_csv(os.path.join(output_dir, "top_features.csv"), index=False)

    # get the connectivity of the top features
    top_feature_genes = top_features_df["feature"].values
    top_feature_genes = [x.split("_")[0] for x in top_feature_genes]
    top_feature_genes = list(set(top_feature_genes))
    network_interactions, _ = get_stringdb_network_interactions(top_feature_genes)
    network_interactions.to_csv(
        os.path.join(output_dir, "network_interactions.csv"), index=False
    )

    # save CV result
    logger.info("Saving CV Results...")
    with open(os.path.join(output_dir, "cv_results.pkl"), "wb") as f:
        pickle.dump(fold_stats, f)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--response_path", type=str, required=True)
    argparser.add_argument("--feature_path", type=str, required=True)
    argparser.add_argument("--output_dir", type=str, required=True)
    args = argparser.parse_args()
    run(args.response_path, args.feature_path, args.output_dir)
