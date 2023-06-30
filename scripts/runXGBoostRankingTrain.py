# Script to run training with a ranking neural network model
# add parent dir to path
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

from refract.datasets import PrismDataset
from refract.losses import lambdaLoss
from refract.metrics import get_stringdb_network_interactions
from refract.models import FeedForwardNet
from refract.ranking_trainers import NNRankerTrainer

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

NUM_FEATURES = 100
SLATE_LENGTH = 10
NUM_EPOCHS = 100
NUM_TREES = 200


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


def run(response_path, feature_path, feature_importance_path, output_dir):
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

    logger.info("Loading response data...")
    response_df = pd.read_csv(response_path)
    feature_importance_df = pd.read_csv(feature_importance_path)
    # filter response data to only include single dose
    response_df = response_df[response_df["pert_idose"] == 2.5]

    # START CV LOOP
    splitter = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_results = []
    all_preds = []
    all_trues = []
    all_ccle_names = []
    for i, (train_val_index, test_index) in enumerate(splitter.split(response_df)):
        logger.info(f"Training fold {i}")
        # also create a validation set out of the training set
        response_train_val = response_df.iloc[train_val_index, :].copy()
        response_test = response_df.iloc[test_index, :].copy()
        train_val_splitter = KFold(n_splits=9, shuffle=True, random_state=42)
        train_index, val_index = next(train_val_splitter.split(response_train_val))
        response_train = response_train_val.iloc[train_index, :].copy()
        response_val = response_train_val.iloc[val_index, :].copy()

        # load datasets
        ds_train = PrismDataset(
            response_train,
            feature_df,
            feature_importance_df,
            top_k_features=NUM_FEATURES,
            slate_length=SLATE_LENGTH,
        )
        ds_val = PrismDataset(
            response_val,
            feature_df,
            feature_importance_df,
            top_k_features=NUM_FEATURES,
            slate_length=SLATE_LENGTH,
            feature_transformer=ds_train.feature_transformer,
        )
        ds_test = PrismDataset(
            response_test,
            feature_df,
            feature_importance_df,
            top_k_features=NUM_FEATURES,
            slate_length=SLATE_LENGTH,
            feature_transformer=ds_train.feature_transformer,
        )

        # convert datasets to numpy arrays
        group_train_features, group_train_labels, train_groups = dataset_to_group_df(
            ds_train
        )
        group_val_features, group_val_labels, val_groups = dataset_to_group_df(ds_val)
        train_features, train_labels, train_ccle_names = dataset_to_individual_df(
            ds_train
        )
        test_features, test_labels, test_ccle_names = dataset_to_individual_df(ds_test)

        group_train_dmatrix = xgb.DMatrix(
            group_train_features, label=group_train_labels, group=train_groups
        )
        group_val_dmatrix = xgb.DMatrix(
            group_val_features, label=group_val_labels, group=val_groups
        )
        train_dmatrix = xgb.DMatrix(train_features, label=train_labels)
        test_dmatrix = xgb.DMatrix(test_features, label=test_labels)

        params = {
            "objective": "rank:pairwise",
            "colsample_bytree": 0.5,
            "colsample_bylevel": 0.5,
            "colsample_bynode": 0.5,
            "seed": 42,
        }

        # Train for a minimum of 50 rounds
        xgb_model = xgb.train(params, group_train_dmatrix, num_boost_round=100)
        # Train until validation score doesn't improve for 10 rounds
        xgb_model = xgb.train(
            params,
            group_train_dmatrix,
            num_boost_round=NUM_TREES,
            evals=[(group_val_dmatrix, "eval")],
            early_stopping_rounds=10,
            verbose_eval=10,
            xgb_model=xgb_model,
        )

        train_preds = xgb_model.predict(train_dmatrix)
        test_preds = xgb_model.predict(test_dmatrix)

        train_corr = np.corrcoef(train_preds, train_labels)[0, 1]
        test_corr = np.corrcoef(test_preds, test_labels)[0, 1]

        # compute SHAP values
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(test_dmatrix)
        print("Train corr ", train_corr)
        print("Test corr ", test_corr)

        cv_results.append(
            {
                "train_ccle_names": train_ccle_names,
                "train_preds": train_preds,
                "train_trues": train_labels,
                "train_corr": train_corr,
                "train_index": train_index,
                "train_X": ds_train.joined_df.values[:, :-1],
                "feature_names": ds_train.joined_df.columns[:-1],
                "test_ccle_names": test_ccle_names,
                "test_preds": test_preds,
                "test_trues": test_labels,
                "test_corr": test_corr,
                "test_index": test_index,
                "test_X": ds_test.joined_df.values[:, :-1],
                "model": xgb_model,
                "shap_values": shap_values,
            }
        )
        all_ccle_names.extend(test_ccle_names)
        all_preds.extend(test_preds)
        all_trues.extend(test_labels)
    # END CV LOOP

    # save predictions
    logger.info("Saving predictions...")
    train_results_df = pd.DataFrame(
        {"ccle_name": all_ccle_names, "preds": all_preds, "trues": all_trues}
    )
    train_results_df.to_csv(os.path.join(output_dir, "train_results.csv"), index=False)
    # plot a scatterplot of preds vs trues
    # dont show, save fig and close
    plt.figure(figsize=(5, 5))
    plt.scatter(all_trues, all_preds, alpha=0.5)
    plt.xlabel("-1 * (Scaled LFC)")
    plt.ylabel("Ranking Score")
    plt.savefig(os.path.join(output_dir, "train_results.png"))
    plt.close()
    # compute pearson correlation between pred and true
    train_corr = np.corrcoef(all_preds, all_trues)[0, 1]
    logger.info(f"Train correlation: {train_corr}")
    # save correlation to file
    with open(os.path.join(output_dir, "train_corr.txt"), "w") as f:
        f.write(str(train_corr))

    # compute SHAP values
    logger.info("Computing SHAP values...")
    all_shap = []
    all_features = []
    feature_names = []
    for cv_result in cv_results:
        train_X = cv_result["test_X"]
        all_shap.append(cv_result["shap_values"])
        all_features.append(train_X)
        feature_names = cv_result["feature_names"]
    shap_values = np.concatenate(all_shap, axis=0)
    all_features = np.concatenate(all_features, axis=0)
    shap_values_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_values_df.to_csv(os.path.join(output_dir, "shap_values.csv"), index=False)
    # create SHAP summary plot
    logger.info("Creating SHAP summary plot...")
    shap.summary_plot(
        shap_values, all_features, feature_names=feature_names, show=False
    )
    plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"))
    plt.close()
    plt.figure()

    # get the top features
    logger.info("Getting top features...")
    top_features = np.argsort(np.abs(shap_values).mean(0))[-20:]
    # save top features
    top_features_df = pd.DataFrame(
        {
            "feature": feature_names[top_features],
            "shap_value": np.abs(shap_values).mean(0)[top_features],
        }
    )
    top_features_df.to_csv(os.path.join(output_dir, "top_features.csv"), index=False)
    # get the connectivity of top features
    top_features_genes = feature_names[top_features]
    top_features_genes = [x.split("_")[-1] for x in top_features_genes]
    top_features_genes = list(set(top_features_genes))
    network_interactions, _ = get_stringdb_network_interactions(top_features_genes)
    network_interactions.to_csv(
        os.path.join(output_dir, "network_interactions.csv"), index=False
    )

    # save CV result
    logger.info("Saving CV results...")
    with open(os.path.join(output_dir, "cv_results.pkl"), "wb") as f:
        pickle.dump(cv_results, f)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--response_path",
        type=str,
        help="Path to response data",
    )
    argparser.add_argument(
        "--feature_path",
        type=str,
        help="Path to feature data in pickle format",
    )
    argparser.add_argument(
        "--feature_importance_path",
        type=str,
        help="Path to feature importance data",
    )
    argparser.add_argument(
        "--output_dir",
        type=str,
        help="Path to output directory",
    )
    args = argparser.parse_args()
    run(
        args.response_path,
        args.feature_path,
        args.feature_importance_path,
        args.output_dir,
    )
