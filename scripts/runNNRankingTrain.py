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
from shap import DeepExplainer
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import logging

from refract.datasets import PrismDataset
from refract.losses import lambdaLoss
from refract.metrics import get_stringdb_network_interactions
from refract.models import FeedForwardNet
from refract.ranking_trainers import NNRankerTrainer

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

NUM_FEATURES = 100
SLATE_LENGTH = 10
NUM_EPOCHS = 50
BATCH_SIZE = 512


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

    # START CV LOOP
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    all_preds = []
    all_trues = []
    all_ccle_names = []
    for i, (train_index, test_index) in enumerate(splitter.split(response_df)):
        logger.info(f"Training fold {i}")
        response_train = response_df.iloc[train_index, :].copy()
        response_test = response_df.iloc[test_index, :].copy()

        # Train model
        ds_train = PrismDataset(
            response_train,
            feature_df,
            feature_importance_df,
            top_k_features=100,
            slate_length=10,
        )
        model = FeedForwardNet(NUM_FEATURES)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer = NNRankerTrainer(model, optimizer, num_workers=3)
        trainer.train(ds_train, BATCH_SIZE, NUM_EPOCHS)

        # Eval model
        ds_test = PrismDataset(
            response_test,
            feature_df,
            feature_importance_df,
            top_k_features=100,
            slate_length=10,
            feature_transformer=ds_train.feature_transformer,
        )
        train_ccle_names, train_preds, train_trues, train_corr = trainer.eval(ds_train)
        test_ccle_names, test_preds, test_trues, test_corr = trainer.eval(ds_test)

        # compute SHAP values
        X = []
        y = []
        for ex in ds_test:
            _, feat, label = ex
            X.append(feat[0, :])
            y.append(label[0])
        X = torch.stack(X)
        X = X.to(trainer.device)
        exp = DeepExplainer(model, X)
        shap_values = exp.shap_values(X)

        cv_results.append(
            {
                "train_ccle_names": train_ccle_names,
                "train_preds": train_preds,
                "train_trues": train_trues,
                "train_corr": train_corr,
                "train_index": train_index,
                "train_X": ds_train.joined_df.values[:, :-1],
                "feature_names": ds_train.joined_df.columns[:-1],
                "test_ccle_names": test_ccle_names,
                "test_preds": test_preds,
                "test_trues": test_trues,
                "test_corr": test_corr,
                "test_index": test_index,
                "test_X": ds_test.joined_df.values[:, :-1],
                "model": model,
                "shap_values": shap_values,
                "trainer": trainer,
            }
        )
        all_ccle_names.extend(test_ccle_names)
        all_preds.extend(test_preds)
        all_trues.extend(test_trues)
    # END CV LOOP

    # save predictions
    logger.info("Saving predictions...")
    train_results_df = pd.DataFrame(
        {"ccle_name": all_ccle_names, "preds": all_preds, "trues": all_trues}
    )
    train_results_df.to_csv(os.path.join(output_dir, "train_results.csv"), index=False)
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
    for d in cv_results:
        model = d["model"]
        trainer = d["trainer"]
        del d["model"]
        del d["trainer"]
        d["model"] = model.state_dict()
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
