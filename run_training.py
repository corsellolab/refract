import argparse
import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import KFold

from refract.datasets import PrismDataset
from refract.trainers import XGBoostRankingTrainer
from refract.utils import get_top_features, save_output

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

SLATE_LENGTH = 10
NUM_EPOCHS = 100
CV_FOLDS = 5


def run(
    response_path,
    feature_path,
    output_dir,
    feature_fraction,
    slate_length=SLATE_LENGTH,
    num_epochs=NUM_EPOCHS,
    cv_folds=CV_FOLDS,
):
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

    # START CV TRAIN
    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    trainers = []
    for i, (train_val_index, test_index) in enumerate(splitter.split(response_df)):
        logger.info(f"Training fold {i}")
        response_train_val = (
            response_df.iloc[train_val_index, :].reset_index(drop=True).copy()
        )
        train_val_splitter = KFold(n_splits=cv_folds - 1, shuffle=True, random_state=42)
        train_index, val_index = next(train_val_splitter.split(response_train_val))
        response_train = (
            response_train_val.iloc[train_index, :].reset_index(drop=True).copy()
        )
        response_val = (
            response_train_val.iloc[val_index, :].reset_index(drop=True).copy()
        )
        response_test = response_df.iloc[test_index, :].reset_index(drop=True).copy()

        # feature selection
        top_features = get_top_features(
            response_train, feature_df, "LFC.cb", feature_fraction
        )

        # load datasets
        ds_train = PrismDataset(
            response_train,
            feature_df,
            slate_length=slate_length,
            feature_cols=top_features,
        )
        ds_val = PrismDataset(
            response_val,
            feature_df,
            slate_length=slate_length,
            feature_cols=top_features,
        )
        ds_test = PrismDataset(
            response_test,
            feature_df,
            slate_length=slate_length,
            feature_cols=top_features,
        )

        # train one fold
        trainer = XGBoostRankingTrainer(
            train_ds=ds_train,
            val_ds=ds_val,
            test_ds=ds_test,
            num_epochs=num_epochs,
        )
        trainer.train()
        trainers.append(trainer)
        ### END CV TRAIN

    # save output
    logger.info("Saving output...")
    save_output(trainers, output_dir)
    logger.info("done")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--response_path", type=str, required=True)
    argparser.add_argument("--feature_path", type=str, required=True)
    argparser.add_argument("--output_dir", type=str, required=True)
    argparser.add_argument("--feature_fraction", type=float, default=0.01)
    args = argparser.parse_args()
    run(args.response_path, args.feature_path, args.output_dir, args.feature_fraction)
