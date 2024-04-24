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
from sklearn.model_selection import KFold, StratifiedKFold

from refract.datasets import PrismDataset
from refract.trainers import AutoMLTrainer, BaselineTrainer
from refract.utils import save_output

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

CV_FOLDS = 10

def run(
    response_path,
    feature_path,
    output_dir,
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
    feature_df.set_index("ccle_name", inplace=True)
    feature_df.fillna(-1, inplace=True)

    logger.info("Loading response data...")
    response_df = pd.read_csv(response_path)

    # only keep cell lines we have features for
    available_ccle_names = set(feature_df.index)
    response_df = response_df[response_df["ccle_name"].isin(available_ccle_names)]

    # drop culture column
    response_df = response_df.drop(columns=["culture"])
    # drop duplicates by ccle_name, keep first
    response_df = response_df.drop_duplicates(subset=["ccle_name"], keep="first")

    # START CV TRAIN
    skf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    trainers = []
    for i, (train_index, test_index) in enumerate(skf.split(response_df)):
        logger.info(f"Training fold {i}")
        response_train = response_df.iloc[train_index, :].reset_index(drop=True).copy()
        response_test = response_df.iloc[test_index, :].reset_index(drop=True).copy()

        # train one fold
        trainer = AutoMLTrainer(
            response_train=response_train,
            response_test=response_test,
            feature_df=feature_df,
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
    args = argparser.parse_args()
    run(args.response_path, args.feature_path, args.output_dir)
