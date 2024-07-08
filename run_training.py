import argparse
import logging
import os
import pickle
import sys
import json
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

from refract.trainers import AutoMLTrainer
from refract.utils import save_output, get_fold_assignment

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

CV_FOLDS = 10

def run(
    drug_name,
    response_dir,
    feature_path,
    output_dir,
    neighborhood_json,
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
    # drop low variance features
    feature_df = feature_df.loc[:, feature_df.var() > 0]

    # read the neighborhood json
    logger.info("Loading neighborhood data...")
    with open(neighborhood_json, "r") as f:
        neighborhood_dict = json.load(f)
        # get similar drugs from the neighborhood
    similar_drugs = neighborhood_dict[drug_name]

    logger.info("Loading response data...")
    response_files = glob.glob(os.path.join(response_dir, "*.csv"))
    response_data = {}
    for response_file in response_files:
        response_name = os.path.basename(response_file).replace(".csv", "")
        response_data[response_name] = pd.read_csv(response_file)
    # for every one, select LFC.CB, pert_name, ccle_name
    response_data = {k: v.loc[:, ["LFC.cb", "pert_name", "ccle_name"]] for k, v in response_data.items()}
    # concatenate all responses
    response_data = pd.concat(response_data.values(), axis=0)
    # drop duplicates on pert_name, ccle_name
    response_data = response_data.drop_duplicates(subset=["pert_name", "ccle_name"])
    # pivot so ccle_name is the columns and pert_name is the index
    response_data = response_data.pivot(index="pert_name", columns="ccle_name", values="LFC.cb")    # get responses for all these
    cluster_responses = response_data.loc[response_data.index.isin(similar_drugs), :]
    # set columns as str
    cluster_responses.columns = cluster_responses.columns.astype(str)
    # drop column named nan
    cluster_responses = cluster_responses.drop("nan", axis=1)
    # transpose cluster_responses
    cluster_responses = cluster_responses.T
    # fill NaN with 0
    cluster_responses = cluster_responses.fillna(0)
    # melt cluster responses
    cluster_responses = cluster_responses.reset_index().melt(id_vars="ccle_name", var_name="pert_name", value_name="LFC.cb")
    # set ccle_name as index
    cluster_responses = cluster_responses.set_index("ccle_name")

    logger.info("Preparing for training...")
    fold_assignment = get_fold_assignment(cluster_responses, drug_name)
    cluster_responses = cluster_responses.loc[cluster_responses.index.isin(fold_assignment.keys()), :]
    # merge all
    df_all = cluster_responses.merge(feature_df, left_index=True, right_index=True, how='inner')
    feature_cols = feature_df.columns
    label_cols = cluster_responses.columns
    df_all["fold"] = df_all.index.map(fold_assignment)

    # START CV TRAIN
    logger.info("Training...")
    X_all = df_all.loc[:, feature_cols]
    y_all = df_all.loc[:, label_cols]
    groups = df_all["fold"]
    outer_cv = GroupKFold(n_splits=10)
    trainers = []
    for i, (train_index, test_index) in enumerate(outer_cv.split(X_all, y_all, groups)):
        logger.info(f"Training fold {i}")
        X_train, X_test = X_all.iloc[train_index], X_all.iloc[test_index]
        y_train, y_test = y_all.iloc[train_index], y_all.iloc[test_index]        
        # train one fold
        trainer = AutoMLTrainer(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_cols=feature_cols,
            drug_name=drug_name,
            fold_assignment=fold_assignment
        )
        trainer.select_features()
        trainer.train()
        trainers.append(trainer)
        ### END CV TRAIN

    # save output
    logger.info("Saving output...")
    save_output(trainers, output_dir)
    logger.info("done")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--drug_name", type=str, required=True)
    argparser.add_argument("--response_dir", type=str, required=True)
    argparser.add_argument("--feature_path", type=str, required=True)
    argparser.add_argument("--output_dir", type=str, required=True)
    argparser.add_argument("--neighborhood_json", type=str, required=True)
    args = argparser.parse_args()
    run(args.drug_name, args.response_dir, args.feature_path, args.output_dir, args.neighborhood_json)
