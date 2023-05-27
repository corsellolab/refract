# Script to run training with a ranking neural network model
# add parent dir to path
import json
import os
import pickle
import sys

import pandas as pd
import torch
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import logging

from refract.datasets import PrismDataset
from refract.losses import lambdaLoss
from refract.models import FeedForwardNet
from refract.ranking_trainers import NNRankerTrainer

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

NUM_FEATURES = 100
SLATE_LENGTH = 10
NUM_EPOCHS = 200
BATCH_SIZE = 512


def run(response_path, feature_path, feature_importance_path, output_dir):
    # load data
    logger.info("Loading feature data...")
    with open(feature_path, "rb") as f:
        feature_df = pickle.load(f)
    feature_df = feature_df.rename_axis("ccle_name")

    logger.info("Loading response data...")
    response_df = pd.read_csv(response_path)
    feature_importance_df = pd.read_csv(feature_importance_path)

    # Simple CV training
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    all_preds = []
    all_trues = []
    for i, (train_index, test_index) in enumerate(splitter.split(response_df)):
        logger.info(f"Training fold {i}")
        response_train = response_df.iloc[train_index, :].copy()
        response_test = response_df.iloc[test_index, :].copy()
        ds_train = PrismDataset(
            response_train,
            feature_df,
            feature_importance_df,
            top_k_features=100,
            slate_length=10,
        )
        ds_test = PrismDataset(
            response_test,
            feature_df,
            feature_importance_df,
            top_k_features=100,
            slate_length=10,
        )
        model = FeedForwardNet(NUM_FEATURES)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer = NNRankerTrainer(model, optimizer, num_workers=8)
        trainer.train(ds_train, BATCH_SIZE, NUM_EPOCHS)
        train_preds, train_trues, train_corr = trainer.eval(ds_train)
        test_preds, test_trues, test_corr = trainer.eval(ds_test)
        cv_results.append(
            {
                "train_preds": train_preds,
                "train_trues": train_trues,
                "train_corr": train_corr,
                "test_preds": test_preds,
                "test_trues": test_trues,
                "test_corr": test_corr,
            }
        )
        all_preds.extend(test_preds)
        all_trues.extend(test_trues)
