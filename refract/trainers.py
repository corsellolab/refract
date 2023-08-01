# Trainer for the XGboost ranking model
import argparse
import json
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

from refract.datasets import FeatureSet, PrismDataset, ResponseSet
from refract.metrics import get_stringdb_network_interactions
from refract.utils import torch_dataset_to_numpy_array

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


class XGBoostRankingTrainer:
    """Trains a XGBoost ranking model."""

    def __init__(
        self, train_ds, val_ds, test_ds, num_trees, num_epochs=1, model_config={}
    ):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.num_epochs = num_epochs
        self.model_config = model_config
        self.num_trees = num_trees

        # training stats
        self.train_corr = None
        self.train_ccle_names = None
        self.train_preds = None
        self.train_labels = None
        self.val_corr = None
        self.val_ccle_names = None
        self.val_preds = None
        self.val_labels = None
        self.test_corr = None
        self.test_ccle_names = None
        self.test_preds = None
        self.test_labels = None

        # only save test labels and preds to save space
        # needed for SHAP
        self.test_features = None
        self.test_feature_names = None

    def get_model_config(self, config={}):
        
        """
        params = {
            "objective": "rank:pairwise",
            "eval_metric": "ndcg",
            "colsample_bytree": 0.5,
            "colsample_bylevel": 0.5,
            "colsample_bynode": 0.5,
            "seed": 42,
            "gamma": 1.0,
            "reg_lambda": 1.0,
            "reg_alpha": 1.0,
            "learning_rate": 0.05,
            "n_jobs": 8,
        }
        """
        params = {
            "objective": "rank:pairwise",
            "eval_metric": "ndcg",
            "colsample_bytree": 0.5,
            "colsample_bylevel": 0.5,
            "colsample_bynode": 0.5,
            "seed": 42,
            "n_jobs": 8,
        }
        params.update(config)
        return params

    def train(self):
        # get training datasets
        (
            group_train_features,
            group_train_labels,
            train_groups,
            _,
        ) = torch_dataset_to_numpy_array(self.train_ds, num_epochs=self.num_epochs)
        (
            group_val_features,
            group_val_labels,
            val_groups,
            _,
        ) = torch_dataset_to_numpy_array(self.val_ds)

        train_dmatrix = xgb.DMatrix(
            group_train_features, label=group_train_labels, group=train_groups
        )
        del group_train_features
        del group_train_labels
        del train_groups

        val_dmatrix = xgb.DMatrix(
            group_val_features, label=group_val_labels, group=val_groups
        )
        del group_val_features
        del group_val_labels
        del val_groups

        params = self.get_model_config(self.model_config)

        watchlist = [(train_dmatrix, "train"), (val_dmatrix, "val")]

        # initial training
        xgb_model = xgb.train(
            params,
            train_dmatrix,
            num_boost_round=self.num_trees,
            evals=watchlist,
            verbose_eval=True,
        )
        self.model = xgb_model

    def predict(self, features_array):
        dmatrix = xgb.DMatrix(features_array)
        preds = self.model.predict(dmatrix)
        return preds

    def compute_stats(self):
        # get feature numpy arrays
        train_features = self.train_ds.joined_df.iloc[:, :-1].values
        train_ccle_names = self.train_ds.joined_df.index.values
        val_features = self.val_ds.joined_df.iloc[:, :-1].values
        val_ccle_names = self.val_ds.joined_df.index.values
        test_features = self.test_ds.joined_df.iloc[:, :-1].values
        test_ccle_names = self.test_ds.joined_df.index.values

        # get unscaled labels
        train_labels = self.train_ds.unscaled_labels
        val_labels = self.val_ds.unscaled_labels
        test_labels = self.test_ds.unscaled_labels

        # save test features to trainer
        self.test_features = test_features
        self.test_feature_names = self.test_ds.cols

        # save SHAP values
        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer.shap_values(test_features)

        # get preds
        train_preds = self.predict(train_features) * -1
        val_preds = self.predict(val_features) * -1
        test_preds = self.predict(test_features) * -1

        # get correlations
        train_corr = np.corrcoef(train_preds, train_labels)[0, 1]
        val_corr = np.corrcoef(val_preds, val_labels)[0, 1]
        test_corr = np.corrcoef(test_preds, test_labels)[0, 1]

        # save stats to trainer
        self.train_corr = train_corr
        self.train_ccle_names = train_ccle_names
        self.train_preds = train_preds
        self.train_labels = train_labels
        self.val_corr = val_corr
        self.val_ccle_names = val_ccle_names
        self.val_preds = val_preds
        self.val_labels = val_labels
        self.test_corr = test_corr
        self.test_ccle_names = test_ccle_names
        self.test_labels = test_labels
        self.test_preds = test_preds

    def log_stats(self):
        logger.info(f"Training correlation: {self.train_corr}")
        logger.info(f"Validation correlation: {self.val_corr}")
        logger.info(f"Test correlation: {self.test_corr}")

    def __str__(self):
        return f"XGBoostRankingTrainer(num_epochs={self.num_epochs}, model_config={self.model_config})"
