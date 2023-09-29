# trainer for the XGBoost ranking model
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from tqdm import tqdm

from refract.utils import (
    dataset_to_group_df,
    dataset_to_individual_df,
    moving_window_average,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


class XGBoostRankingTrainer:
    """Trains an XGBoost ranking model"""

    def __init__(self, train_ds, val_ds, test_ds, num_epochs, model_config={}):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.num_epochs = num_epochs
        self.model_config = model_config

        # training stats
        self.train_corr = None
        self.train_results_df = None
        self.val_corr = None
        self.val_results_df = None
        self.test_corr = None
        self.test_results_df = None
        self.xgb_model = None

        # only save test labels and preds to save space
        # needed for SHAP
        self.shap_values = None
        self.test_features = None
        self.test_feature_names = None

    def get_model_config(self, config={}):
        params = {
            "objective": "rank:pairwise",
            "colsample_bytree": 0.5,
            "colsample_bylevel": 0.5,
            "colsample_bynode": 0.5,
            "max_depth": 2,
            "seed": 42,
        }
        params.update(config)
        return params

    def train(self):
        # get grouped training features
        logger.info("    Generating slates...")
        # get train, val, test features
        train_features, train_labels, train_ccle_names = dataset_to_individual_df(
            self.train_ds
        )
        val_features, val_labels, val_ccle_names = dataset_to_individual_df(self.val_ds)
        test_features, test_labels, test_ccle_names = dataset_to_individual_df(
            self.test_ds
        )

        # convert the data to DMatrix
        logger.info("    Converting to DMatrix...")
        train_dmatrix = xgb.DMatrix(train_features, label=train_labels)
        val_dmatrix = xgb.DMatrix(val_features, label=val_labels)
        test_dmatrix = xgb.DMatrix(test_features, label=test_labels)

        # get model params
        params = self.get_model_config(self.model_config)

        # train
        models = []
        val_corr = []
        logger.info("    Incremental training...")
        for _ in tqdm(range(self.num_epochs)):
            # get one training epoch
            group_train_features, group_train_labels, groups = dataset_to_group_df(
                self.train_ds, 1
            )
            # format for training
            group_train_dmatrix = xgb.DMatrix(
                group_train_features, label=group_train_labels, group=groups
            )
            if not self.xgb_model:
                self.xgb_model = xgb.train(
                    params, group_train_dmatrix, num_boost_round=1
                )
            else:
                self.xgb_model = xgb.train(
                    params,
                    group_train_dmatrix,
                    num_boost_round=1,
                    xgb_model=self.xgb_model,
                )
            val_preds = self.xgb_model.predict(val_dmatrix)
            val_corr.append(np.corrcoef(val_preds, val_labels)[0, 1])
            models.append(self.xgb_model)

        smoothed_val_corr = moving_window_average(val_corr, 5)
        best_val_corr_idx = np.argmax(smoothed_val_corr)
        self.xgb_model = models[best_val_corr_idx]

        # get train, val, test preds
        self.train_preds = self.xgb_model.predict(train_dmatrix)
        self.val_preds = self.xgb_model.predict(val_dmatrix)
        self.test_preds = self.xgb_model.predict(test_dmatrix)

        # store as dataframes
        self.train_results_df = pd.DataFrame(
            {
                "ccle_name": train_ccle_names,
                "preds": -1 * self.train_preds,
            }
        )
        self.val_results_df = pd.DataFrame(
            {
                "ccle_name": val_ccle_names,
                "preds": -1 * self.val_preds,
            }
        )
        self.test_results_df = pd.DataFrame(
            {
                "ccle_name": test_ccle_names,
                "preds": -1 * self.test_preds,
            }
        )

        # join to get unscaled responses
        unscaled_train = self.train_ds.unscaled_response_df.loc[
            :, ["ccle_name", "LFC.cb"]
        ]
        unscaled_val = self.val_ds.unscaled_response_df.loc[:, ["ccle_name", "LFC.cb"]]
        unscaled_test = self.test_ds.unscaled_response_df.loc[
            :, ["ccle_name", "LFC.cb"]
        ]

        self.train_results_df = self.train_results_df.merge(
            unscaled_train, on="ccle_name"
        )
        self.val_results_df = self.val_results_df.merge(unscaled_val, on="ccle_name")
        self.test_results_df = self.test_results_df.merge(unscaled_test, on="ccle_name")

        # compute correlations
        self.train_corr = np.corrcoef(
            self.train_results_df["LFC.cb"], self.train_results_df.preds
        )[0, 1]
        self.val_corr = np.corrcoef(
            self.val_results_df["LFC.cb"], self.val_results_df.preds
        )[0, 1]
        self.test_corr = np.corrcoef(
            self.test_results_df["LFC.cb"], self.test_results_df.preds
        )[0, 1]

        # get SHAP values
        explainer = shap.TreeExplainer(self.xgb_model)
        self.shap_values = explainer.shap_values(test_features)
        self.test_features = self.test_ds.joined_df.iloc[:, :-1].values
        self.test_feature_names = self.test_ds.top_features

        # free up some memory
        del group_train_dmatrix
        del train_dmatrix
        del val_dmatrix
        del test_dmatrix
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
