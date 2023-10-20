# trainer for the XGBoost ranking model
import logging
from functools import partial

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from refract.utils import get_top_features

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


class LGBMTrainer:
    """Trains a LGBM Regression Model"""

    def __init__(
        self,
        response_train,
        response_test,
        feature_df,
        response_col="LFC.cb",
        cell_line_col="ccle_name",
        num_hyp_trials=200,
        feature_fraction=0.03,
    ):
        self.response_train = response_train
        self.response_test = response_test
        self.feature_df = feature_df
        self.response_col = response_col
        self.cell_line_col = cell_line_col
        self.num_hyp_trials = num_hyp_trials
        self.feature_fraction = feature_fraction

        self.top_feature_names = None
        self.best_params = None
        self.lgbm_model = None

        self.X_test_df = None
        self.y_test = None
        self.cell_line_test = None
        self.y_test_pred = None
        self.shap_df = None
        self.test_corr = None

    def top_n_feature_indices(self, rf_model, n):
        # get feature importances
        importances = rf_model.feature_importances_
        # get the indices of the top n features
        indices = sorted(
            range(len(importances)), key=lambda i: importances[i], reverse=True
        )[:n]
        return indices

    def train_lgbm_regressor(self, X_train, y_train, params):
        # set label weight
        weighting = params["weighting"]
        if weighting == 0:
            train_data = lgb.Dataset(X_train, label=y_train)
        else:
            num_bins = 1000
            bins = np.linspace(y_train.min(), y_train.max(), num=num_bins)
            y_train_weight = (
                np.abs(
                    np.digitize(y_train, bins) - np.median(np.digitize(y_train, bins))
                )
                ** params["weighting"]
            )
            # create training dataset
            train_data = lgb.Dataset(X_train, label=y_train, weight=y_train_weight)

        # train model
        num_boost_round = params["num_trees_train"]
        bst = lgb.train(params, train_data, num_boost_round=num_boost_round)

        return bst

    def _optuna_objective(self, trial):
        params = {
            "objective": "regression",
            "metric": "l2",
            "verbosity": -1,
            "weighting": trial.suggest_int("weighting", 0, 2),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "num_leaves": trial.suggest_int("num_leaves", 2, 128),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 20),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "num_trees_train": trial.suggest_int("num_trees_train", 10, 150),
        }
        ranker_corrs = []
        for _ in range(5):
            val_corr = self._optuna_train(params)
            ranker_corrs.append(val_corr)
        return np.mean(ranker_corrs)

    def _optuna_train(self, params):
        response_train, response_val = train_test_split(
            self.response_train, test_size=0.2, random_state=42
        )
        y_train = response_train[self.response_col].values
        y_val = response_val[self.response_col].values
        cell_line_train = response_train[self.cell_line_col].values
        cell_line_val = response_val[self.cell_line_col].values

        X_train = self.feature_df.loc[cell_line_train, self.top_feature_names].values
        X_val = self.feature_df.loc[cell_line_val, self.top_feature_names].values

        # train the model with best params and all data
        self.lgbm_model = self.train_lgbm_regressor(X_train, y_train, params)

        # predict
        self.y_val_pred = self.lgbm_model.predict(X_val)

        # compute loss
        val_loss = np.mean((y_val - self.y_val_pred) ** 2)
        return val_loss

    def get_correlated_features(self):
        top_features = get_top_features(
            self.response_train,
            self.feature_df,
            self.response_col,
            self.feature_fraction,
            n_jobs=4,
        )
        self.correlated_features = top_features

    def hyperparameter_optimization(self):
        # hyperparameter sweep with optuna
        study = optuna.create_study(direction="minimize")
        study.optimize(self._optuna_objective, n_trials=self.num_hyp_trials)
        self.best_params = study.best_trial.params

    def train_first_stage(self):
        top_features = []
        for train_iter in range(5):
            response_train, _ = train_test_split(
                self.response_train, test_size=0.2, random_state=train_iter
            )
            y_train = response_train[self.response_col].values
            cell_line_train = response_train[self.cell_line_col].values
            X_train_df = self.feature_df.loc[cell_line_train, self.correlated_features]
            X_train = X_train_df.values
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=4)
            rf.fit(X_train, y_train)
            top_feature_idx = self.top_n_feature_indices(rf, 100)
            top_feature_names = X_train_df.columns[top_feature_idx]
            top_features.append(top_feature_names)
        # compute the intersection
        self.top_feature_names = list(set.union(*map(set, top_features)))

    def train_second_stage(self):
        # get the X and y train
        y_train = self.response_train[self.response_col].values
        cell_line_train = self.response_train[self.cell_line_col].values
        X_train = self.feature_df.loc[cell_line_train, self.top_feature_names].values

        # get the X and y test
        self.y_test = self.response_test[self.response_col].values
        self.cell_line_test = self.response_test[self.cell_line_col].values
        X_test = self.feature_df.loc[self.cell_line_test, self.top_feature_names].values

        # train the model with best params and all data
        self.lgbm_model = self.train_lgbm_regressor(X_train, y_train, self.best_params)

        # predict
        self.y_test_pred = self.lgbm_model.predict(X_test)

        # compute_correlation
        self.test_corr = pearsonr(self.y_test, self.y_test_pred)[0]

        # compute shap values
        self.shap_df = self.compute_shap_values(X_test)
        self.X_test_df = pd.DataFrame(X_test, columns=self.top_feature_names)

    def compute_shap_values(self, X):
        explainer = shap.TreeExplainer(self.lgbm_model)
        shap_values = explainer.shap_values(X)
        shap_df = pd.DataFrame(shap_values, columns=self.top_feature_names)
        return shap_df
