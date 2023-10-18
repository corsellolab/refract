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

        self.X_train_val = None
        self.y_train_val = None
        self.X_test = None
        self.X_test_df = None
        self.y_test = None
        self.cell_line_train_val = None
        self.cell_line_test = None

        self.y_test_pred = None
        self.shap_df = None

    def top_n_feature_indices(self, rf_model, n):
        # get feature importances
        importances = rf_model.feature_importances_
        # get the indices of the top n features
        indices = sorted(
            range(len(importances)), key=lambda i: importances[i], reverse=True
        )[:n]
        return indices

    def train_ranker(self, X_train, y_train, params, X_val=None, y_val=None):
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
        bst = lgb.train(params, train_data)

        if X_val and y_val:
            # compute validation pearson correlation
            val_corr = pearsonr(y_val, bst.predict(X_val))[0]
            return bst, val_corr
        else:
            return bst

    def _trial_objective(self, trial, X_train, y_train):
        params = {
            "objective": "regression",
            "metric": "l2",
            "verbosity": -1,
            "weighting": trial.suggest_int("weighting", 0, 2),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 50),
            "max_depth": trial.suggest_int("max_depth", 1, 20),
            "num_boost_round": trial.suggest_int("num_boost_round", 50, 1000),
        }

        ranker_corrs = []
        for _ in range(5):
            _, val_corr = self._train_ranker(X_train, y_train, params, val_set=True)
            ranker_corrs.append(val_corr)
        return np.mean(ranker_corrs)

    def train(self):
        ### STAGE 1: get correlated features
        # compute the top features
        train_idx, val_idx = train_test_split(self.response_train, test_size=0.2)
        response_train = self.response_train.iloc[train_idx, :].copy()
        response_val = self.response_train.iloc[val_idx, :].copy()
        response_test = self.response_test.copy()

        y_train = response_train[self.response_col].values
        cell_line_train = response_train[self.cell_line_col].values
        y_val = response_val[self.response_col].values
        cell_line_val = response_val[self.cell_line_col].values
        self.y_train_val = np.concatenate([y_train, y_val])
        self.cell_line_train_val = np.concatenate([cell_line_train, cell_line_val])
        self.y_test = response_test[self.response_col].values
        self.cell_line_test = response_test[self.cell_line_col].values

        top_features = get_top_features(
            response_train,
            self.feature_df,
            self.response_col,
            self.feature_fraction,
            n_jobs=4,
        )
        top_features_df = self.feature_df.loc[:, top_features]

        ### STAGE 2: random forest regression
        X_train = top_features_df.loc[cell_line_train, :].values

        # fit a random forest regressor object
        rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=4)
        rf.fit(X_train, y_train)

        top_features = self.top_n_feature_indices(rf, 200)
        self.top_feature_names = top_features
        first_stage_top_features_df = top_features_df.iloc[:, top_features]

        #### STAGE 3: fit LGBM model on top features
        self.X_train_val = first_stage_top_features_df.loc[
            self.cell_line_train_val, :
        ].values
        X_train = first_stage_top_features_df.loc[cell_line_train, :].values
        X_val = first_stage_top_features_df.loc[cell_line_val, :].values
        self.X_test = first_stage_top_features_df.loc[self.cell_line_test, :].values

        # optimize hyperparameters
        study = optuna.create_study(direction="maximize")
        obj = partial(
            self._trial_objective,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )
        study.optimize(obj, n_trials=self.num_hyp_trial)

        # get best params
        trial = study.best_trial
        self.best_params = trial.params

        # train the model with best params and all data
        self.lgbm_model = self._train_ranker(
            self.X_train_val, self.y_train_val, self.best_params
        )

        # predict
        self.y_test_pred = self.predict(self.X_test)

        # compute shap values
        self.shap_df = self.compute_shap_values(self.X_test)
        # set X_test_df
        self.X_test_df = first_stage_top_features_df.loc[self.cell_line_test, :]

    def predict(self, X):
        return self.model.predict(X)

    def compute_shap_values(self, X):
        explainer = shap.TreeExplainer(self.lgbm_model)
        shap_values = explainer.shap_values(X)
        shap_df = pd.DataFrame(shap_values, columns=self.top_feature_names)
        return shap_df
