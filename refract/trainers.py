# trainer for the XGBoost ranking model
import logging
from functools import partial

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupKFold, GridSearchCV
from sklearn.preprocessing import QuantileTransformer
from flaml import AutoML
from tqdm import tqdm

from refract.utils import get_correlated_features

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

# Hyperparameter grid
PARAM_GRID = {
    'n_estimators': [300],
    'max_depth': [7],
    'min_samples_split': [2, 5, 7],
    'min_samples_leaf': [2, 5, 7]
}


class AutoMLTrainer:
    """Trains a LGBM Regression Model"""

    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        feature_cols,
        drug_name,
        fold_assignment
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_cols = feature_cols
        self.drug_name = drug_name
        self.fold_assignment = fold_assignment

    def select_features(self):
        # get the top features
        selected_feature_names = get_correlated_features(
            self.y_train.loc[:, "LFC.cb"],
            self.X_train.values,
            self.feature_cols
        )
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test
        # drop duplicates
        X_train = X_train.drop_duplicates()
        X_test = X_test.drop_duplicates()
        # subset to the key pert and selected features
        y_train = y_train.loc[y_train.pert_name == self.drug_name, :]
        y_test = y_test.loc[y_test.pert_name == self.drug_name, :]
        X_train = y_train.merge(X_train, left_index=True, right_index=True, how='left').loc[:, selected_feature_names]
        X_test = y_test.merge(X_test, left_index=True, right_index=True, how='left').loc[:, selected_feature_names]
        y_train = y_train.loc[:, "LFC.cb"]
        y_test = y_test.loc[:, "LFC.cb"]
        # get groups for this subset
        train_groups = [self.fold_assignment[ccle_name] for ccle_name in X_train.index]

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.train_groups = train_groups

    def train(self):        
        X_train = self.X_train.values
        X_test = self.X_test.values
        y_train = self.y_train.values
        y_test = self.y_test.values

        # Randomized Search CV for hyperparameter tuning
        inner_cv = GroupKFold(n_splits=9)
        rf = RandomForestRegressor(random_state=42, n_jobs=8)
        search = GridSearchCV(rf, PARAM_GRID, cv=inner_cv, n_jobs=2)
        search.fit(X_train, y_train, groups=self.train_groups)

        # Best model
        best_model = search.best_estimator_

        self.top_feature_names = self.X_train.columns
        self.model = best_model
        
        # predict
        y_test_pred = best_model.predict(X_test)
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        shap_values_df = pd.DataFrame(shap_values, columns=self.X_test.columns)

        # print fold correlation
        print(f"Fold correlation: {pearsonr(y_test, y_test_pred)[0]}")

        # save to self
        self.X_test_df = self.X_test
        self.y_test = y_test
        self.cell_line_test = self.X_test.index.values
        self.y_test_pred = y_test_pred
        self.shap_df = shap_values_df
        self.test_corr = pearsonr(y_test, y_test_pred)[0]