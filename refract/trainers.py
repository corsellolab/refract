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
from sklearn.preprocessing import QuantileTransformer
from flaml import AutoML
from tqdm import tqdm

from refract.utils import get_top_features

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

def get_correlated_features(y, X, colnames, p):
    # Step 1: Compute the correlation for each column
    correlations = np.array([np.corrcoef(y, X[:, i])[0, 1] for i in range(X.shape[1])])
    
    # Step 2: Group columns by TYPE
    type_dict = {}
    for i, colname in enumerate(colnames):
        type_name = colname.split("_")[0]
        if type_name not in type_dict:
            type_dict[type_name] = []
        type_dict[type_name].append((correlations[i], colname))
    
    # Step 3: Sample the top p proportion of correlated features within each type
    selected_colnames = []
    for type_name in type_dict:
        if type_name != "LIN":
            sorted_correlations = sorted(type_dict[type_name], key=lambda x: -abs(x[0]))  # sort by absolute correlation value in descending order
            top_p_count = int(p * len(sorted_correlations))
            top_p_colnames = [colname for _, colname in sorted_correlations[:top_p_count]]
            selected_colnames.extend(top_p_colnames)
        else:
            sorted_correlations = sorted(type_dict[type_name], key=lambda x: -abs(x[0]))  # sort by absolute correlation value in descending order
            top_p_count = int(p * len(sorted_correlations))
            top_p_colnames = [colname for _, colname in sorted_correlations]
            selected_colnames.extend(top_p_colnames)

    
    # Step 4: Return the list of column names
    return selected_colnames


class AutoMLTrainer:
    """Trains a LGBM Regression Model"""

    def __init__(
        self,
        response_train,
        response_test,
        feature_df,
        response_col="LFC.cb",
        cell_line_col="ccle_name",
        feature_fraction=0.03,
    ):
        self.response_train = response_train
        self.response_test = response_test
        self.feature_df = feature_df
        self.response_col = response_col
        self.cell_line_col = cell_line_col
        self.feature_fraction = feature_fraction

        self.top_feature_names = None
        self.model = None

        self.X_test_df = None
        self.y_test = None
        self.cell_line_test = None
        self.y_test_pred = None
        self.shap_df = None
        self.test_corr = None


    def train(self):
        X_train_df = self.feature_df.loc[self.response_train[self.cell_line_col], :]
        X_train = X_train_df.values
        y_train = self.response_train[self.response_col].values
        X_test_df = self.feature_df.loc[self.response_test[self.cell_line_col], :]
        y_test = self.response_test[self.response_col].values
        
        top_features = get_correlated_features(y_train, X_train, X_train_df.columns, p=self.feature_fraction)

        X_train_df = X_train_df.loc[:, top_features]
        X_train = X_train_df.values
        X_test_df = X_test_df.loc[:, top_features]
        X_test = X_test_df.values

        sample_weights = np.abs(y_train)
        automl = AutoML()
        automl.fit(
            X_train, 
            y_train, 
            task="regression", 
            time_budget=120, 
            metric="rmse", 
            estimator_list=['xgboost', 'rf', 'lgbm'],
            sample_weight=sample_weights
        )
        self.top_feature_names = X_train_df.columns
        self.model = automl.model.estimator
        
        # predict
        y_test_pred = automl.predict(X_test)
        explainer = shap.TreeExplainer(automl.model.estimator)
        shap_values = explainer.shap_values(X_test)
        shap_values_df = pd.DataFrame(shap_values, columns=X_test_df.columns)

        # print fold correlation
        print(f"Fold correlation: {pearsonr(y_test, y_test_pred)[0]}")

        # save to self
        self.X_test_df = X_test_df
        self.y_test = y_test
        self.cell_line_test = self.response_test[self.cell_line_col]
        self.y_test_pred = y_test_pred
        self.shap_df = shap_values_df
        self.test_corr = pearsonr(y_test, y_test_pred)[0]