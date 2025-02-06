import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from shap import TreeExplainer, summary_plot
import xgboost as xgb


def train_xgboost_with_early_stopping(
        X_train, y_train, X_val, y_val, num_rounds=1000, early_stopping_rounds=50, n_threads=8
):
    """
    Trains an XGBoost regression model with early stopping and outlier emphasis in the loss function.

    Parameters:
    X_train (pd.DataFrame or np.ndarray): Training features.
    y_train (pd.Series or np.ndarray): Training target values.
    X_val (pd.DataFrame or np.ndarray): Validation features.
    num_rounds (int): Maximum number of training rounds.
    early_stopping_rounds (int): Number of rounds without improvement to trigger early stopping.
    n_threads (int): Number of CPU threads to use.

    Returns:
    xgb.Booster: The trained XGBoost model.
    """
    # Compute mean_label once
    mean_label = np.mean(y_train)

    # compute weights 
    weights = 1 + np.abs(y_train - mean_label)

    # convert data to XGBoost DMatrix format with precomputed weights
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
    dval = xgb.DMatrix(X_val, label=y_val)

    # define parameters for the XGBoost model
    params = {
        'objective': 'reg:squarederror',  # Standard regression loss
        'eval_metric': 'rmse',            # Root mean squared error for evaluation
        'eta': 0.01,                      # Learning rate
        'max_depth': 6,                   # Tree depth
        'subsample': 0.8,                 # Row subsampling
        'colsample_bytree': 0.8,          # Feature subsampling
        'lambda': 1.0,                    # L2 regularization term
        'alpha': 0.1,                     # L1 regularization term
        'tree_method': 'hist',            # Use the faster histogram-based algorithm
        'nthread': n_threads              # Number of threads (adjust to your available cores)
    }

    # Define the custom loss function without redundant calculations
    def weighted_mse_with_outlier_emphasis(preds, dtrain):
        labels = dtrain.get_label()
        errors = preds - labels

        # Retrieve precomputed weights
        weights = dtrain.get_weight()

        grad = weights * errors          # Weighted gradient
        hess = weights                   # Weighted Hessian

        return grad, hess
        
    watchlist = [(dtrain, "train"), (dval, "eval")]

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_rounds,
        evals=watchlist,
        early_stopping_rounds=early_stopping_rounds,
        obj=weighted_mse_with_outlier_emphasis,
        verbose_eval=True
    )

    print(f"Best iteration: {model.best_iteration}")
    return model
