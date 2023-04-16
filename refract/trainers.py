from sklearn.ensemble import RandomForestRegressor
import logging
import pandas as pd
import os
import numpy as np
import json
import pickle
from typing import List, Dict
from utils import RandomForestConfig, RandomForestCVConfig
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

logger = logging.getLogger(__name__)


class RFBaseTrainer:
    """Base trainer for RF model"""

    def train(self, feature_set, response_set, output_dir, config=RandomForestConfig):
        rf = RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            bootstrap=config.bootstrap,
            oob_score=config.oob_score,
            n_jobs=config.n_jobs,
            random_state=config.random_state,
        )

        # get all unique runs
        LFC = response_set.LFC
        runs = LFC[["pert_name", "dose", "pert_mfc_id"]].drop_duplicates()
        logger.info("Found %d unique runs", len(runs))

        # prepare output dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_stats = []
        for feature_name in config.feature_name_list:
            logger.info("Using feature set: %s", feature_name)
            for _, run in runs.iterrows():
                pert_name = run["pert_name"]
                pert_mfc_id = run["pert_mfc_id"]
                dose = run["dose"]
                logger.info(
                    "    Training model for run %s, %s, %s",
                    pert_name,
                    pert_mfc_id,
                    dose,
                )

                # get features and response
                X, y = response_set.get_joined_features(
                    pert_name=pert_name,
                    pert_mfc_id=pert_mfc_id,
                    dose=dose,
                    feature_set=feature_set,
                    feature_name=feature_name,
                )
                y = y.values.ravel()
                rf.fit(X, y)

                # get the feature importances
                importances = pd.DataFrame(
                    rf.feature_importances_, index=X.columns, columns=["importance"]
                )
                importances = importances.sort_values(by="importance", ascending=False)

                # add data to the imp table
                importances["pert_name"] = pert_name
                importances["pert_mfc_id"] = pert_mfc_id
                importances["dose"] = dose
                # add rank to the improtances
                importances["rank"] = importances["importance"].rank(ascending=False)
                # add the feature name
                importances["feature_name"] = feature_name

                # compute MSE, MSE standard error, R2, Pearson correlation for the  model
                y_pred = rf.predict(X)
                y_true = y
                mse = ((y_true - y_pred) ** 2).mean()
                mse_se = mse / len(y_true)
                r2 = rf.score(X, y)
                pearson = np.corrcoef(y_true, y_pred)[0, 1]
                model_stats.append(
                    {
                        "pert_name": pert_name,
                        "pert_mfc_id": pert_mfc_id,
                        "dose": dose,
                        "feature_name": feature_name,
                        "mse": mse,
                        "mse_se": mse_se,
                        "r2": r2,
                        "pearson": pearson,
                    }
                )

                # save feature importance output
                if not os.path.exists(f"{output_dir}/{pert_name}_{feature_name}"):
                    os.makedirs(f"{output_dir}/{pert_name}_{feature_name}")
                importances.to_csv(
                    f"{output_dir}/{pert_name}_{feature_name}/{pert_mfc_id}_{feature_name}_{dose}.csv"
                )

        # save model stats
        model_stats = pd.DataFrame(model_stats)
        model_stats.to_csv(f"{output_dir}/Model_table.csv")

        # save config
        with open(f"{output_dir}/config.json", "w") as f:
            json.dump(config.__dict__, f)


class NestedCVRFTrainer:
    """Perform nested cross validation training for RF model"""

    def train(
        self,
        feature_set,
        response_set,
        output_dir,
        model=RandomForestRegressor,
        config=RandomForestCVConfig,
    ):
        # get all unique runs
        runs = self._get_unique_runs(response_set)
        logger.info("Found %d unique runs", len(runs))

        # prepare output dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # run training over feature sets and runs
        model_stats = []
        for feature_name in config.feature_name_list:
            logger.info("Using feature set: %s", feature_name)
            for _, run in runs.iterrows():
                pert_name = run["pert_name"]
                pert_mfc_id = run["pert_mfc_id"]
                dose = run["dose"]
                logger.info(
                    "    Training model for run %s, %s, %s",
                    pert_name,
                    pert_mfc_id,
                    dose,
                )

                # get features and response
                X, y = response_set.get_joined_features(
                    pert_name=pert_name,
                    pert_mfc_id=pert_mfc_id,
                    dose=dose,
                    feature_set=feature_set,
                    feature_name=feature_name,
                )
                y = y.values.ravel()

                # perform nested cross validation
                logger.info("Training nested cross validation...")
                cv_results = self._train_nested_cv(X, y, model, config)
                # get feature importances
                logger.info("Computing feature importances...")
                importances = self._compute_cv_feature_importances(
                    X,
                    cv_results,
                    pert_name=pert_name,
                    pert_mfc_id=pert_mfc_id,
                    dose=dose,
                    feature_name=feature_name,
                )

                # get model stats
                logger.info("Computing model stats...")
                model_stat = self._compute_cv_model_performance(
                    X,
                    y,
                    cv_results,
                    pert_name=pert_name,
                    pert_mfc_id=pert_mfc_id,
                    dose=dose,
                    feature_name=feature_name,
                )
                model_stats.append(model_stat)

                # save feature importance output
                logger.info("Saving output...")
                if not os.path.exists(f"{output_dir}/{pert_name}_{feature_name}"):
                    os.makedirs(f"{output_dir}/{pert_name}_{feature_name}")
                importances.to_csv(
                    f"{output_dir}/{pert_name}_{feature_name}/{pert_mfc_id}_{feature_name}_{dose}.csv"
                )

                # save cv_results
                with open(
                    f"{output_dir}/{pert_name}_{feature_name}/{pert_mfc_id}_{feature_name}_{dose}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(cv_results, f)

        # save model stats
        model_stats = pd.DataFrame(model_stats)
        model_stats.to_csv(f"{output_dir}/Model_table.csv", index=False)

        # save config
        with open(f"{output_dir}/config.json", "w") as f:
            json.dump(config.__dict__, f)

    def _get_unique_runs(self, response_set):
        """Get the unique runs from the response set"""
        LFC = response_set.LFC
        runs = LFC[["pert_name", "dose", "pert_mfc_id"]].drop_duplicates()
        return runs

    def _init_model(self, model, config):
        """Initialize the RF model"""
        return model(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            bootstrap=config.bootstrap,
            oob_score=config.oob_score,
            n_jobs=config.n_jobs,
            random_state=config.random_state,
        )

    def _train_nested_cv(self, X, y, model, config):
        """Perform nested cross validation training for RF model"""
        kf_outer = KFold(
            n_splits=config.n_splits, shuffle=True, random_state=config.random_state
        )
        kf_inner = KFold(
            n_splits=config.n_splits - 1, shuffle=True, random_state=config.random_state
        )
        param_grid = ParameterGrid(config.param_grid)

        # outer loop
        outer_cv_results = []
        for idx, (train_val_index, test_index) in enumerate(kf_outer.split(X, y)):
            logger.info("Fitting outer fold %d/%d", idx + 1, config.n_splits)
            X_train_val, X_test = X.iloc[train_val_index], X.iloc[test_index]
            y_train_val, y_test = y[train_val_index], y[test_index]

            inner_cv_results = []
            for idx, (train_index, val_index) in enumerate(
                kf_inner.split(X_train_val, y_train_val)
            ):
                logger.info(
                    "    Fitting inner fold %d/%d", idx + 1, config.n_splits - 1
                )
                X_train, X_val = (
                    X_train_val.iloc[train_index],
                    X_train_val.iloc[val_index],
                )
                y_train, y_val = y_train_val[train_index], y_train_val[val_index]

                grid_search_results = []
                for params in tqdm(param_grid):
                    # set the model parameters
                    cv_model = self._init_model(model, config)
                    cv_model.set_params(**params)
                    # fit the model
                    cv_model.fit(X_train, y_train)
                    # get the scores
                    train_mse = mean_squared_error(cv_model.predict(X_train), y_train)
                    val_mse = mean_squared_error(cv_model.predict(X_val), y_val)
                    # save the results
                    grid_search_results.append(
                        {
                            "train_index": train_index,
                            "val_index": val_index,
                            "params": params,
                            "train_mse": train_mse,
                            "val_mse": val_mse,
                            "model": cv_model,
                        }
                    )
                # select the dictionary of the best model in the inner loop
                best_model = min(grid_search_results, key=lambda x: x["val_mse"])
                # save the results
                inner_cv_results.append(best_model)

            # get results from all best models in the inner loop
            outer_cv_results.append(
                {
                    "train_val_index": train_val_index,
                    "test_index": test_index,
                    "models": [x["model"] for x in inner_cv_results],
                    "best_params": [x["params"] for x in inner_cv_results],
                    "train_mse": np.mean([x["train_mse"] for x in inner_cv_results]),
                    "val_mse": np.mean([x["val_mse"] for x in inner_cv_results]),
                    "test_mse": np.mean(
                        [
                            mean_squared_error(x["model"].predict(X_test), y_test)
                            for x in inner_cv_results
                        ]
                    ),
                }
            )

        return outer_cv_results

    def _compute_cv_feature_importances(
        self,
        X: pd.DataFrame,
        cv_results: List[Dict],
        pert_name: str,
        pert_mfc_id: str,
        dose: float,
        feature_name: str,
    ):
        """Compute the feature importances from a list of CV results.
        cv_results is the output of _train_nested_cv

        Args:
            X (pd.DataFrame): Feature dataframe
            cv_results (List[Dict]): Output of _train_nested_cv
            pert_name (str): Perturbation name
            pert_mfc_id (str): Perturbation MFC ID
            dose (float): Perturbation dose
            feature_name (str): Feature name

        Returns:
            pd.DataFrame: Feature importances, as a dataframe
        """
        fold_importance = []
        for cv_fold in cv_results:
            # get the models
            models = cv_fold["models"]
            # compute the mean feature importance
            for model in models:
                importances = pd.DataFrame(
                    model.feature_importances_, index=X.columns, columns=["importance"]
                )
                fold_importance.append(importances)

        # get the mean feature importance
        importance = (
            pd.concat(fold_importance, axis=1).mean(axis=1).sort_values(ascending=False)
        )
        importance = pd.DataFrame(importance, columns=["importance"])

        # add metadata
        importance["pert_name"] = pert_name
        importance["pert_mfc_id"] = pert_mfc_id
        importance["dose"] = dose
        importance["feature_set"] = feature_name
        importance["rank"] = importance["importance"].rank(ascending=False).astype(int)
        importance.index.rename("feature", inplace=True)

        return importance

    def _compute_cv_model_performance(
        self,
        X: pd.DataFrame,
        y: np.array,
        cv_results: List[Dict],
        pert_name: str,
        pert_mfc_id: str,
        dose: float,
        feature_name: str,
    ):
        """Compute the model performance from a list of CV results.
        cv_results is the output of _train_nested_cv

        Args:
            X (pd.DataFrame): Feature matrix
            y (np.array): Response vector
            cv_results (List[Dict]): Output of _train_nested_cv
            pert_name (str): Perturbation name
            pert_mfc_id (str): Perturbation MFC ID
            dose (float): Perturbation dose
            feature_name (str): Feature name

        Returns:
            pd.DataFrame: Model performance, as a dataframe
        """
        # get the predictions on each test fold
        y_pred_all = []
        y_true_all = []
        for cv_fold in cv_results:
            test_index = cv_fold["test_index"]
            X_test = X.iloc[test_index]
            y_true = y[test_index]
            models = cv_fold["models"]
            # get predictions for all models
            individiual_model_predictions = []
            for model in models:
                y_pred = model.predict(X_test)
                individiual_model_predictions.append(y_pred)
            # average predictions
            y_pred = np.mean(individiual_model_predictions, axis=0)
            # save the predictions
            y_pred_all.extend(y_pred)
            y_true_all.extend(y_true)

        # get the performance metrics
        mse = mean_squared_error(y_true_all, y_pred_all)
        mse_se = mse / np.sqrt(len(y_true_all))
        r2 = r2_score(y_true_all, y_pred_all)
        pearson = np.corrcoef(y_true_all, y_pred_all)[0, 1]
        model_stats = {"mse": mse, "mse_se": mse_se, "r2": r2, "pearson": pearson}
        model_stats["pert_name"] = pert_name
        model_stats["pert_mfc_id"] = pert_mfc_id
        model_stats["dose"] = dose
        model_stats["feature_name"] = feature_name
        return model_stats
