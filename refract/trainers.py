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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer, r2_score

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

    def train(self, feature_set, response_set, output_dir, config=RandomForestCVConfig):
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

                # perform nested cross validation
                logger.info("Training nested cross validation...")
                cv_results = self._train_nested_cv(X, y, config)
                # get feature importances
                logger.info("Computing feature importances...")
                importances = self._compute_cv_feature_importances(cv_results)

                # add data to the imp table
                importances["pert_name"] = pert_name
                importances["pert_mfc_id"] = pert_mfc_id
                importances["dose"] = dose
                # add rank to the improtances
                importances["rank"] = importances["importance"].rank(ascending=False)
                # add the feature name
                importances["feature_name"] = feature_name

                # get model stats
                logger.info("Computing model stats...")
                model_stats.append(
                    self._compute_cv_model_performance(X, y, cv_results)
                )

                # save feature importance output
                if not os.path.exists(f"{output_dir}/{pert_name}_{feature_name}"):
                    os.makedirs(f"{output_dir}/{pert_name}_{feature_name}")
                importances.to_csv(
                    f"{output_dir}/{pert_name}_{feature_name}/{pert_mfc_id}_{feature_name}_{dose}.csv"
                )

                # save cv_results
                with open("{output_dir}/{pert_name}_{feature_name}/{pert_mfc_id}_{feature_name}_{dose}.pkl", "wb") as f:
                    pickle.dump(cv_results, f)

        # save model stats
        model_stats = pd.DataFrame(model_stats)
        model_stats.to_csv(f"{output_dir}/Model_table.csv")

        # save config
        with open(f"{output_dir}/config.json", "w") as f:
            json.dump(config.__dict__, f)


    def _train_nested_cv(self, X, y, config):
        """Perform nested cross validation training for RF model"""
        kf_outer = KFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_state)
        mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

        cv_results = []
        for train_val_index, test_index in kf_outer.split(X,y):
            X_train_val, X_test = X.iloc[train_val_index], X.iloc[test_index]
            y_train_val, y_test = y[train_val_index], y[test_index]
            y_train_val = y_train_val.ravel()
            y_test = y_test.ravel()

            # grid search
            grid = GridSearchCV(
                estimator=RandomForestRegressor(
                    n_estimators=config.n_estimators,
                    max_depth=config.max_depth,
                    min_samples_split=config.min_samples_split,
                    min_samples_leaf=config.min_samples_leaf,
                    bootstrap=config.bootstrap,
                    oob_score=config.oob_score,
                    n_jobs=config.n_jobs,
                    random_state=config.random_state,
                ),
                param_grid=config.param_grid,
                cv=config.n_splits - 1,
                scoring=mse_scorer,
                n_jobs=config.n_jobs,
                verbose=1,
                refit=True
            )

            # fit the grid search model
            grid.fit(X_train_val, y_train_val)
            # get the best estimator
            best_model = grid.best_estimator_
            # get the best parameters
            best_params = grid.best_params_

            # get the train_val score
            train_val_mse = mean_squared_error(best_model.predict(X_train_val), y_train_val)
            # get the test score
            test_mse = mean_squared_error(best_model.predict(X_test), y_test)

            # save the results
            fold_out = {}
            fold_out["train_val_index"] = train_val_index
            fold_out["test_index"] = test_index
            fold_out["train_val_mse"] = train_val_mse
            fold_out["test_mse"] = test_mse
            fold_out["best_params"] = best_params
            fold_out["best_model"] = best_model
            cv_results.append(fold_out)

        return cv_results


    def _compute_cv_feature_importances(self, cv_results: List[Dict]):
        """Compute the feature importances from a list of CV results. 
        cv_results is the output of _train_nested_cv

        Args:
            cv_results (List[Dict]): Output of _train_nested_cv

        Returns:
            pd.DataFrame: Feature importances, as a dataframe
        """
        fold_importance = []
        for cv_fold in cv_results:
            # get best model
            rf = cv_fold["best_model"]
            # get feature importances
            importances = pd.DataFrame(
                rf.feature_importances_, index=X.columns, columns=["importance"]
            )
            fold_importance.append(importances)

        # get the mean feature importance 
        merged_df = pd.concat(fold_importance)
        average_df = merged_df.groupby(merged_df.index).agg({"importance": "mean"})
        result_df = merged_df.drop("importance", axis=1).drop_duplicates().join(average_df, left_index=True, right_index=True)
        return result_df

    def _compute_cv_model_performance(X: pd.DataFrame, y: np.array, cv_results: List[Dict]):
        """Compute the model performance from a list of CV results. 
        cv_results is the output of _train_nested_cv

        Args:
            X (pd.DataFrame): Feature matrix
            y (np.array): Response vector
            cv_results (List[Dict]): Output of _train_nested_cv

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
            rf = cv_fold["best_model"]
            y_pred = rf.predict(X_test)

            y_pred_all.append(y_pred)
            y_true_all.append(y_true)

        # get the performance metrics
        mse = mean_squared_error(y_true_all, y_pred_all)
        mse_se = mse / np.sqrt(len(y_true_all))
        r2 = r2_score(y_true_all, y_pred_all)
        pearson = np.corrcoef(y_true_all, y_pred_all)[0,1]
        model_stats = {
            "mse": mse,
            "mse_se": mse_se,
            "r2": r2,
            "pearson": pearson
        }
        return model_stats