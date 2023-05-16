import json
import logging
import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid, cross_val_score
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from xgboost import XGBRegressor

from refract.datasets import FeatureSet, ResponseSet
from refract.results import CVResult, GridSearchResult, ModelStats, TrainerResult
from refract.utils import RandomForestCVConfig

logger = logging.getLogger(__name__)


class NestedCVRFTrainerNoRetrain:
    """Perform nested cross validation training for RF model
    Average predictions across all models, do not retrain with all data
    """

    def __init__(
        self,
        pert_name: str,
        pert_mfc_id: str,
        dose: float,
        feature_name: str,
        output_dir: str,
        feature_set: FeatureSet,
        response_set: ResponseSet,
        config: RandomForestCVConfig,
        model_class=RandomForestRegressor,
    ):
        # store instance variables
        self.pert_name = pert_name
        self.pert_mfc_id = pert_mfc_id
        self.dose = dose
        self.feature_name = feature_name
        self.output_dir = output_dir
        self.feature_set = feature_set
        self.response_set = response_set
        self.config = config
        self.model_class = model_class

        # create output dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # create TrainerResult to store results
        self.trainer_result = TrainerResult(
            pert_name=pert_name,
            pert_mfc_id=pert_mfc_id,
            dose=dose,
            feature_name=feature_name,
            output_dir=output_dir,
        )

    def train(self):
        # get features and response
        X, y = self.response_set.get_joined_features(
            pert_name=self.pert_name,
            pert_mfc_id=self.pert_mfc_id,
            dose=self.dose,
            feature_set=self.feature_set,
            feature_name=self.feature_name,
        )

        # perform nested cross validation
        logger.info("Training nested cross validation...")
        self._train_nested_cv(X, y)

        # get feature importances
        logger.info("Computing feature importances...")
        self._compute_cv_feature_importances(X)

        # get model stats
        logger.info("Computing model stats...")
        self._compute_model_stats(X, y)

        # save output
        logger.info("Saving output...")
        self._save_output()

        # save config
        with open(f"{self.trainer_result.output_dir}/config.json", "w") as f:
            json.dump(self.config.__dict__, f)

    def _save_output(self):
        """Save output to disk"""
        # save model stats
        self.trainer_result.save_model_stats()
        # save predictions
        self.trainer_result.save_predictions()
        # save feature importances
        self.trainer_result.save_feature_importances()
        # save trainer
        self.trainer_result.save_trainer_result()

    def _get_unique_runs(self, response_set):
        """Get the unique runs from the response set"""
        LFC = response_set.LFC
        runs = LFC[["pert_name", "dose", "pert_mfc_id"]].drop_duplicates()
        return runs

    def _init_model(self, model, config):
        """Initialize the RF model"""
        return model(
            n_jobs=config.n_jobs,
            random_state=config.random_state,
        )

    def _train_nested_cv(self, X, y):
        """Perform nested cross validation training for RF model"""
        kf_outer = KFold(
            n_splits=self.config.n_splits,
            shuffle=True,
            random_state=self.config.random_state,
        )
        kf_inner = KFold(
            n_splits=self.config.n_splits - 1,
            shuffle=True,
            random_state=self.config.random_state,
        )
        param_grid = ParameterGrid(self.config.param_grid)

        #### OUTER LOOP ####
        for idx, (train_val_index, test_index) in enumerate(kf_outer.split(X, y)):
            logger.info("Fitting outer fold %d/%d", idx + 1, self.config.n_splits)
            X_train_val, X_test = X.iloc[train_val_index], X.iloc[test_index]
            y_train_val, y_test = y.iloc[train_val_index], y.iloc[test_index]

            ccle_names = y_test.index.tolist()
            y_train_val = y_train_val.values.ravel()
            y_test = y_test.values.ravel()

            #### INNER LOOP ####
            inner_cv_results = []
            for idx, (train_index, val_index) in enumerate(
                kf_inner.split(X_train_val, y_train_val)
            ):
                logger.info(
                    "    Fitting inner fold %d/%d", idx + 1, self.config.n_splits - 1
                )
                X_train, X_val = (
                    X_train_val.iloc[train_index],
                    X_train_val.iloc[val_index],
                )
                y_train, y_val = y_train_val[train_index], y_train_val[val_index]

                imputer = SimpleImputer(strategy="median", keep_empty_features=True)
                logger.info("Imputing missing values...")
                logger.info("    imputation strategy: median")
                X_train = imputer.fit_transform(X_train)
                X_val = imputer.transform(X_val)

                ### GRID SEARCH ###
                grid_search_results = []
                for params in tqdm(param_grid):
                    # set the model parameters
                    cv_model = self._init_model(self.model_class, self.config)
                    cv_model.set_params(**params)
                    # fit the model
                    cv_model.fit(X_train, y_train)
                    # get the scores
                    train_mse = mean_squared_error(cv_model.predict(X_train), y_train)
                    val_mse = mean_squared_error(cv_model.predict(X_val), y_val)
                    # save the results
                    gs_result = GridSearchResult(
                        train_index=train_index,
                        val_index=val_index,
                        params=params,
                        train_mse=train_mse,
                        val_mse=val_mse,
                        model=cv_model,
                        imputer=imputer,
                    )
                    grid_search_results.append(gs_result.__dict__)

                ### END GRID SEARCH ###
                # select the dictionary of the best model in the inner loop
                best_model = min(grid_search_results, key=lambda x: x["val_mse"])
                # save the results
                inner_cv_results.append(best_model)
            #### END INNER LOOP ####

            # get results from all best models in the inner loop
            models = [x["model"] for x in inner_cv_results]
            imputers = [x["imputer"] for x in inner_cv_results]
            best_params = [x["params"] for x in inner_cv_results]
            train_mse = np.mean([x["train_mse"] for x in inner_cv_results])
            val_mse = np.mean([x["val_mse"] for x in inner_cv_results])
            y_pred = []
            for idx, model in enumerate(models):
                imp = imputers[idx]
                X_test_imp = imp.transform(X_test)
                y_pred.append(model.predict(X_test_imp))
            y_pred = np.mean(y_pred, axis=0)
            test_mse = mean_squared_error(y_pred, y_test)

            # save results
            cv_result = CVResult(
                train_val_index=train_val_index,
                test_index=test_index,
                ccle_names=ccle_names,
                y_true=y_test,
                y_pred=y_pred,
                models=models,
                imputers=imputers,
                best_params=best_params,
                train_mse=train_mse,
                val_mse=val_mse,
                test_mse=test_mse,
            )
            self.trainer_result.cv_results.append(cv_result)
            #### END OUTER LOOP ####

    def _compute_cv_feature_importances(self, X: pd.DataFrame):
        """Compute the feature importances from a list of CV results.

        Args:
            X (pd.DataFrame): Feature dataframe
        """
        fold_importance = []
        cv_results = self.trainer_result.cv_results
        for cv_fold in cv_results:
            # get the models
            models = cv_fold.models

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
        importance["pert_name"] = self.pert_name
        importance["pert_mfc_id"] = self.pert_mfc_id
        importance["dose"] = self.dose
        importance["feature_name"] = self.feature_name
        importance["rank"] = importance["importance"].rank(ascending=False).astype(int)
        importance.index.rename("feature", inplace=True)

        self.trainer_result.feature_importances = importance

    def _compute_model_stats(self, X: pd.DataFrame, y: np.array):
        """Compute the model performance from a list of CV results.
        cv_results is the output of _train_nested_cv

        Args:
            X (pd.DataFrame): Feature matrix
            y (np.array): Response vector
        """
        # get the predictions on each test fold
        y_pred_all = []
        y_true_all = []
        for cv_fold in self.trainer_result.cv_results:
            y_pred_all.extend(cv_fold.y_pred)
            y_true_all.extend(cv_fold.y_true)

        # get the performance metrics
        mse = mean_squared_error(y_true_all, y_pred_all)
        mse_se = mse / np.sqrt(len(y_true_all))
        r2 = r2_score(y_true_all, y_pred_all)
        pearson = np.corrcoef(y_true_all, y_pred_all)[0, 1]

        model_stats = ModelStats(
            pert_name=self.trainer_result.pert_name,
            pert_mfc_id=self.trainer_result.pert_mfc_id,
            dose=self.trainer_result.dose,
            feature_name=self.trainer_result.feature_name,
            mse=mse,
            mse_se=mse_se,
            r2=r2,
            pearson=pearson,
        )
        self.trainer_result.model_stats = model_stats


class NestedCVRFTrainer(NestedCVRFTrainerNoRetrain):
    """Perform nested cross validation training for RF model
    Select best hyperparameters and retrain with all data"""

    def _train_nested_cv(self, X, y):
        kf_outer = KFold(
            n_splits=self.config.n_splits,
            shuffle=True,
            random_state=self.config.random_state,
        )
        kf_inner = KFold(
            n_splits=self.config.n_splits - 1,
            shuffle=True,
            random_state=self.config.random_state,
        )
        param_grid = self.config.param_grid
        # edit params only for the model
        param_grid = {f"model__{k}": v for k, v in param_grid.items()}

        for idx, (train_index, test_index) in enumerate(kf_outer.split(X, y)):
            logger.info("Fitting outer fold %d/%d", idx + 1, self.config.n_splits)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # grid search
            model = self._init_model(self.model_class, self.config)
            pipe = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median", keep_empty_features=True),
                    ),
                    ("model", model),
                ]
            )
            clf = GridSearchCV(
                pipe,
                param_grid,
                cv=kf_inner,
                scoring="neg_mean_squared_error",
                n_jobs=self.config.cv_n_jobs,
                verbose=10,
            )
            clf.fit(X_train, y_train.values.ravel())

            # get the best model
            best_params = clf.best_params_
            best_pipeline = clf.best_estimator_
            best_model = best_pipeline.named_steps["model"]
            best_imputer = best_pipeline.named_steps["imputer"]

            # get the train mse
            train_mse = mean_squared_error(
                best_pipeline.predict(X_train), y_train.values.ravel()
            )

            # get the predictions on the test
            y_pred = best_pipeline.predict(X_test)
            test_mse = mean_squared_error(y_pred, y_test.values.ravel())

            cv_result = CVResult(
                train_val_index=train_index,
                test_index=test_index,
                ccle_names=y_test.index.tolist(),
                y_true=y_test.values.ravel(),
                y_pred=y_pred,
                models=[
                    best_model,
                ],
                imputers=[
                    best_imputer,
                ],
                best_params=[
                    best_params,
                ],
                train_mse=train_mse,
                val_mse=train_mse,
                test_mse=test_mse,
            )
            self.trainer_result.cv_results.append(cv_result)


class CVRFTrainer(NestedCVRFTrainer):
    def _train_nested_cv(self, X, y):
        kf_outer = KFold(
            n_splits=self.config.n_splits,
            shuffle=True,
            random_state=self.config.random_state,
        )

        for idx, (train_index, test_index) in enumerate(kf_outer.split(X, y)):
            logger.info("Fitting outer fold %d/%d", idx + 1, self.config.n_splits)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # grid search
            model = self._init_model(self.model_class, self.config)
            pipe = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median", keep_empty_features=True),
                    ),
                    ("model", model),
                ]
            )
            pipe.fit(X_train, y_train.values.ravel())

            # get the best model
            best_params = pipe.get_params()
            best_pipeline = pipe
            best_model = best_pipeline.named_steps["model"]
            best_imputer = best_pipeline.named_steps["imputer"]

            # get the train mse
            train_mse = mean_squared_error(
                best_pipeline.predict(X_train), y_train.values.ravel()
            )

            # get the predictions on the test
            y_pred = best_pipeline.predict(X_test)
            test_mse = mean_squared_error(y_pred, y_test.values.ravel())

            cv_result = CVResult(
                train_val_index=train_index,
                test_index=test_index,
                ccle_names=y_test.index.tolist(),
                y_true=y_test.values.ravel(),
                y_pred=y_pred,
                models=[
                    best_model,
                ],
                imputers=[
                    best_imputer,
                ],
                best_params=[
                    best_params,
                ],
                train_mse=train_mse,
                val_mse=train_mse,
                test_mse=test_mse,
            )
            self.trainer_result.cv_results.append(cv_result)


class WeightedRFRegressor(RandomForestRegressor):
    def __init__(
        self,
        alpha=1,
        max_depth=3,
        n_estimators=100,
        n_jobs=1,
        random_state=42,
        max_features=1.0,
    ):
        super().__init__(
            max_depth=max_depth,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
            max_features=max_features,
        )
        self.alpha = alpha

    def fit(self, X, y):
        sw = (self.alpha * y) ** 2
        return super().fit(X, y, sample_weight=sw)


class WeightedNestedCVRFTrainer(NestedCVRFTrainer):
    def _init_model(self, _, config):
        return super()._init_model(WeightedRFRegressor, config)


class WeightedCVRFTrainer(CVRFTrainer):
    def _init_model(self, _, config):
        return super()._init_model(WeightedRFRegressor, config)


class NestedCVXGBoostTrainer(NestedCVRFTrainer):
    def train(self, *args, **kwargs):
        kwargs["model"] = XGBRegressor
        super().train(*args, **kwargs)

    def _init_model(self, model, config):
        """Initialize the RF model"""
        return model(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )


class CVXGBoostTrainer(CVRFTrainer):
    def _init_model(self, _, config):
        return XGBRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            objective=config.objective,
        )

    def _train_nested_cv(self, X, y):
        y = ((y - y.max()) * -1 + 1) ** 2
        super()._train_nested_cv(X, y)


class NestedCVLGBMTrainer(NestedCVRFTrainer):
    def train(self, *args, **kwargs):
        kwargs["model"] = LGBMRegressor
        super().train(*args, **kwargs)

    def _init_model(self, model, config):
        """Initialize the RF model"""
        return model(n_jobs=config.n_jobs, random_state=config.random_state)

    def _train_nested_cv(self, X, y, model, config):
        # replace JSON special characters in the feature names
        # see: https://github.com/autogluon/autogluon/issues/399
        for char in ["'", '"', ":", ",", "{", "}", "[", "]"]:
            X.columns = X.columns.str.replace(char, "_")
        return super()._train_nested_cv(X, y, model, config)
