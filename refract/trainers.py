from sklearn.ensemble import RandomForestRegressor
import logging
import pandas as pd
import os
import numpy as np
import json
import pickle
from typing import List, Dict
from .utils import RandomForestConfig, RandomForestCVConfig, XGBoostCVConfig
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from dataclasses import dataclass, field, asdict
from .datasets import ResponseSet, FeatureSet

logger = logging.getLogger(__name__)


@dataclass
class ModelStats:
    """Statistics for a model"""

    pert_name: str
    pert_mfc_id: str
    dose: str
    feature_name: str
    mse: float
    mse_se: float
    r2: float
    pearson: float


@dataclass
class GridSearchResult:
    """Result of HP grid search"""

    train_index: np.array
    val_index: np.array
    params: Dict[str, float]
    train_mse: float
    val_mse: float
    model: object
    imputer: object


@dataclass
class CVResult:
    """CV result for a single fold"""

    train_val_index: np.array
    test_index: np.array
    ccle_names: List[str]
    y_true: np.array
    y_pred: np.array
    models: List[object]
    imputers: List[object]
    best_params: List[Dict[str, float]]
    train_mse: float
    val_mse: float
    test_mse: float


@dataclass
class TrainerResult:
    """Result of nested CV"""

    pert_name: str
    pert_mfc_id: str
    dose: str
    feature_name: str
    output_dir: str
    cv_results: List[CVResult] = field(default_factory=list)
    model_stats: ModelStats = None
    feature_importances: pd.DataFrame = None

    def save_model_stats(self):
        """Save to model stats to file"""
        output_path = os.path.join(
            self.output_dir,
            f"{self.pert_name}_{self.pert_mfc_id}_{self.dose}_{self.feature_name}_model_stats.csv",
        )
        df = pd.DataFrame([asdict(self.model_stats)])
        df.to_csv(output_path, index=False)

    def save_predictions(self):
        """Save predictions to a file"""
        output_path = os.path.join(
            self.output_dir,
            f"{self.pert_name}_{self.pert_mfc_id}_{self.dose}_{self.feature_name}_predictions.csv",
        )
        ccle_names = np.concatenate(
            [cv_result.ccle_names for cv_result in self.cv_results]
        )
        y_true = np.concatenate([cv_result.y_true for cv_result in self.cv_results])
        y_pred = np.concatenate([cv_result.y_pred for cv_result in self.cv_results])

        df = pd.DataFrame({"ccle_name": ccle_names, "y_true": y_true, "y_pred": y_pred})
        df.to_csv(output_path, index=False)

    def save_feature_importances(self):
        """Save feature importances"""
        output_path = os.path.join(
            self.output_dir,
            f"{self.pert_name}_{self.pert_mfc_id}_{self.dose}_{self.feature_name}_feature_importances.csv",
        )
        self.feature_importances.to_csv(output_path, index=False)

    def save_trainer_result(self):
        """Save trainer result to a file as pkl"""
        output_path = os.path.join(
            self.output_dir,
            f"{self.pert_name}_{self.pert_mfc_id}_{self.dose}_{self.feature_name}_trainer_result.pkl",
        )
        with open(output_path, "wb") as f:
            pickle.dump(self, f)


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
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            bootstrap=config.bootstrap,
            oob_score=config.oob_score,
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


class NestedCVXGBoostTrainer(NestedCVRFTrainerNoRetrain):
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


class NestedCVLGBMTrainer(NestedCVRFTrainerNoRetrain):
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
