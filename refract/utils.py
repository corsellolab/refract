from typing import Dict, List
from scipy.stats import pearsonr
import pandas as pd
import numpy as np

from pydantic import BaseModel


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            value = self[name]
            if isinstance(value, dict):
                return AttrDict(value)
            else:
                return value
        except KeyError:
            raise AttributeError(f"'{name}' is not a valid attribute")


class RandomForestNestedCVConfig(BaseModel):
    # only use all features
    feature_name_list: List[str] = ["all"]
    # CV Grid search params
    param_grid: Dict[str, List] = {
        "n_estimators": [50],
        "max_depth": [3],
        "max_features": [
            1.0,
        ],
    }
    # CV details
    n_splits: int = 5
    random_state: int = 42
    n_jobs: int = 15
    cv_n_jobs: int = 1


class RandomForestCVConfig(BaseModel):
    # only use all features
    feature_name_list: List[str] = ["all"]
    # CV details
    n_estimators: int = 50
    max_depth: int = 3
    max_features: float = 1.0
    n_splits: int = 5
    random_state: int = 42
    n_jobs: int = 15
    cv_n_jobs: int = 1


class WeightedRandomForestNestedCVConfig(RandomForestNestedCVConfig):
    param_grid: Dict[str, List] = {
        "n_estimators": [50],
        "max_depth": [3],
        "alpha": [
            1.0,
        ],
        "max_features": [
            1.0,
        ],
    }


class WeightedRandomForestCVConfig(RandomForestCVConfig):
    alpha: float = 1.0


class XGBoostCVConfig(RandomForestNestedCVConfig):
    n_jobs: int = 10
    n_estimators = 200
    max_depth = 10
    random_state = 42
    n_jobs = 12
    objective = "rank:pairwise"


class LGBMCVConfig(RandomForestNestedCVConfig):
    param_grid: Dict[str, List] = {
        "num_leaves": [4, 32, 64],
        "max_depth": [2, 4, 8],
    }
    n_jobs: int = 10

def get_top_features(response_df, feature_df, response_col, p):
        df = response_df.merge(feature_df, on="ccle_name")
        df_features = df.loc[:, feature_df.columns]
        series_response = df.loc[:, response_col]

        corrs = []
        for col in df_features.columns:
            c = pearsonr(df_features[col], series_response)[0]
            feature_type = col.split("_")[0]
            corrs.append({"corr": c, "feature_type": feature_type, "feature": col})

        corr_df = pd.DataFrame(corrs)
        top_correlated = corr_df.groupby("feature_type").apply(lambda group: group.nlargest(int(len(group)*p), "corr")).reset_index(level=0, drop=True)

        return top_correlated.feature.tolist()

def dataset_to_group_df(ds, num_epochs):
    features = []
    labels = []
    groups = []
    for i in range(num_epochs):
        for ex in ds:
            ccle_name, feat, label = ex
            # convert to numpy
            feat = feat.numpy()
            label = label.numpy()
            features.append(feat)
            labels.append(label)
            groups.append(label.shape)
    group_train_features = np.concatenate(features, axis=0)
    group_train_labels = np.concatenate(labels, axis=0)
    groups = np.array(groups)

    return group_train_features, group_train_labels, groups


def dataset_to_individual_df(ds):
    features = []
    labels = []
    ccle_names = []
    for ex in ds:
        ccle_name, feat, label = ex
        feat = feat.numpy()
        label = label.numpy()
        features.append(feat[0, :].reshape(1, -1))
        labels.append(label[0])
        ccle_names.append(ccle_name)
    individual_train_features = np.concatenate(features, axis=0)
    individual_train_labels = np.array(labels)
    return individual_train_features, individual_train_labels, ccle_names