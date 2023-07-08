from typing import Dict, List

from pydantic import BaseModel
import numpy as np


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

### Functions for creating matrices from Datasets
def torch_dataset_to_numpy_array(ds, index_only=False, num_epochs=1):
    """Load from a Dataset into a numpy array.

    Args:
        ds (torch.Dataset): 
        index_only (bool, optional): Only include index variant. 
            Defaults to False.
        num_epochs (int, optional): Number of epochs to run.
    """
    features = []
    labels = []
    ccle_names = []
    groups = []
    for _ in range(num_epochs):
        for ex in ds:
            ccle_name, feat, label = ex
            feat = feat.numpy()
            label = label.numpy()

            if index_only:
                feat = feat[0, :].reshape(1, -1)
                label = label[0].reshape(1, -1)

            features.append(feat)
            labels.append(label)
            groups.append(label.shape)
            ccle_names.append(ccle_name)
    features_array = np.concatenate(features, axis=0)
    labels_array = np.concatenate(labels, axis=0)
    group_array = np.array(groups)

    return features_array, labels_array, group_array, ccle_names