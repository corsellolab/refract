from typing import Dict, List

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
