from pydantic import BaseModel
from typing import List, Dict


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


class RandomForestCVConfig(BaseModel):
    # only use all features
    feature_name_list: List[str] = ["all"]
    # CV Grid search params
    param_grid: Dict[str, List] = {
        "n_estimators": [100],
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


class WeightedRandomForestCVConfig(RandomForestCVConfig):
    param_grid: Dict[str, List] = {
        "n_estimators": [100],
        "max_depth": [3],
        "alpha": [0.1, 1, 10],
        "max_features": [
            1.0,
        ],
    }


class XGBoostCVConfig(RandomForestCVConfig):
    n_jobs: int = 10
    param_grid: Dict[str, List] = {
        "n_estimators": [5, 10, 25, 50],
        "max_depth": [2, 3, 5, 10],
    }


class LGBMCVConfig(RandomForestCVConfig):
    param_grid: Dict[str, List] = {
        "num_leaves": [4, 32, 64],
        "max_depth": [2, 4, 8],
    }
    n_jobs: int = 10
