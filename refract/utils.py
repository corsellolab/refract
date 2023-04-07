from pydantic import BaseModel
from typing import List, Union, Literal


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


# pydantic model for configuration
class RandomForestConfig(BaseModel):
    # RF params
    n_estimators: int = 100
    max_depth: Union[int, None] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    bootstrap: bool = True
    oob_score: bool = False
    n_jobs: int = 8
    random_state: int = 42
    # train config
    feature_name_list: List[str] = ["X_all", "X_ccle"]
