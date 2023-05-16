"""Dataclasses to store results of training"""
import os
import pickle
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


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
