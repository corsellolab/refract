import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import shap
from shap import TreeExplainer, summary_plot, LinearExplainer
import xgboost as xgb
import pickle
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import QuantileRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .utils import (
    load_feature_df, load_response_df, intersect_depmap_ids,
    load_split, load_selected_features, evaluate_predictions
)
from .visualize import save_shap_summary_plot, save_prediction_scatter

# Global variable to track if PyTorch thread limits have been set
_pytorch_threads_initialized = False

def set_pytorch_threads(n_threads: int):
    """
    Set PyTorch thread limits globally. This should be called once per process.
    
    Parameters:
    n_threads (int): Number of threads to use for PyTorch operations
    """
    global _pytorch_threads_initialized
    
    if _pytorch_threads_initialized:
        # Only set intra-op threads if already initialized
        torch.set_num_threads(n_threads)
        return f"PyTorch intra-op threads updated to: {n_threads}"
    
    try:
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(n_threads)
        _pytorch_threads_initialized = True
        return f"PyTorch thread limits set to: {n_threads} threads"
    except RuntimeError as e:
        if "cannot set number of interop threads" in str(e):
            # Interop threads already set, just set intra-op threads
            torch.set_num_threads(n_threads)
            _pytorch_threads_initialized = True
            return f"PyTorch intra-op threads set to: {n_threads} (interop threads already configured)"
        else:
            raise e

class BaseTrainer(ABC):
    """
    Base trainer class that defines the interface for all model trainers.
    
    This class provides a common structure for training machine learning models
    with cross-validation, feature importance analysis, and prediction capabilities.
    """
    
    def __init__(self, output_dir: str, n_threads: int = 8):
        """
        Initialize the base trainer.
        
        Parameters:
        output_dir (str): Directory to save results and models.
        n_threads (int): Number of threads to use for training.
        """
        self.output_dir = output_dir
        self.n_threads = n_threads
        self.models = []  # Store trained models for each fold
        self.all_test_X = [] # Store test data for each fold
        self.fold_results = []  # Store performance metrics for each fold
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    @abstractmethod
    def train_single_model(self, X_train, y_train, X_val, y_val, **kwargs):
        """
        Train a single model on the given training data.
        
        Parameters:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        **kwargs: Additional model-specific parameters
        
        Returns:
        Trained model object
        """
        pass
    
    @abstractmethod
    def predict(self, model, X):
        """
        Make predictions using a trained model.
        
        Parameters:
        model: Trained model object
        X: Features to make predictions on
        
        Returns:
        Array of predictions
        """
        pass

    def train_cross_validation(
        self, 
        feature_file: str, 
        response_file: str, 
        split_dir: str,
        n_splits: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train models using cross-validation.
        
        Parameters:
        feature_file (str): Path to feature file
        response_file (str): Path to response file
        split_dir (str): Directory containing split files
        n_splits (int): Number of cross-validation splits
        **kwargs: Additional model-specific parameters
        
        Returns:
        Dictionary containing training results and metrics
        """
        # Load and preprocess data
        feature_df = load_feature_df(feature_file)
        response_df = load_response_df(response_file)
        response_df, feature_df = intersect_depmap_ids(response_df, feature_df)

        # get the drug name from the response df
        drug_name = response_df["name"].values[0]
        # get the response name from the response file
        response_name = response_df["broad_id"].values[0]
        
        # Store all predictions for overall metrics
        all_val_true = []
        all_val_pred = []
        all_test_true = []
        all_test_pred = []
        all_test_ids = []
        
        # Train models for each fold
        for fold in range(n_splits):
            print(f"\n{'='*60}")
            print(f"PROCESSING FOLD {fold+1}/{n_splits} for {drug_name} ({response_name})")
            print(f"{'='*60}")
            
            # Load split assignments and selected features
            split_file = os.path.join(split_dir, f'{fold}.split.txt')
            features_file = os.path.join(split_dir, f'{fold}.features.csv')
            
            train_ids, val_ids, test_ids = load_split(split_file)
            selected_features = load_selected_features(features_file)
            
            print(f"Loaded {len(selected_features)} selected features")
            print(f"Data split: {len(train_ids)} train, {len(val_ids)} validation, {len(test_ids)} test samples")
            
            # Select relevant features and samples
            X = feature_df[selected_features]
            y = response_df['LFC']
            
            # Split data according to fold assignments
            X_train = X.loc[train_ids]
            y_train = y.loc[train_ids]
            X_val = X.loc[val_ids]
            y_val = y.loc[val_ids]
            X_test = X.loc[test_ids]
            y_test = y.loc[test_ids]
            
            # Train model
            print(f"\nStarting model training for fold {fold+1}...")
            model = self.train_single_model(X_train, y_train, X_val, y_val, **kwargs)
            self.models.append(model)
            self.all_test_X.append(X_test)
            print(f"Model training completed for fold {fold+1}")
            
            # Make predictions
            print(f"Making predictions for fold {fold+1}...")
            val_preds = self.predict(model, X_val)
            test_preds = self.predict(model, X_test)
            
            # Evaluate performance
            print(f"\nFold {fold+1} Results:")
            val_metrics = evaluate_predictions(y_val, val_preds, "Validation")
            test_metrics = evaluate_predictions(y_test, test_preds, "Test")
            
            # Store predictions for overall metrics
            all_val_true.extend(y_val)
            all_val_pred.extend(val_preds)
            all_test_true.extend(y_test)
            all_test_pred.extend(test_preds)
            all_test_ids.extend(test_ids)
            
            # Store results
            self.fold_results.append({
                'fold': fold,
                'val_rmse': val_metrics['rmse'],
                'val_r2': val_metrics['r2'],
                'val_pearson': val_metrics['pearson'],
                'test_rmse': test_metrics['rmse'],
                'test_r2': test_metrics['r2'],
                'test_pearson': test_metrics['pearson'],
                'drug_name': drug_name,
                'response_name': response_name
            })
        
        # Calculate overall metrics
        print("\nOverall Results (across all folds):")
        overall_val_metrics = evaluate_predictions(
            np.array(all_val_true), 
            np.array(all_val_pred), 
            "Overall Validation"
        )
        overall_test_metrics = evaluate_predictions(
            np.array(all_test_true), 
            np.array(all_test_pred), 
            "Overall Test"
        )

        # Add overall results to the fold_results
        overall_results = {
            'fold': 'overall',
            'val_rmse': overall_val_metrics['rmse'],
            'val_r2': overall_val_metrics['r2'],
            'val_pearson': overall_val_metrics['pearson'],
            'test_rmse': overall_test_metrics['rmse'],
            'test_r2': overall_test_metrics['r2'],
            'test_pearson': overall_test_metrics['pearson'],
            'drug_name': drug_name,
            'response_name': response_name
        }
        self.fold_results.append(overall_results)
        
        # Save and print summary
        self.save_summary_results()
        self.print_summary_statistics()
        self.save_all_test_predictions(all_test_ids, all_test_true, all_test_pred)
        self.save_trainer()
        save_prediction_scatter(all_test_pred, all_test_true, os.path.join(self.output_dir, 'predictions_scatter.png'))

        # save response file and path to feature file
        # save response file to output directory
        response_df = load_response_df(response_file)
        response_df.to_csv(os.path.join(self.output_dir, 'response_file.csv'), index=False)

        # save path to the feature file
        with open(os.path.join(self.output_dir, 'path_to_feature_file.txt'), 'w') as f:
            f.write(feature_file)
        
        return {
            'models': self.models,
            'fold_results': self.fold_results,
            'overall_val_metrics': overall_val_metrics,
            'overall_test_metrics': overall_test_metrics,
            'all_test_ids': all_test_ids,
            'all_test_true': all_test_true,
            'all_test_pred': all_test_pred
        }
    
    def save_summary_results(self):
        """Save summary statistics to CSV file."""
        results_df = pd.DataFrame(self.fold_results)
        results_df.to_csv(os.path.join(self.output_dir, 'model_performance_summary.csv'), index=False)

    def save_all_test_predictions(self, all_test_ids, all_test_true, all_test_pred):
        """Save all test predictions to CSV file."""
        results_df = pd.DataFrame({
            'id': all_test_ids,
            'true': all_test_true,
            'pred': all_test_pred
        })
        results_df.to_csv(os.path.join(self.output_dir, 'all_test_predictions.csv'), index=False)
    
    def print_summary_statistics(self):
        """Print summary statistics across all folds."""
        results_df = pd.DataFrame(self.fold_results[:-1])  # Exclude overall results for mean/std calculation
        
        print("\nPer-fold Summary Statistics:")
        print("\nValidation Metrics:")
        print(f"Mean RMSE: {results_df['val_rmse'].mean():.4f} ± {results_df['val_rmse'].std():.4f}")
        print(f"Mean R2: {results_df['val_r2'].mean():.4f} ± {results_df['val_r2'].std():.4f}")
        print(f"Mean Pearson: {results_df['val_pearson'].mean():.4f} ± {results_df['val_pearson'].std():.4f}")
        print("\nTest Metrics:")
        print(f"Mean RMSE: {results_df['test_rmse'].mean():.4f} ± {results_df['test_rmse'].std():.4f}")
        print(f"Mean R2: {results_df['test_r2'].mean():.4f} ± {results_df['test_r2'].std():.4f}")
        print(f"Mean Pearson: {results_df['test_pearson'].mean():.4f} ± {results_df['test_pearson'].std():.4f}")

    def save_models(self):
        """Save as pkl file to subdir models"""
        os.makedirs(os.path.join(self.output_dir, 'models'), exist_ok=True)
        for i, model in enumerate(self.models):
            with open(os.path.join(self.output_dir, 'models', f'model_{i}.pkl'), 'wb') as f:
                pickle.dump(model, f)

    @abstractmethod
    def compute_feature_importance(self):
        """
        Compute feature importance scores for the model.
        
        Returns:
        Dictionary containing importance results
        """
        pass

    @abstractmethod
    def save_feature_importance(self):
        """
        Save feature importance results for a specific fold.
        """
        pass

    def save_trainer(self):
        """Save the trainer object to a pkl file"""
        with open(os.path.join(self.output_dir, 'trainer.pkl'), 'wb') as f:
            pickle.dump(self, f)

class XGBoostTrainer(BaseTrainer):
    """
    XGBoost trainer that implements the BaseTrainer interface.
    
    This class provides XGBoost-specific implementations for training,
    prediction, and feature importance analysis using SHAP values.
    """
    
    def __init__(
        self, 
        output_dir: str, 
        n_threads: int = 8,
        num_rounds: int = 1000,
        early_stopping_rounds: int = 50,
        **xgb_params
    ):
        """
        Initialize the XGBoost trainer.
        
        Parameters:
        output_dir (str): Directory to save results and models.
        n_threads (int): Number of threads to use for training.
        num_rounds (int): Maximum number of training rounds.
        early_stopping_rounds (int): Number of rounds without improvement to trigger early stopping.
        **xgb_params: Additional XGBoost parameters to override defaults.
        """
        super().__init__(output_dir, n_threads)
        self.num_rounds = num_rounds
        self.early_stopping_rounds = early_stopping_rounds
        
        # Default XGBoost parameters
        self.default_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': 0.01,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'lambda': 1.0,
            'alpha': 0.1,
            'tree_method': 'hist',
            'nthread': self.n_threads
        }
        
        # Update with any provided parameters
        self.default_params.update(xgb_params)
    
    def train_single_model(self, X_train, y_train, X_val, y_val, **kwargs):
        """
        Train a single XGBoost model with early stopping and outlier emphasis.
        
        Parameters:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        **kwargs: Additional parameters (currently unused)
        
        Returns:
        xgb.Booster: The trained XGBoost model
        """
        # Compute mean_label once
        mean_label = np.mean(y_train)
        
        # Compute weights for outlier emphasis
        weights = 1 + np.abs(y_train - mean_label)
        
        # Convert data to XGBoost DMatrix format with precomputed weights
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Define the custom loss function
        def weighted_mse_with_outlier_emphasis(preds, dtrain):
            labels = dtrain.get_label()
            errors = preds - labels
            
            # Retrieve precomputed weights
            weights = dtrain.get_weight()
            
            grad = weights * errors  # Weighted gradient
            hess = weights           # Weighted Hessian
            
            return grad, hess
        
        watchlist = [(dtrain, "train"), (dval, "eval")]
        
        model = xgb.train(
            params=self.default_params,
            dtrain=dtrain,
            num_boost_round=self.num_rounds,
            evals=watchlist,
            early_stopping_rounds=self.early_stopping_rounds,
            obj=weighted_mse_with_outlier_emphasis,
            verbose_eval=True
        )
        
        print(f"Best iteration: {model.best_iteration}")
        return model
    
    def predict(self, model, X):
        """
        Make predictions using a trained XGBoost model.
        
        Parameters:
        model: Trained XGBoost model
        X: Features to make predictions on
        
        Returns:
        Array of predictions
        """
        dmatrix = xgb.DMatrix(X)
        return model.predict(dmatrix)
    
    def _compute_top_features_by_shap(self, shap_values, feature_names):
        # fill nans with 0
        shap_values = shap_values.fillna(0)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": mean_abs_shap
        })
        feature_importance = feature_importance.sort_values(
            "importance", ascending=False
        )
        return feature_importance

    def _compute_cv_shap_importance(
        self,
        models: list,  # List of 10 trained models from CV
        X_test_folds: list,  # List of test sets from each fold
    ) -> Dict[str, Any]:
        """
        Compute aggregated SHAP feature importance across CV folds.
        
        Args:
            models: List of trained XGBoost models from each CV fold
            X_test_folds: List of test DataFrames from each fold
            
        Returns:
            Dictionary with aggregated SHAP results
        """
        all_shap_values = []
        all_test_data = []
        
        # Compute SHAP values for each fold
        for i, (model, X_test) in enumerate(zip(models, X_test_folds)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            shap_value_df = pd.DataFrame(shap_values, columns=X_test.columns)

            all_shap_values.append(shap_value_df)
            all_test_data.append(X_test)
        
        # Concatenate all SHAP values and test data
        combined_shap_values = pd.concat(all_shap_values, axis=0)
        combined_test_data = pd.concat(all_test_data, axis=0)
        
        # Compute feature importance using aggregated SHAP values
        feature_importance = self._compute_top_features_by_shap(
            combined_shap_values, 
            combined_test_data.columns.tolist()
        )
        
        return {
            'combined_shap_values': combined_shap_values,
            'combined_test_data': combined_test_data,
            'feature_importance': feature_importance,
            'feature_names': combined_test_data.columns.tolist()
        }
    
    def compute_feature_importance(self):
        """
        Compute SHAP-based feature importance for XGBoost model.
            
        Returns:
        Dictionary containing SHAP results and top features
        """
        feature_importance = self._compute_cv_shap_importance(
            self.models,
            self.all_test_X
        )
        return feature_importance
        

    def save_feature_importance(self):
        """
        Save SHAP analysis results to disk. To subdir feature_importance
        
        Parameters:
        fold (int): Fold number
        importance_results (Dict): Results from compute_feature_importance
        """
        # get and save the feature importance results
        feature_importance = self.compute_feature_importance()
        shap_values = feature_importance['combined_shap_values']
        test_data = feature_importance['combined_test_data']
        feature_importance = feature_importance['feature_importance']

        # save to output directory
        if not os.path.exists(os.path.join(self.output_dir, 'feature_importance')):
            os.makedirs(os.path.join(self.output_dir, 'feature_importance'))
        feature_importance.to_csv(os.path.join(self.output_dir, 'feature_importance', 'feature_importance.csv'), index=False)
        shap_values.to_csv(os.path.join(self.output_dir, 'feature_importance', 'shap_values.csv'), index=False)
        test_data.to_csv(os.path.join(self.output_dir, 'feature_importance', 'test_data.csv'), index=False)

        # save shap summary plot
        save_shap_summary_plot(shap_values, test_data, os.path.join(self.output_dir, 'feature_importance', 'shap_summary.png'))

class LinearTrainer(BaseTrainer):
    """
    Linear regression trainer with L1 regularization that implements the BaseTrainer interface.
    
    This class provides Lasso regression implementations for training,
    prediction, and feature importance analysis using SHAP values.
    """
    
    def __init__(
        self, 
        output_dir: str, 
        n_threads: int = 8,
        alphas: List[float] = None,
        max_iter: int = 10000,
        **lasso_params
    ):
        """
        Initialize the Linear trainer.
        
        Parameters:
        output_dir (str): Directory to save results and models.
        n_threads (int): Number of threads to use for training (not used for Lasso).
        alphas (List[float]): List of alpha values to try for regularization.
        max_iter (int): Maximum number of iterations for convergence.
        **lasso_params: Additional Lasso parameters to override defaults.
        """
        super().__init__(output_dir, n_threads)
        
        # Default alpha values to search over
        if alphas is None:
            self.alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        else:
            self.alphas = alphas
            
        self.max_iter = max_iter
        
        # Default Lasso parameters
        self.default_params = {
            'max_iter': self.max_iter,
            'random_state': 42,
            'selection': 'random'
        }
        
        # Update with any provided parameters
        self.default_params.update(lasso_params)
        
        # Store scalers for each fold
        self.scalers = []
    
    def train_single_model(self, X_train, y_train, X_val, y_val, **kwargs):
        """
        Train a single Lasso model with hyperparameter optimization using validation set.
        
        Parameters:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        **kwargs: Additional parameters (currently unused)
        
        Returns:
        dict: Dictionary containing the trained model, scaler, and imputer
        """
        # Handle missing values with median imputation
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_val_imputed = imputer.transform(X_val)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_val_scaled = scaler.transform(X_val_imputed)
        
        best_alpha = None
        best_score = float('inf')
        best_model = None
        
        # Try different alpha values and select best based on validation performance
        for alpha in self.alphas:
            model = Lasso(alpha=alpha, **self.default_params)
            model.fit(X_train_scaled, y_train)
            
            val_pred = model.predict(X_val_scaled)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            
            if val_rmse < best_score:
                best_score = val_rmse
                best_alpha = alpha
                best_model = model
        
        print(f"Best alpha: {best_alpha}, Validation RMSE: {best_score:.4f}")
        
        # Store the scaler for this fold
        self.scalers.append(scaler)
        
        return {
            'model': best_model,
            'scaler': scaler,
            'imputer': imputer,
            'best_alpha': best_alpha,
            'best_score': best_score
        }
    
    def predict(self, model_dict, X):
        """
        Make predictions using a trained Lasso model.
        
        Parameters:
        model_dict: Dictionary containing trained model, scaler, and imputer
        X: Features to make predictions on
        
        Returns:
        Array of predictions
        """
        model = model_dict['model']
        scaler = model_dict['scaler']
        imputer = model_dict['imputer']
        
        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed)
        return model.predict(X_scaled)
    
    def _compute_top_features_by_shap(self, shap_values, feature_names):
        # fill nans with 0
        shap_values = shap_values.fillna(0)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": mean_abs_shap
        })
        feature_importance = feature_importance.sort_values(
            "importance", ascending=False
        )
        return feature_importance

    def _compute_cv_shap_importance(
        self,
        models: list,  # List of 10 trained model dicts from CV
        X_test_folds: list,  # List of test sets from each fold
    ) -> Dict[str, Any]:
        """
        Compute aggregated SHAP feature importance across CV folds.
        
        Args:
            models: List of trained model dictionaries from each CV fold
            X_test_folds: List of test DataFrames from each fold
            
        Returns:
            Dictionary with aggregated SHAP results
        """
        all_shap_values = []
        all_test_data = []
        
        # Compute SHAP values for each fold
        for i, (model_dict, X_test) in enumerate(zip(models, X_test_folds)):
            model = model_dict['model']
            scaler = model_dict['scaler']
            imputer = model_dict['imputer']
            
            # Handle missing values with median imputation
            X_test_imputed = imputer.transform(X_test)
            X_test_scaled = scaler.transform(X_test_imputed)
            
            # Create SHAP explainer for linear model
            explainer = shap.LinearExplainer(model, X_test_scaled)
            shap_values = explainer.shap_values(X_test_scaled)
            
            shap_value_df = pd.DataFrame(shap_values, columns=X_test.columns)

            all_shap_values.append(shap_value_df)
            all_test_data.append(X_test)
        
        # Concatenate all SHAP values and test data
        combined_shap_values = pd.concat(all_shap_values, axis=0)
        combined_test_data = pd.concat(all_test_data, axis=0)
        
        # Compute feature importance using aggregated SHAP values
        feature_importance = self._compute_top_features_by_shap(
            combined_shap_values, 
            combined_test_data.columns.tolist()
        )
        
        return {
            'combined_shap_values': combined_shap_values,
            'combined_test_data': combined_test_data,
            'feature_importance': feature_importance,
            'feature_names': combined_test_data.columns.tolist()
        }
    
    def compute_feature_importance(self):
        """
        Compute SHAP-based feature importance for Linear model.
            
        Returns:
        Dictionary containing SHAP results and top features
        """
        feature_importance = self._compute_cv_shap_importance(
            self.models,
            self.all_test_X
        )
        return feature_importance
        

    def save_feature_importance(self):
        """
        Save SHAP analysis results to disk. To subdir feature_importance
        """
        # get and save the feature importance results
        feature_importance = self.compute_feature_importance()
        shap_values = feature_importance['combined_shap_values']
        test_data = feature_importance['combined_test_data']
        feature_importance = feature_importance['feature_importance']

        # save to output directory
        if not os.path.exists(os.path.join(self.output_dir, 'feature_importance')):
            os.makedirs(os.path.join(self.output_dir, 'feature_importance'))
        feature_importance.to_csv(os.path.join(self.output_dir, 'feature_importance', 'feature_importance.csv'), index=False)
        shap_values.to_csv(os.path.join(self.output_dir, 'feature_importance', 'shap_values.csv'), index=False)
        test_data.to_csv(os.path.join(self.output_dir, 'feature_importance', 'test_data.csv'), index=False)

        # save shap summary plot
        save_shap_summary_plot(shap_values, test_data, os.path.join(self.output_dir, 'feature_importance', 'shap_summary.png'))

    def save_models(self):
        """Save models and scalers as pkl files to subdir models"""
        os.makedirs(os.path.join(self.output_dir, 'models'), exist_ok=True)
        for i, model_dict in enumerate(self.models):
            with open(os.path.join(self.output_dir, 'models', f'model_{i}.pkl'), 'wb') as f:
                pickle.dump(model_dict, f)

class RandomForestTrainer(BaseTrainer):
    """
    Random Forest trainer that implements the BaseTrainer interface.
    
    This class provides Random Forest implementations for training,
    prediction, and feature importance analysis using SHAP values.
    """
    
    def __init__(
        self, 
        output_dir: str, 
        n_threads: int = 8,
        param_grid: Dict[str, List] = None,
        **rf_params
    ):
        """
        Initialize the Random Forest trainer.
        
        Parameters:
        output_dir (str): Directory to save results and models.
        n_threads (int): Number of threads to use for training.
        param_grid (Dict[str, List]): Parameter grid for grid search.
        **rf_params: Additional Random Forest parameters to override defaults.
        """
        super().__init__(output_dir, n_threads)
        
        # Default parameter grid for grid search
        if param_grid is None:
            self.param_grid = {
                'n_estimators': [100, 200, 500],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        else:
            self.param_grid = param_grid
        
        # Default Random Forest parameters
        self.default_params = {
            'random_state': 42,
            'n_jobs': self.n_threads,
            'bootstrap': True
        }
        
        # Update with any provided parameters
        self.default_params.update(rf_params)
    
    def train_single_model(self, X_train, y_train, X_val, y_val, **kwargs):
        """
        Train a single Random Forest model with grid search optimization using validation set.
        
        Parameters:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        **kwargs: Additional parameters (currently unused)
        
        Returns:
        dict: Dictionary containing the trained model and best parameters
        """
        # Handle missing values with median imputation
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_val_imputed = imputer.transform(X_val)
        
        best_params = None
        best_score = float('inf')
        best_model = None
        
        # Generate all parameter combinations
        from itertools import product
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        print(f"Testing {len(list(product(*param_values)))} parameter combinations...")
        
        # Grid search over parameters
        for i, param_combination in enumerate(product(*param_values)):
            current_params = dict(zip(param_names, param_combination))
            current_params.update(self.default_params)
            
            model = RandomForestRegressor(**current_params)
            model.fit(X_train_imputed, y_train)
            
            val_pred = model.predict(X_val_imputed)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            
            if val_rmse < best_score:
                best_score = val_rmse
                best_params = current_params.copy()
                best_model = model
                
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1} parameter combinations, best RMSE so far: {best_score:.4f}")
        
        print(f"Best parameters: {best_params}")
        print(f"Best validation RMSE: {best_score:.4f}")
        
        return {
            'model': best_model,
            'imputer': imputer,
            'best_params': best_params,
            'best_score': best_score
        }
    
    def predict(self, model_dict, X):
        """
        Make predictions using a trained Random Forest model.
        
        Parameters:
        model_dict: Dictionary containing trained model and imputer
        X: Features to make predictions on
        
        Returns:
        Array of predictions
        """
        model = model_dict['model']
        imputer = model_dict['imputer']
        
        X_imputed = imputer.transform(X)
        return model.predict(X_imputed)
    
    def _compute_top_features_by_shap(self, shap_values, feature_names):
        # fill nans with 0
        shap_values = shap_values.fillna(0)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": mean_abs_shap
        })
        feature_importance = feature_importance.sort_values(
            "importance", ascending=False
        )
        return feature_importance

    def _compute_cv_shap_importance(
        self,
        models: list,  # List of 10 trained model dicts from CV
        X_test_folds: list,  # List of test sets from each fold
    ) -> Dict[str, Any]:
        """
        Compute aggregated SHAP feature importance across CV folds.
        
        Args:
            models: List of trained model dictionaries from each CV fold
            X_test_folds: List of test DataFrames from each fold
            
        Returns:
            Dictionary with aggregated SHAP results
        """
        all_shap_values = []
        all_test_data = []
        
        # Compute SHAP values for each fold
        for i, (model_dict, X_test) in enumerate(zip(models, X_test_folds)):
            model = model_dict['model']
            imputer = model_dict['imputer']
            
            # Handle missing values with median imputation
            X_test_imputed = imputer.transform(X_test)
            
            # Create SHAP explainer for tree model
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_imputed)
            
            shap_value_df = pd.DataFrame(shap_values, columns=X_test.columns)

            all_shap_values.append(shap_value_df)
            all_test_data.append(X_test)
        
        # Concatenate all SHAP values and test data
        combined_shap_values = pd.concat(all_shap_values, axis=0)
        combined_test_data = pd.concat(all_test_data, axis=0)
        
        # Compute feature importance using aggregated SHAP values
        feature_importance = self._compute_top_features_by_shap(
            combined_shap_values, 
            combined_test_data.columns.tolist()
        )
        
        return {
            'combined_shap_values': combined_shap_values,
            'combined_test_data': combined_test_data,
            'feature_importance': feature_importance,
            'feature_names': combined_test_data.columns.tolist()
        }
    
    def compute_feature_importance(self):
        """
        Compute SHAP-based feature importance for Random Forest model.
            
        Returns:
        Dictionary containing SHAP results and top features
        """
        feature_importance = self._compute_cv_shap_importance(
            self.models,
            self.all_test_X
        )
        return feature_importance
        

    def save_feature_importance(self):
        """
        Save SHAP analysis results to disk. To subdir feature_importance
        """
        # get and save the feature importance results
        feature_importance = self.compute_feature_importance()
        shap_values = feature_importance['combined_shap_values']
        test_data = feature_importance['combined_test_data']
        feature_importance = feature_importance['feature_importance']

        # save to output directory
        if not os.path.exists(os.path.join(self.output_dir, 'feature_importance')):
            os.makedirs(os.path.join(self.output_dir, 'feature_importance'))
        feature_importance.to_csv(os.path.join(self.output_dir, 'feature_importance', 'feature_importance.csv'), index=False)
        shap_values.to_csv(os.path.join(self.output_dir, 'feature_importance', 'shap_values.csv'), index=False)
        test_data.to_csv(os.path.join(self.output_dir, 'feature_importance', 'test_data.csv'), index=False)

        # save shap summary plot
        save_shap_summary_plot(shap_values, test_data, os.path.join(self.output_dir, 'feature_importance', 'shap_summary.png'))

    def save_models(self):
        """Save models and imputers as pkl files to subdir models"""
        os.makedirs(os.path.join(self.output_dir, 'models'), exist_ok=True)
        for i, model_dict in enumerate(self.models):
            with open(os.path.join(self.output_dir, 'models', f'model_{i}.pkl'), 'wb') as f:
                pickle.dump(model_dict, f)

class QuantileRegressionTrainer(BaseTrainer):
    """
    Quantile Regression trainer that implements the BaseTrainer interface.
    
    This class provides Quantile Regression implementations for training,
    prediction, and feature importance analysis using SHAP values.
    Focuses on the 0.1 quantile (10th percentile) using pinball loss.
    """
    
    def __init__(
        self, 
        output_dir: str, 
        n_threads: int = 8,
        quantile: float = 0.1,
        alphas: List[float] = None,
        solvers: List[str] = None,
        **qr_params
    ):
        """
        Initialize the Quantile Regression trainer.
        
        Parameters:
        output_dir (str): Directory to save results and models.
        n_threads (int): Number of threads to use for training.
        quantile (float): Quantile to estimate (default: 0.1 for bottom 10%).
        alphas (List[float]): List of alpha values for regularization.
        solvers (List[str]): List of solvers to try.
        **qr_params: Additional QuantileRegressor parameters to override defaults.
        """
        super().__init__(output_dir, n_threads)
        
        self.quantile = quantile
        
        # Default alpha values for regularization
        if alphas is None:
            self.alphas = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
        else:
            self.alphas = alphas
            
        # Default solvers to try
        if solvers is None:
            self.solvers = ['highs', 'interior-point']  # Most robust solvers
        else:
            self.solvers = solvers
        
        # Default QuantileRegressor parameters
        self.default_params = {
            'quantile': self.quantile,
            'fit_intercept': True,
            'solver_options': {'presolve': True}
        }
        
        # Update with any provided parameters
        self.default_params.update(qr_params)
        
    def train_single_model(self, X_train, y_train, X_val, y_val, **kwargs):
        """
        Train a single Quantile Regression model with hyperparameter optimization using validation set.
        
        Parameters:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        **kwargs: Additional parameters (currently unused)
        
        Returns:
        dict: Dictionary containing the trained model, scaler, imputer, and best parameters
        """
        # Handle missing values with median imputation
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_val_imputed = imputer.transform(X_val)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_val_scaled = scaler.transform(X_val_imputed)
        
        best_params = None
        best_score = float('inf')
        best_model = None
        
        print(f"Testing {len(self.alphas) * len(self.solvers)} parameter combinations for quantile {self.quantile}...")
        
        # Grid search over alpha and solver
        for alpha in self.alphas:
            for solver in self.solvers:
                try:
                    current_params = self.default_params.copy()
                    current_params['alpha'] = alpha
                    current_params['solver'] = solver
                    
                    model = QuantileRegressor(**current_params)
                    model.fit(X_train_scaled, y_train)
                    
                    val_pred = model.predict(X_val_scaled)
                    # compute standard loss
                    val_loss = mean_squared_error(y_val, val_pred)
                    
                    if val_loss < best_score:
                        best_score = val_loss
                        best_params = current_params.copy()
                        best_model = model
                        
                    print(f"Alpha: {alpha}, Solver: {solver}, Pinball Loss: {val_loss:.4f}")
                    
                except Exception as e:
                    print(f"Failed with alpha={alpha}, solver={solver}: {str(e)}")
                    continue
        
        if best_model is None:
            raise RuntimeError("No valid model found. All parameter combinations failed.")
        
        print(f"Best parameters: {best_params}")
        print(f"Best validation pinball loss: {best_score:.4f}")
        
        return {
            'model': best_model,
            'scaler': scaler,
            'imputer': imputer,
            'best_params': best_params,
            'best_score': best_score
        }
    
    def predict(self, model_dict, X):
        """
        Make predictions using a trained Quantile Regression model.
        
        Parameters:
        model_dict: Dictionary containing trained model, scaler, and imputer
        X: Features to make predictions on
        
        Returns:
        Array of predictions
        """
        model = model_dict['model']
        scaler = model_dict['scaler']
        imputer = model_dict['imputer']
        
        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed)
        return model.predict(X_scaled)
    
    def _compute_top_features_by_shap(self, shap_values, feature_names):
        # fill nans with 0
        shap_values = shap_values.fillna(0)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": mean_abs_shap
        })
        feature_importance = feature_importance.sort_values(
            "importance", ascending=False
        )
        return feature_importance

    def _compute_cv_shap_importance(
        self,
        models: list,  # List of 10 trained model dicts from CV
        X_test_folds: list,  # List of test sets from each fold
    ) -> Dict[str, Any]:
        """
        Compute aggregated SHAP feature importance across CV folds.
        
        Args:
            models: List of trained model dictionaries from each CV fold
            X_test_folds: List of test DataFrames from each fold
            
        Returns:
            Dictionary with aggregated SHAP results
        """
        all_shap_values = []
        all_test_data = []
        
        # Compute SHAP values for each fold
        for i, (model_dict, X_test) in enumerate(zip(models, X_test_folds)):
            model = model_dict['model']
            scaler = model_dict['scaler']
            imputer = model_dict['imputer']
            
            # Preprocess test data
            X_test_imputed = imputer.transform(X_test)
            X_test_scaled = scaler.transform(X_test_imputed)
            
            # Create SHAP explainer for linear model
            explainer = shap.LinearExplainer(model, X_test_scaled)
            shap_values = explainer.shap_values(X_test_scaled)
            
            shap_value_df = pd.DataFrame(shap_values, columns=X_test.columns)

            all_shap_values.append(shap_value_df)
            all_test_data.append(X_test)
        
        # Concatenate all SHAP values and test data
        combined_shap_values = pd.concat(all_shap_values, axis=0)
        combined_test_data = pd.concat(all_test_data, axis=0)
        
        # Compute feature importance using aggregated SHAP values
        feature_importance = self._compute_top_features_by_shap(
            combined_shap_values, 
            combined_test_data.columns.tolist()
        )
        
        return {
            'combined_shap_values': combined_shap_values,
            'combined_test_data': combined_test_data,
            'feature_importance': feature_importance,
            'feature_names': combined_test_data.columns.tolist()
        }
    
    def compute_feature_importance(self):
        """
        Compute SHAP-based feature importance for Quantile Regression model.
            
        Returns:
        Dictionary containing SHAP results and top features
        """
        feature_importance = self._compute_cv_shap_importance(
            self.models,
            self.all_test_X
        )
        return feature_importance
        

    def save_feature_importance(self):
        """
        Save SHAP analysis results to disk. To subdir feature_importance
        """
        # get and save the feature importance results
        feature_importance = self.compute_feature_importance()
        shap_values = feature_importance['combined_shap_values']
        test_data = feature_importance['combined_test_data']
        feature_importance = feature_importance['feature_importance']

        # save to output directory
        if not os.path.exists(os.path.join(self.output_dir, 'feature_importance')):
            os.makedirs(os.path.join(self.output_dir, 'feature_importance'))
        feature_importance.to_csv(os.path.join(self.output_dir, 'feature_importance', 'feature_importance.csv'), index=False)
        shap_values.to_csv(os.path.join(self.output_dir, 'feature_importance', 'shap_values.csv'), index=False)
        test_data.to_csv(os.path.join(self.output_dir, 'feature_importance', 'test_data.csv'), index=False)

        # save shap summary plot
        save_shap_summary_plot(shap_values, test_data, os.path.join(self.output_dir, 'feature_importance', 'shap_summary.png'))

    def save_models(self):
        """Save models, scalers, and imputers as pkl files to subdir models"""
        os.makedirs(os.path.join(self.output_dir, 'models'), exist_ok=True)
        for i, model_dict in enumerate(self.models):
            with open(os.path.join(self.output_dir, 'models', f'model_{i}.pkl'), 'wb') as f:
                pickle.dump(model_dict, f)

class NeuralNetworkQuantileTrainer(BaseTrainer):
    """
    Neural Network Quantile Regression trainer that implements the BaseTrainer interface.
    
    This class provides a PyTorch-based feedforward neural network implementation 
    for quantile regression using pinball loss function.
    """
    
    def __init__(
        self, 
        output_dir: str, 
        n_threads: int = 8,
        quantile: float = 0.1,
        hidden_sizes: List[int] = None,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        dropout_rate: float = 0.2,
        n_epochs: int = 200,
        patience: int = 20,
        **nn_params
    ):
        """
        Initialize the Neural Network Quantile Regression trainer.
        
        Parameters:
        output_dir (str): Directory to save results and models.
        n_threads (int): Number of threads to use for training.
        quantile (float): Quantile to estimate (default: 0.1 for bottom 10%).
        hidden_sizes (List[int]): Hidden layer sizes (default: [128, 64, 32]).
        learning_rate (float): Learning rate for optimization.
        batch_size (int): Batch size for training.
        dropout_rate (float): Dropout rate for regularization.
        n_epochs (int): Maximum number of training epochs.
        patience (int): Early stopping patience.
        **nn_params: Additional neural network parameters.
        """
        super().__init__(output_dir, n_threads)
        
        self.quantile = quantile
        self.n_epochs = n_epochs
        self.patience = patience
        
        # Single configuration parameters
        if hidden_sizes is None:
            self.hidden_sizes = [128, 64, 32]
        else:
            self.hidden_sizes = hidden_sizes
            
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        
        # Device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set PyTorch thread limits to respect n_threads parameter
        message = set_pytorch_threads(self.n_threads)
        print(message)
        
        # Additional parameters
        self.nn_params = nn_params
    
    def _create_model(self, input_size: int, hidden_sizes: List[int], dropout_rate: float):
        """Create a feedforward neural network model."""
        class QuantileNet(nn.Module):
            def __init__(self, input_size, hidden_sizes, dropout_rate):
                super(QuantileNet, self).__init__()
                
                layers = []
                prev_size = input_size
                
                for hidden_size in hidden_sizes:
                    layers.append(nn.Linear(prev_size, hidden_size))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout_rate))
                    prev_size = hidden_size
                
                # Output layer
                layers.append(nn.Linear(prev_size, 1))
                
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                output = self.network(x)
                # Ensure output is always 2D for SHAP compatibility: (batch_size, 1)
                if output.dim() == 1:
                    output = output.unsqueeze(-1)
                return output
        
        return QuantileNet(input_size, hidden_sizes, dropout_rate)
    
    def _pinball_loss(self, y_pred, y_true, quantile):
        """Compute pinball loss for quantile regression."""
        # Ensure both tensors have the same shape
        if y_pred.dim() > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(-1)
        
        errors = y_true - y_pred
        loss = torch.where(
            errors >= 0,
            quantile * errors,
            (quantile - 1) * errors
        )
        return torch.mean(loss)
    
    def _train_neural_network(
        self, 
        model, 
        train_loader, 
        val_loader, 
        learning_rate: float
    ):
        """Train a single neural network model."""
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.n_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = self._pinball_loss(outputs, batch_y, self.quantile)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_x)
                    loss = self._pinball_loss(outputs, batch_y, self.quantile)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.n_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model, best_val_loss, train_losses, val_losses
    
    def train_single_model(self, X_train, y_train, X_val, y_val, **kwargs):
        """
        Train a single Neural Network Quantile model with fixed configuration.
        
        Parameters:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        **kwargs: Additional parameters (currently unused)
        
        Returns:
        dict: Dictionary containing the trained model, scaler, imputer, and parameters
        """
        # Handle missing values with median imputation
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_val_imputed = imputer.transform(X_val)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_val_scaled = scaler.transform(X_val_imputed)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train.values)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.FloatTensor(y_val.values)
        
        input_size = X_train_scaled.shape[1]
        
        # Create data loaders with thread-limited workers
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Limit DataLoader workers to respect thread limits (use at most n_threads-1 for workers)
        num_workers = min(max(1, self.n_threads - 1), 4) if self.n_threads > 1 else 0
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=self.device.type == 'cuda'
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=self.device.type == 'cuda'
        )
        
        # Create and train model
        model = self._create_model(input_size, self.hidden_sizes, self.dropout_rate)
        model = model.to(self.device)
        
        print(f"Training model with architecture: {self.hidden_sizes}, "
              f"LR: {self.learning_rate}, Batch: {self.batch_size}, "
              f"Dropout: {self.dropout_rate}")
        
        trained_model, val_loss, _, _ = self._train_neural_network(
            model, train_loader, val_loader, self.learning_rate
        )
        
        params = {
            'hidden_sizes': self.hidden_sizes,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'dropout_rate': self.dropout_rate
        }
        
        print(f"Training completed with validation pinball loss: {val_loss:.4f}")
        
        return {
            'model': trained_model,
            'scaler': scaler,
            'imputer': imputer,
            'params': params,
            'val_loss': val_loss,
            'device': self.device
        }
    
    def predict(self, model_dict, X):
        """
        Make predictions using a trained Neural Network model.
        
        Parameters:
        model_dict: Dictionary containing trained model, scaler, and imputer
        X: Features to make predictions on
        
        Returns:
        Array of predictions
        """
        model = model_dict['model']
        scaler = model_dict['scaler']
        imputer = model_dict['imputer']
        device = model_dict['device']
        
        # Preprocess data
        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed)
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        
        # Make predictions
        model.eval()
        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy()
            # Squeeze to get 1D predictions if needed
            if predictions.ndim > 1 and predictions.shape[1] == 1:
                predictions = predictions.squeeze(-1)
        
        return predictions
    
    def _compute_top_features_by_shap(self, shap_values, feature_names):
        # fill nans with 0
        shap_values = shap_values.fillna(0)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": mean_abs_shap
        })
        feature_importance = feature_importance.sort_values(
            "importance", ascending=False
        )
        return feature_importance

    def _compute_cv_shap_importance(
        self,
        models: list,  # List of trained model dicts from CV
        X_test_folds: list,  # List of test sets from each fold
    ) -> Dict[str, Any]:
        """
        Compute aggregated SHAP feature importance across CV folds.
        
        Args:
            models: List of trained model dictionaries from each CV fold
            X_test_folds: List of test DataFrames from each fold
            
        Returns:
            Dictionary with aggregated SHAP results
        """
        all_shap_values = []
        all_test_data = []
        
        # Compute SHAP values for each fold
        for i, (model_dict, X_test) in enumerate(zip(models, X_test_folds)):
            model = model_dict['model']
            scaler = model_dict['scaler']
            imputer = model_dict['imputer']
            device = model_dict['device']
            
            # Preprocess test data
            X_test_imputed = imputer.transform(X_test)
            X_test_scaled = scaler.transform(X_test_imputed)
            
            # Convert to tensors
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
            
            # Use a subset of data for background (SHAP can be memory intensive)
            background_size = min(100, X_test_scaled.shape[0])
            background_tensor = X_test_tensor[:background_size]
            
            # Create DeepExplainer
            explainer = shap.DeepExplainer(model, background_tensor)
            shap_values = explainer.shap_values(X_test_tensor)
            
            # Handle multi-output case (shap_values might be a list)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Take the first (and only) output
            
            # Ensure shap_values is 2D for DataFrame creation
            if shap_values.ndim == 3 and shap_values.shape[2] == 1:
                shap_values = shap_values.squeeze(-1)
            
            shap_value_df = pd.DataFrame(shap_values, columns=X_test.columns)

            all_shap_values.append(shap_value_df)
            all_test_data.append(X_test)
        
        # Concatenate all SHAP values and test data
        combined_shap_values = pd.concat(all_shap_values, axis=0)
        combined_test_data = pd.concat(all_test_data, axis=0)
        
        # Compute feature importance using aggregated SHAP values
        feature_importance = self._compute_top_features_by_shap(
            combined_shap_values, 
            combined_test_data.columns.tolist()
        )
        
        return {
            'combined_shap_values': combined_shap_values,
            'combined_test_data': combined_test_data,
            'feature_importance': feature_importance,
            'feature_names': combined_test_data.columns.tolist()
        }
    
    def compute_feature_importance(self):
        """
        Compute SHAP-based feature importance for Neural Network model.
            
        Returns:
        Dictionary containing SHAP results and top features
        """
        feature_importance = self._compute_cv_shap_importance(
            self.models,
            self.all_test_X
        )
        return feature_importance
        

    def save_feature_importance(self):
        """
        Save SHAP analysis results to disk. To subdir feature_importance
        """
        # get and save the feature importance results
        feature_importance = self.compute_feature_importance()
        shap_values = feature_importance['combined_shap_values']
        test_data = feature_importance['combined_test_data']
        feature_importance = feature_importance['feature_importance']

        # save to output directory
        if not os.path.exists(os.path.join(self.output_dir, 'feature_importance')):
            os.makedirs(os.path.join(self.output_dir, 'feature_importance'))
        feature_importance.to_csv(os.path.join(self.output_dir, 'feature_importance', 'feature_importance.csv'), index=False)
        shap_values.to_csv(os.path.join(self.output_dir, 'feature_importance', 'shap_values.csv'), index=False)
        test_data.to_csv(os.path.join(self.output_dir, 'feature_importance', 'test_data.csv'), index=False)

        # save shap summary plot
        save_shap_summary_plot(shap_values, test_data, os.path.join(self.output_dir, 'feature_importance', 'shap_summary.png'))

    def save_models(self):
        """Save models, scalers, and imputers as pkl files to subdir models"""
        os.makedirs(os.path.join(self.output_dir, 'models'), exist_ok=True)
        for i, model_dict in enumerate(self.models):
            # Save the entire model dictionary
            with open(os.path.join(self.output_dir, 'models', f'model_{i}.pkl'), 'wb') as f:
                # For PyTorch models, we need to save the state dict separately
                model_dict_copy = model_dict.copy()
                model_state = model_dict_copy['model'].state_dict()
                model_dict_copy['model_state_dict'] = model_state
                del model_dict_copy['model']  # Remove the model object itself
                # Convert device to string for serialization
                model_dict_copy['device_str'] = str(model_dict_copy['device'])
                del model_dict_copy['device']
                pickle.dump(model_dict_copy, f)
            
            # Save the model architecture and state separately
            torch.save({
                'model_state_dict': model_dict['model'].state_dict(),
                'params': model_dict['params']
            }, os.path.join(self.output_dir, 'models', f'model_{i}.pth'))

    def save_trainer(self):
        """Save the trainer object to a pkl file, handling PyTorch-specific serialization issues"""
        # Create a copy of the trainer state without problematic objects
        trainer_state = {
            'output_dir': self.output_dir,
            'n_threads': self.n_threads,
            'quantile': self.quantile,
            'hidden_sizes': self.hidden_sizes,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'dropout_rate': self.dropout_rate,
            'n_epochs': self.n_epochs,
            'patience': self.patience,
            'nn_params': self.nn_params,
            'device_str': str(self.device),  # Convert device to string
            'fold_results': self.fold_results,
            # Don't save models here - they're saved separately
            'num_models': len(self.models)
        }
        
        # Save the trainer state
        with open(os.path.join(self.output_dir, 'trainer_state.pkl'), 'wb') as f:
            pickle.dump(trainer_state, f)
        
        # Save additional info about the test data shapes for reconstruction
        if self.all_test_X:
            test_shapes = [X.shape for X in self.all_test_X]
            test_columns = [list(X.columns) for X in self.all_test_X]
            test_indices = [list(X.index) for X in self.all_test_X]
            
            with open(os.path.join(self.output_dir, 'test_data_info.pkl'), 'wb') as f:
                pickle.dump({
                    'shapes': test_shapes,
                    'columns': test_columns,
                    'indices': test_indices
                }, f)
        
        print(f"Neural Network trainer state saved to {self.output_dir}")

    @classmethod
    def load_trainer(cls, output_dir):
        """Load a saved neural network trainer from disk"""
        # Load trainer state
        with open(os.path.join(output_dir, 'trainer_state.pkl'), 'rb') as f:
            trainer_state = pickle.load(f)
        
        # Reconstruct trainer
        trainer = cls(
            output_dir=trainer_state['output_dir'],
            n_threads=trainer_state['n_threads'],
            quantile=trainer_state['quantile'],
            hidden_sizes=trainer_state['hidden_sizes'],
            learning_rate=trainer_state['learning_rate'],
            batch_size=trainer_state['batch_size'],
            dropout_rate=trainer_state['dropout_rate'],
            n_epochs=trainer_state['n_epochs'],
            patience=trainer_state['patience'],
            **trainer_state['nn_params']
        )
        
        # Restore device (it will auto-detect, but we can validate)
        trainer.device = torch.device(trainer_state['device_str'] if torch.cuda.is_available() 
                                    and 'cuda' in trainer_state['device_str'] else 'cpu')
        
        # Restore fold results
        trainer.fold_results = trainer_state['fold_results']
        
        # Load models if they exist
        models_dir = os.path.join(output_dir, 'models')
        if os.path.exists(models_dir):
            trainer.models = []
            for i in range(trainer_state['num_models']):
                # Load the model dictionary (without the model itself)
                with open(os.path.join(models_dir, f'model_{i}.pkl'), 'rb') as f:
                    model_dict = pickle.load(f)
                
                # Load the PyTorch model state
                model_checkpoint = torch.load(os.path.join(models_dir, f'model_{i}.pth'), 
                                            map_location=trainer.device)
                
                # Reconstruct the model
                input_size = model_dict['model_state_dict']['network.0.weight'].shape[1]
                model = trainer._create_model(input_size, trainer.hidden_sizes, trainer.dropout_rate)
                model.load_state_dict(model_checkpoint['model_state_dict'])
                model = model.to(trainer.device)
                
                # Reconstruct the full model dictionary
                model_dict['model'] = model
                model_dict['device'] = trainer.device
                trainer.models.append(model_dict)
        
        print(f"Neural Network trainer loaded from {output_dir}")
        return trainer


