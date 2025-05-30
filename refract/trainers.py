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

from .utils import (
    load_feature_df, load_response_df, intersect_depmap_ids,
    load_split, load_selected_features, evaluate_predictions
)
from .visualize import save_shap_summary_plot, save_prediction_scatter

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
            print(f"\nProcessing fold {fold}")
            
            # Load split assignments and selected features
            split_file = os.path.join(split_dir, f'{fold}.split.txt')
            features_file = os.path.join(split_dir, f'{fold}.features.csv')
            
            train_ids, val_ids, test_ids = load_split(split_file)
            selected_features = load_selected_features(features_file)
            
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
            model = self.train_single_model(X_train, y_train, X_val, y_val, **kwargs)
            self.models.append(model)
            self.all_test_X.append(X_test)
            
            # Make predictions
            val_preds = self.predict(model, X_val)
            test_preds = self.predict(model, X_test)
            
            # Evaluate performance
            print(f"\nFold {fold} Results:")
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


