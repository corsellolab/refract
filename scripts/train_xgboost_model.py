import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import pickle

# get path to ../refract
refract_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(refract_path)

from refract.trainers import train_xgboost_with_early_stopping
from refract.utils import load_feature_df, load_response_df, intersect_depmap_ids
from refract.importance import compute_shap_values, get_top_features_by_shap

def load_split(split_file):
    """Load train/val/test split assignments from a split file."""
    split_df = pd.read_csv(split_file)
    train_ids = split_df[split_df['split'] == 'train']['depmap_id'].values
    val_ids = split_df[split_df['split'] == 'val']['depmap_id'].values
    test_ids = split_df[split_df['split'] == 'test']['depmap_id'].values
    return train_ids, val_ids, test_ids

def load_selected_features(features_file):
    """Load selected features from features file."""
    features_df = pd.read_csv(features_file)
    return features_df['feature_name'].values

def evaluate_predictions(y_true, y_pred, set_name):
    """Calculate and print evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    pearson = np.corrcoef(y_true, y_pred)[0,1]
    print(f"{set_name} RMSE: {rmse:.4f}")
    print(f"{set_name} R2: {r2:.4f}")
    print(f"{set_name} Pearson: {pearson:.4f}")
    return {'rmse': rmse, 'r2': r2, 'pearson': pearson}

def save_model_predictions(fold: int, y_true, y_pred, set_name: str, output_dir: str):
    """Save model predictions to disk."""
    predictions_df = pd.DataFrame({
        'model_id': f'fold_{fold}',
        'true_value': y_true,
        'predicted_value': y_pred
    })
    predictions_df.to_csv(
        os.path.join(output_dir, f'fold_{fold}_{set_name}_predictions.csv'),
        index=True
    )

def save_shap_results(fold: int, shap_results: dict, output_dir: str):
    """Save SHAP analysis results to disk."""
    # Save SHAP values and feature importance
    shap_df = pd.DataFrame(
        shap_results['shap_values'],
        columns=shap_results['X_test_df'].columns,
        index=shap_results['X_test_df'].index
    )
    shap_df.to_csv(os.path.join(output_dir, f'fold_{fold}_shap_values.csv'))
    
    # Save test data used for SHAP calculation
    shap_results['X_test_df'].to_csv(
        os.path.join(output_dir, f'fold_{fold}_X_test.csv')
    )
    
    # Save explainer and expected value
    explainer_dict = {
        'explainer': shap_results['explainer'],
        'expected_value': shap_results['expected_value']
    }
    with open(os.path.join(output_dir, f'fold_{fold}_explainer.pkl'), 'wb') as f:
        pickle.dump(explainer_dict, f)

def main():
    parser = argparse.ArgumentParser(description='Train XGBoost model using cross-validation')
    parser.add_argument('--feature_file', type=str, required=True, help='Path to feature file')
    parser.add_argument('--response_file', type=str, required=True, help='Path to response file')
    parser.add_argument('--split_dir', type=str, required=True, help='Directory containing split files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--n_threads', type=int, default=8, help='Number of threads for XGBoost')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and preprocess data
    feature_df = load_feature_df(args.feature_file)
    response_df = load_response_df(args.response_file)
    response_df, feature_df = intersect_depmap_ids(response_df, feature_df)
    
    # Store results for each fold
    fold_results = []
    
    # Store all predictions for overall metrics
    all_val_true = []
    all_val_pred = []
    all_test_true = []
    all_test_pred = []
    
    # For each fold
    n_splits = 10
    for fold in range(n_splits):
        print(f"\nProcessing fold {fold}")
        
        # Load split assignments and selected features
        split_file = os.path.join(args.split_dir, f'{fold}.split.txt')
        features_file = os.path.join(args.split_dir, f'{fold}.features.csv')
        
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
        model = train_xgboost_with_early_stopping(
            X_train, y_train,
            X_val, y_val,
            n_threads=args.n_threads
        )
        
        # Make predictions
        dval = xgb.DMatrix(X_val)
        dtest = xgb.DMatrix(X_test)
        val_preds = model.predict(dval)
        test_preds = model.predict(dtest)
        
        # Save predictions
        save_model_predictions(fold, y_val, val_preds, 'val', args.output_dir)
        save_model_predictions(fold, y_test, test_preds, 'test', args.output_dir)
        
        # Compute and save SHAP values
        shap_results = compute_shap_values(model, X_test)
        save_shap_results(fold, shap_results, args.output_dir)
        
        # Get top features by SHAP values
        top_features = get_top_features_by_shap(
            shap_results['shap_values'],
            selected_features
        )
        
        print("\nTop 10 features by SHAP importance:")
        print(top_features)
        
        # Evaluate performance
        print(f"\nFold {fold} Results:")
        val_metrics = evaluate_predictions(y_val, val_preds, "Validation")
        test_metrics = evaluate_predictions(y_test, test_preds, "Test")
        
        # Store predictions for overall metrics
        all_val_true.extend(y_val)
        all_val_pred.extend(val_preds)
        all_test_true.extend(y_test)
        all_test_pred.extend(test_preds)
        
        # Store results
        fold_results.append({
            'fold': fold,
            'val_rmse': val_metrics['rmse'],
            'val_r2': val_metrics['r2'],
            'val_pearson': val_metrics['pearson'],
            'test_rmse': test_metrics['rmse'],
            'test_r2': test_metrics['r2'],
            'test_pearson': test_metrics['pearson']
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
        'test_pearson': overall_test_metrics['pearson']
    }
    fold_results.append(overall_results)
    
    # Save summary statistics
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(os.path.join(args.output_dir, 'model_performance_summary.csv'), index=False)
    
    # Print summary statistics
    print("\nPer-fold Summary Statistics:")
    print("\nValidation Metrics:")
    print(f"Mean RMSE: {results_df['val_rmse'].mean():.4f} ± {results_df['val_rmse'].std():.4f}")
    print(f"Mean R2: {results_df['val_r2'].mean():.4f} ± {results_df['val_r2'].std():.4f}")
    print(f"Mean Pearson: {results_df['val_pearson'].mean():.4f} ± {results_df['val_pearson'].std():.4f}")
    print("\nTest Metrics:")
    print(f"Mean RMSE: {results_df['test_rmse'].mean():.4f} ± {results_df['test_rmse'].std():.4f}")
    print(f"Mean R2: {results_df['test_r2'].mean():.4f} ± {results_df['test_r2'].std():.4f}")
    print(f"Mean Pearson: {results_df['test_pearson'].mean():.4f} ± {results_df['test_pearson'].std():.4f}")

if __name__ == "__main__":
    main()

