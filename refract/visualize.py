import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import shap
from pathlib import Path
from glob import glob

'''
def concatenate_fold_results(model_dir: str, pattern: str) -> pd.DataFrame:
    """Concatenate results files across all folds in a model directory."""
    files = list(Path(model_dir).glob(pattern))
    dfs = []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        fold = int(f.stem.split('_')[1])  # Extract fold number from filename
        df['fold'] = fold
        dfs.append(df)
    return pd.concat(dfs, ignore_index=False)
'''

def concatenate_fold_results(model_dir: str, pattern: str) -> pd.DataFrame:
    files = glob(f"{model_dir}/{pattern}")
    files = [pd.read_csv(f, index_col=0) for f in files]
    return pd.concat(files, ignore_index=False)

def create_shap_summary_plot(shap_values: np.ndarray, X_test: pd.DataFrame, output_path: str):
    """Create and save a SHAP summary plot."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_prediction_scatter(pred_df: pd.DataFrame, output_path: str):
    """Create and save a scatter plot of predicted vs true values."""
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=pred_df,
        x='true_value',
        y='predicted_value',
        alpha=0.5
    )
    
    # Add diagonal line
    lims = [
        min(pred_df['true_value'].min(), pred_df['predicted_value'].min()),
        max(pred_df['true_value'].max(), pred_df['predicted_value'].max())
    ]
    plt.plot(lims, lims, 'k--', alpha=0.5)
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs True Values')
    
    # Add correlation coefficient to plot
    corr = pred_df['true_value'].corr(pred_df['predicted_value'])
    plt.text(0.05, 0.95, f'r = {corr:.3f}', 
             transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def summarize_model_results(model_dir: str, output_dir: str):
    """
    Summarize model results across all folds.

    Args:
        model_dir: Directory containing model fold results
        output_dir: Directory to save summary results
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Concatenate SHAP values
    shap_df = concatenate_fold_results(model_dir, '*_shap_values.csv')
    # get the column order
    col_order = shap_df.columns
    # get the index order
    index_order = shap_df.index
    shap_df.to_csv(f'{output_dir}/combined_shap_values.csv')

    # Concatenate test data
    X_test_df = concatenate_fold_results(model_dir, '*_X_test.csv')
    # make the column order the same
    X_test_df = X_test_df.loc[:, col_order]
    # make the index order the same
    X_test_df = X_test_df.loc[index_order]
    X_test_df.to_csv(f'{output_dir}/combined_X_test.csv')

    # Concatenate predictions
    pred_df = concatenate_fold_results(model_dir, '*_test_predictions.csv')
    pred_df.to_csv(f'{output_dir}/combined_predictions.csv')

    # Create SHAP summary plot
    # fill shap with 0s for missing values
    shap_df = shap_df.fillna(0)
    # fill with mean feature value for each feature
    X_test_df = X_test_df.fillna(X_test_df.mean())
    shap_df = shap_df.astype(float)
    X_test_df = X_test_df.astype(float)
    create_shap_summary_plot(shap_df.values, X_test_df, f'{output_dir}/shap_summary.png')

    # Create prediction scatter plot
    create_prediction_scatter(pred_df, f'{output_dir}/prediction_scatter.png')
