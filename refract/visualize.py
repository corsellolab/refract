import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import shap
from pathlib import Path
from glob import glob
from scipy.stats import pearsonr

def save_shap_summary_plot(shap_values: pd.DataFrame, X_test: pd.DataFrame, output_path: str):
    """Save SHAP summary plot to output path"""
    s = shap_values.fillna(0).values
    plt.figure(figsize=(10, 8))
    shap.summary_plot(s, X_test, show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_prediction_scatter(pred, true, output_path):
    """Save prediction scatter plot to output path"""
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=true, y=pred, alpha=0.5)

    lims = [
        min(min(pred), min(true)),
        max(max(pred), max(true))
    ]
    plt.plot(lims, lims, 'k--', alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs True Values')

    # add correlation coefficient to plot
    corr, _ = pearsonr(pred, true)
    plt.text(0.05, 0.95, f'r = {corr:.3f}', 
             transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()