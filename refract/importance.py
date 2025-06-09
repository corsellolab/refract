from typing import Any, Dict

import numpy as np
import pandas as pd
import shap
import xgboost as xgb


def compute_shap_values(
    model: xgb.Booster, X_test: pd.DataFrame, compute_interactions: bool = True
) -> Dict[str, Any]:
    """
    Compute SHAP values for a trained XGBoost model.

    Args:
        model: Trained XGBoost model
        X_test: Test features as a pandas DataFrame
        compute_interactions: Whether to compute SHAP interaction values

    Returns:
        Dictionary containing:
            - explainer: TreeExplainer object
            - expected_value: Base value that would be predicted if no features were known
            - shap_values: SHAP values for each prediction
            - shap_interaction_values: SHAP interaction values (if compute_interactions=True)
            - X_test_df: The test dataset used for computing SHAP values
    """
    # Initialize the SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)

    # Get expected value
    expected_value = explainer.expected_value

    # Initialize results dictionary
    results = {
        "explainer": explainer,
        "expected_value": expected_value,
        "shap_values": shap_values,
        "X_test_df": X_test,
    }

    # Optionally compute interaction values
    if compute_interactions:
        shap_interaction_values = explainer.shap_interaction_values(X_test)
        results["shap_interaction_values"] = shap_interaction_values

    return results


def get_top_features_by_shap(
    shap_values: np.ndarray,
    feature_names: list,
) -> pd.DataFrame:
    """
    Get the top features ranked by mean absolute SHAP values.

    Args:
        shap_values: SHAP values array from TreeExplainer
        feature_names: List of feature names
        top_n: Number of top features to return

    Returns:
        DataFrame with feature names and their mean absolute SHAP values,
        sorted by importance
    """
    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Create DataFrame with feature names and importance
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": mean_abs_shap}
    )

    # Sort by importance and get top N features
    feature_importance = feature_importance.sort_values("importance", ascending=False)

    return feature_importance


def compute_top_feature_shap_values(
    model: xgb.Booster, X_test: pd.DataFrame, compute_interactions: bool = True
) -> Dict[str, Any]:
    """
    Compute SHAP values for the most important feature in a trained XGBoost model.

    Args:
        model: Trained XGBoost model
        X_test: Test features as a pandas DataFrame
        compute_interactions: Whether to compute SHAP interaction values

    Returns:
        Dictionary containing:
            - explainer: TreeExplainer object
            - expected_value: Base value that would be predicted if no features were known
            - shap_values: SHAP values for the top feature
            - shap_interaction_values: SHAP interaction values for the top feature (if compute_interactions=True)
            - X_test_df: The test dataset used for computing SHAP values
            - top_feature: Name of the most important feature
    """
    # First compute all SHAP values to identify the top feature
    all_shap = compute_shap_values(
        model, X_test, compute_interactions=compute_interactions
    )
    # Get the top feature
    feature_importance = get_top_features_by_shap(
        all_shap["shap_values"], X_test.columns
    )
    all_shap["top_features"] = feature_importance

    return all_shap
