"""Metrics for evaluating recommender systems."""
from typing import List

import numpy as np
import pandas as pd
import stringdb


def compute_accuracy_at_k(df, k_list=[1, 5, 10, 20, 50]):
    out = {}
    for k in k_list:
        df["preds_rank"] = df["preds"].rank(ascending=False)
        df["true_rank"] = df["true"].rank(ascending=False)

        df["preds_rank"] = df["preds_rank"].astype(int)
        df["true_rank"] = df["true_rank"].astype(int)

        df["top_true"] = df["true_rank"].apply(lambda x: 1 if x <= k else 0)
        df["top_pred"] = df["preds_rank"].apply(lambda x: 1 if x <= k else 0)
        out[k] = (df["top_pred"] & df["top_true"]).sum() / k
    return out


def get_stringdb_network_interactions(gene_list):
    """Call STRINGdb API to get the network interactions for genes in gene_list
    Return the network interaction as a dataframe and the list of edges.

    Args:
        gene_list (List): List of genes to pass to the API
    """
    network_interactions = stringdb.get_network(gene_list)
    edges = []
    for _, row in network_interactions.iterrows():
        edges.append((row["preferredName_A"], row["preferredName_B"]))
        edges.append((row["preferredName_B"], row["preferredName_A"]))
    edges = list(set(edges))
    return network_interactions, edges


def get_merged_shap_values_and_features(trainer_list):
    """Aggregate SHAP values and features from multiple trainers."""
    # get the shap values and features from each trainer
    shap_dfs = [
        pd.DataFrame(i.shap_values, columns=i.test_feature_names) for i in trainer_list
    ]
    feature_dfs = [
        pd.DataFrame(i.test_features, columns=i.test_feature_names)
        for i in trainer_list
    ]

    # concatenate
    shap_df = pd.concat(shap_dfs, axis=0)
    feature_df = pd.concat(feature_dfs, axis=0)

    # fill in the missing values with 0
    shap_df = shap_df.fillna(0)
    feature_df = feature_df.fillna(0)

    # return as numpy arrays
    return shap_df.values, feature_df.values, feature_df.columns


def get_top_k_features(merged_shap_values, feature_names, k=20):
    """Get the top k features from the SHAP values."""
    top_features = np.argsort(np.abs(merged_shap_values).mean(0))[-k:][::-1]
    top_feature_names = [feature_names[i] for i in top_features]
    return top_feature_names


def get_test_predictions(trainer_list):
    """Return test_df from multiple trainers.

    Args:
        trainer_list (List): List of trainer objects
    """
    # create a dataframe of test predictions
    test_preds = []
    test_labels = []
    test_ccle_names = []
    for trainer in trainer_list:
        test_df = trainer.test_results_df
        test_preds.append(test_df["preds"].values)
        test_labels.append(test_df["LFC.cb"].values)
        test_ccle_names.append(test_df["ccle_name"].values)
    test_preds = np.concatenate(test_preds, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    test_ccle_names = np.concatenate(test_ccle_names, axis=0)
    test_df = pd.DataFrame(
        {
            "ccle_name": test_ccle_names,
            "pred": test_preds,
            "label": test_labels,
        }
    )
    return test_df
