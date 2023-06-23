"""Metrics for evaluating recommender systems."""
from typing import List

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
