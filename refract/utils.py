import logging
import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from tqdm import tqdm

from refract.metrics import (
    get_merged_shap_values_and_features,
    get_stringdb_network_interactions,
    get_test_predictions,
    get_top_k_features,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


from scipy.stats import kendalltau
from kneed import KneeLocator
from scipy.stats import zscore

def get_fold_assignment(responses, key_pert_name):
    tmp_y = responses.loc[responses.pert_name == key_pert_name, :]
    tmp_y = tmp_y.drop_duplicates()
    tmp_y["decile"] = pd.qcut(tmp_y["LFC.cb"], 10, labels=False, duplicates="drop")
    # assign to a fold 
    tmp_y["fold"] = np.nan
    for i in range(10):
        tmp_y.loc[tmp_y.decile == i, "fold"] =  np.random.choice(range(10), sum(tmp_y["decile"] == i), replace=True)
    tmp_y["fold"] = tmp_y["fold"].astype(int)
    # get a dictionary mapping cell lines to fold
    fold_dict = tmp_y["fold"].to_dict()
    return fold_dict

def get_correlated_features(y, X, colnames):
    # Step 1: Compute the correlation for each column
    # compute correlations with kendall tau
    correlations = np.array([kendalltau(y, X[:, i])[0] for i in range(X.shape[1])])
    # fill NaN with 0
    correlations[np.isnan(correlations)] = 0
    
    # Step 2: Group columns by TYPE
    correlations_dict = {}
    feature_names = {}
    for i, colname in enumerate(colnames):
        type_name = colname.split("_")[0]
        if type_name not in correlations_dict:
            correlations_dict[type_name] = []
        if type_name not in feature_names:
            feature_names[type_name] = []
        correlations_dict[type_name].append(correlations[i])
        feature_names[type_name].append(colname)
    
    # Step 3: Sample the top p proportion of correlated features within each type
    selected_colnames = []
    for type_name in correlations_dict:
        tmp_corr = np.array(correlations_dict[type_name])
        k_features = _select_k_features(tmp_corr)
        selected_colnames.extend([feature_names[type_name][i] for i in k_features])

    # Step 4: Return the list of column names
    return selected_colnames

def _select_k_features(correlations):
    z_scores = zscore(correlations)
    sorted_z_scores = np.sort(np.abs(z_scores))[::-1]
    # Calculate the differences between consecutive z-scores
    differences = np.diff(sorted_z_scores)

    # Use KneeLocator to find the elbow point, starting from index 10
    kneedle = KneeLocator(range(10, len(differences) + 1), differences[9:], 
                        curve='convex', direction='decreasing')

    optimal_k = kneedle.knee

    # If no elbow is found after 10, default to 10
    if optimal_k is None:
        optimal_k = 10
    else:
        # Add 10 to the result since we started from index 10
        optimal_k += 10
    # Select top k features
    # TODO: Remove: take optimal_k = 50
    optimal_k = 50
    top_features = np.argsort(np.abs(z_scores))[::-1][:optimal_k]
    return top_features


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


def save_output(trainers, output_dir):
    # compute SHAP values and predictions across the full dataset
    logger.info("Aggregating SHAP values and predictions...")
    shap_values, features, feature_names = get_merged_shap_values_and_features(trainers)
    test_df = get_test_predictions(trainers)

    # save test_df to file
    logger.info("Saving training results to train_results.csv...")
    test_df.to_csv(os.path.join(output_dir, "train_results.csv"), index=False)

    # plot a scatter plot of predictions vs actual
    logger.info("Plotting scatterplot to train_results.png...")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(test_df["label"], test_df["pred"], alpha=0.5)
    ax.set_xlabel("LFC")
    ax.set_ylabel("Ranking Score")
    fig.savefig(os.path.join(output_dir, "train_results.png"))
    plt.close()

    # compute pearson correlation between pred and true
    logger.info("Computing pearson correlation...")
    train_corr = np.corrcoef(test_df["label"], test_df["pred"])[0, 1]
    logger.info(f"Overall test correlation: {train_corr}")
    with open(os.path.join(output_dir, "train_corr.txt"), "w") as f:
        f.write(str(train_corr))

    # save SHAP summary plot
    logger.info("Saving SHAP summary plot to shap_summary_plot.png...")
    shap.summary_plot(shap_values, features, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"))
    plt.close()

    # get the gene name of top features
    logger.info("Getting top features...")
    top_feature_names = get_top_k_features(shap_values, feature_names, k=20)
    top_feature_genes = [i.split("_")[1] for i in top_feature_names]

    # save the shap values and the top feature names
    logger.info("Saving the SHAP values and top feature names...")
    # save shap values as a dataframe
    shap_values_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_values_df.to_csv(os.path.join(output_dir, "shap_values.csv"), index=False)
    # save top feature names as a text file with one feature per line
    with open(os.path.join(output_dir, "top_feature_names.txt"), "w") as f:
        for item in top_feature_names:
            f.write("%s\n" % item)

    # get connectivity of top features
    logger.info("Getting network interactions...")
    network_interactions, _ = get_stringdb_network_interactions(top_feature_genes)
    network_interactions.to_csv(
        os.path.join(output_dir, "network_interactions.csv"), index=False
    )

    # save training details
    #logger.info("Saving training details...")
    #training_details = get_training_details(trainers)
    #training_details.to_csv(os.path.join(output_dir, "training_details.csv"), index=False)

    # save trainers
    #logger.info("Saving trainers to trainers.pkl...")
    #with open(os.path.join(output_dir, "trainers.pkl"), "wb") as f:
    #    pickle.dump(trainers, f)




def moving_window_average(lst, window_size=3):
    if window_size % 2 == 0:
        raise ValueError("Please provide an odd window size.")

    half_window = window_size // 2
    extended_list = [lst[0]] * half_window + lst + [lst[-1]] * half_window

    averages = []
    for i in range(half_window, len(extended_list) - half_window):
        avg = sum(extended_list[i - half_window : i + half_window + 1]) / window_size
        averages.append(avg)

    return averages

def get_training_details(trainers):
    # For each trainer, save training details
    all_fold_trainers = []
    for trainer in trainers:
        # get model_name 
        model_name = trainer.automl.best_estimator
        # get config
        config = trainer.automl.best_config
        # train time
        train_time = trainer.automl.best_config_train_time

        # construct a dictionary
        training_details = {
            "model_name": model_name,
            "train_time": train_time,
            **config,
        }
        all_fold_trainers.append(training_details)
    df = pd.DataFrame(all_fold_trainers)
    return df
