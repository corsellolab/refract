import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


def get_data_splits(response_df, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits)
    split_assignments = []
    for train_index, test_index in skf.split(response_df, response_df.decile):
        split_assignments.append((train_index, test_index))
    return split_assignments

def get_data_for_split(response_df, feature_df, data_splits, split_index):
    train_index, test_index = data_splits[split_index]
    train_response_df = response_df.iloc[train_index]
    test_response_df = response_df.iloc[test_index]
    train_feature_df = feature_df.iloc[train_index]
    test_feature_df = feature_df.iloc[test_index]
    return train_response_df, test_response_df, train_feature_df, test_feature_df
