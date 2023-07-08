import logging
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from torch.utils.data import Dataset

from refract.utils import AttrDict

logger = logging.getLogger(__name__)


class FeatureSet:
    """Feature dataset used for MTS analysis"""

    def __init__(self, feature_dir, verbose=True):
        self.feature_dir = feature_dir
        self.feature_tables = AttrDict()
        self.continuous_feature_tables = AttrDict()
        self.discrete_feature_tables = AttrDict()

    def load_individual_feature_table(self, feature_name: str):
        """Load feature tables from a directory"""
        self.feature_tables[feature_name] = pd.read_pickle(
            f"{self.feature_dir}/{feature_name}.pkl"
        )

    def load_concatenated_feature_tables(self):
        X_all = pd.read_pickle(f"{self.feature_dir}/x-all.pkl")
        X_all.index.name = "ccle_name"
        # remove duplicate rows and columns
        X_all = X_all.drop_duplicates()
        X_all = X_all.loc[:, list(set(X_all.columns))]
        self.feature_tables["all"] = X_all

    def get_feature_df(self, feature_name):
        return self.feature_tables[feature_name]


class ResponseSet:
    """Response data"""

    def __init__(self, response_path, label_col="LFC.cb"):
        self.response_path = response_path
        self.response_table = pd.DataFrame()
        self.label_col = label_col

    def load_response_table(self):
        rt = pd.read_csv(self.response_path)
        rt = rt.drop_duplicates(
            subset=[
                "pert_name",
                "ccle_name",
                "culture",
                "pert_idose",
                "pert_mfc_id",
                self.label_col,
            ]
        )
        rt = rt.rename(columns={self.label_col: "response", "pert_idose": "dose"})
        rt = rt[rt["response"].notna()]
        self.response_table = rt

    def get_response_df(self, dose=None):
        if dose:
            return self.response_table.loc[self.response_table["dose"] == dose, :]
        else:
            return self.response_table


class PrismDataset(Dataset):
    def __init__(
        self,
        response_df,
        feature_df,
        slate_length=10,
        label_transformer=None,
        prioritize_sensitive=True,
    ):
        self.response_df = response_df
        self.feature_df = feature_df
        self.slate_length = slate_length
        self.prioritize_sensitive = prioritize_sensitive

        # quantile transform labels
        self.response_df["unscaled_response"] = self.response_df["response"].values
        if not label_transformer:
            self.label_transformer = MinMaxScaler()
        else:
            self.label_transformer = label_transformer
        self.response_df["response"] = self.label_transformer.fit_transform(
            self.response_df[["response"]]
        )
        if self.prioritize_sensitive:
            self.response_df.loc[:, "response"] = (
                1 - self.response_df.loc[:, "response"]
            )
        # scale from 0 to 5
        self.response_df.loc[:, "response"] = self.response_df.loc[:, "response"] * 5

        # Join response_df and feature_df on "ccle_name"
        self.joined_df = pd.merge(self.response_df, self.feature_df, on="ccle_name")
        # set index to "ccle_name"
        self.joined_df = self.joined_df.set_index("ccle_name")
        self.unscaled_labels = self.joined_df["unscaled_response"].values

        # Filter and order the columns
        self.cols = list(self.feature_df.columns) + ["response"]
        self.joined_df = self.joined_df.loc[:, self.cols]

        # impute missing values
        self.joined_df = self.joined_df.fillna(-1)

        # get ccle_names
        self.ccle_names = self.joined_df.index.tolist()

        # threshold slate_length, edge case
        if len(self.ccle_names) < self.slate_length:
            self.slate_length = len(self.ccle_names)

    def __len__(self):
        return len(self.ccle_names)

    def __getitem__(self, idx):
        # get slate_length - 1 samples from ccle_names
        index_name = self.ccle_names[idx]
        ccle_names = np.random.choice(
            self.ccle_names, self.slate_length - 1, replace=False
        ).tolist()

        # get [index_name, *ccle_names] from joined_df
        samples = self.joined_df.loc[[index_name] + ccle_names, :]

        # Extract features and labels from the samples
        ccle_name = self.ccle_names[idx]
        features = torch.tensor(samples.iloc[:, :-1].values, dtype=torch.float32)
        labels = torch.tensor(samples.iloc[:, -1].values.squeeze(), dtype=torch.float32)

        return ccle_name, features, labels
