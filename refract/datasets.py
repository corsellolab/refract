import logging
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
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

    def load_individual_feature_tables(self):
        """Load feature tables from a directory"""
        self.feature_tables["GE"] = pd.read_csv(
            f"{self.feature_dir}/ge.csv", index_col=0
        )
        self.feature_tables["CNA"] = pd.read_csv(
            f"{self.feature_dir}/cna.csv", index_col=0
        )
        self.feature_tables["MET"] = pd.read_csv(
            f"{self.feature_dir}/met.csv", index_col=0
        )
        self.feature_tables["miRNA"] = pd.read_csv(
            f"{self.feature_dir}/mirna.csv", index_col=0
        )
        self.feature_tables["MUT"] = pd.read_csv(
            f"{self.feature_dir}/mut.csv", index_col=0
        )
        self.feature_tables["PROT"] = pd.read_csv(
            f"{self.feature_dir}/prot.csv", index_col=0
        )
        self.feature_tables["XPR"] = pd.read_csv(
            f"{self.feature_dir}/xpr.csv", index_col=0
        )
        self.feature_tables["LIN"] = pd.read_csv(
            f"{self.feature_dir}/lin.csv", index_col=0
        )
        self.feature_tables["shRNA"] = pd.read_csv(
            f"{self.feature_dir}/shrna.csv", index_col=0
        )
        self.feature_tables["REP"] = pd.read_csv(
            f"{self.feature_dir}/rep.csv", index_col=0
        )

        self.continuous_features = {
            "GE": self.feature_tables.GE,
            "CNA": self.feature_tables.CNA,
            "MET": self.feature_tables.MET,
            "miRNA": self.feature_tables.miRNA,
            "PROT": self.feature_tables.PROT,
            "XPR": self.feature_tables.XPR,
            "shRNA": self.feature_tables.shRNA,
            "REP": self.feature_tables.REP,
        }
        self.discrete_features = {
            "MUT": self.feature_tables.MUT,
            "LIN": self.feature_tables.LIN,
        }

    def load_concatenated_feature_tables(self):
        X_all = pd.read_pickle(f"{self.feature_dir}/x-all.pkl")
        X_all.index.name = "ccle_name"
        # remove duplicate rows and columns
        X_all = X_all.drop_duplicates()
        X_all = X_all.loc[:, list(set(X_all.columns))]
        self.feature_tables["all"] = X_all

    def get_lineage_PCs(self):
        pca = PCA()
        LIN_PCs = pca.fit_transform(self.feature_tables["LIN"])
        # get important PCs
        selected_PCs = pca.components_[pca.explained_variance_ > 0.2]
        LIN_PCs = self.feature_tables["LIN"] @ selected_PCs.T
        self.feature_tables["LIN_PCs"] = LIN_PCs


class ResponseSet:
    """Response data"""

    def __init__(self, response_path, verbose=True):
        self.response_path = response_path
        self.response_table = None
        self.response_with_features = None
        self.verbose = verbose

    def load_response_table(self):
        LFC = pd.read_csv(self.response_path)
        LFC = LFC.drop_duplicates(
            subset=[
                "pert_name",
                "ccle_name",
                "culture",
                "pert_idose",
                "pert_mfc_id",
                "LFC.cb",
            ]
        )
        LFC = LFC.rename(columns={"LFC.cb": "response", "pert_idose": "dose"})
        LFC = LFC[LFC["response"].notna()]
        self.LFC = LFC

    def get_joined_features(
        self,
        pert_name: str,
        pert_mfc_id: str,
        dose: float,
        feature_set: FeatureSet,
        feature_name: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get features for a given perturbation"""
        # get response for perturbation
        Y = self.LFC.loc[
            (self.LFC["pert_name"] == pert_name)
            & (self.LFC["pert_mfc_id"] == pert_mfc_id)
            & (self.LFC["dose"] == dose),
            :,
        ]
        y = Y.loc[:, ["ccle_name", "response"]].set_index("ccle_name")

        # get feature set
        X = feature_set.feature_tables[feature_name]

        # join features and response
        joined = y.merge(X, how="inner", left_index=True, right_index=True)

        # print some stats
        if self.verbose:
            logger.info(f"Total responses: {len(y)}")
            logger.info(f"Number of responses with features: {len(joined)}")
            logger.info(f"Number of features: {len(joined.columns) - 1}")

        y = joined.loc[:, ["response"]]
        X = joined.drop(columns=["response"])
        return X, y


class PrismDataset(Dataset):
    def __init__(
        self,
        response_df,
        feature_df,
        slate_length,
        feature_cols,
        feature_transformer=None,
        label_transformer=None,
    ):
        self.response_df = response_df
        self.unscaled_response_df = response_df.copy()
        self.feature_df = feature_df
        self.slate_length = slate_length
        self.top_features = feature_cols

        # filter self.feature_df to include only the top features
        self.feature_df = self.feature_df.loc[:, self.top_features].copy()

        # quantile transform all features
        if not feature_transformer:
            self.feature_transformer = PowerTransformer()
            self.feature_transformer.fit(self.feature_df)
        else:
            self.feature_transformer = feature_transformer

        # transform the feature_df
        self.feature_df = pd.DataFrame(
            self.feature_transformer.transform(self.feature_df),
            columns=self.feature_df.columns,
            index=self.feature_df.index,
        )

        # MinMax transform labels
        if not label_transformer:
            self.label_transformer = MinMaxScaler()
        else:
            self.label_transformer = label_transformer
        self.response_df["LFC.cb"] = self.label_transformer.fit_transform(
            self.response_df[["LFC.cb"]]
        )
        self.response_df.loc[:, "LFC.cb"] = 1 - self.response_df.loc[:, "LFC.cb"]

        # scale from 0 to 5
        self.response_df.loc[:, "LFC.cb"] = self.response_df.loc[:, "LFC.cb"] * 5

        # Join response_df and feature_df on "ccle_name"
        self.joined_df = pd.merge(self.response_df, self.feature_df, on="ccle_name")

        # set index to "ccle_name"
        self.joined_df = self.joined_df.set_index("ccle_name")

        # Order the columns
        self.cols = self.top_features + ["LFC.cb"]
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
