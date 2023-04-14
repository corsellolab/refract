import pandas as pd
import re
import logging
from sklearn.decomposition import PCA
from utils import AttrDict
from typing import Tuple, List

logger = logging.getLogger(__name__)


class FeatureSet:
    """Feature dataset used for MTS analysis"""

    def __init__(self, feature_dir, verbose=True):
        self.feature_dir = feature_dir
        self.feature_tables = AttrDict()
        self.continuous_feature_tables = AttrDict()
        self.discrete_feature_tables = AttrDict()

    def load_feature_tables(self):
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
        self.feature_tables["repurposing"] = pd.read_csv(
            f"{self.feature_dir}/rep.csv", index_col=0
        )
        # repurposing meta table requires some formatting
        repurposing_meta = pd.read_csv(f"{self.feature_dir}/rep_info.csv")
        # format the repurposing_meta table
        repurposing_meta["column_name"] = "REP_" + repurposing_meta[
            "column_name"
        ].astype(str)
        repurposing_meta["name"] = (
            repurposing_meta["name"]
            .astype(str)
            .apply(lambda x: re.sub(r"[^\w\s]", "-", x))
        )
        # remove columns
        repurposing_meta = repurposing_meta.drop(columns=["dose", "screen_id"])
        self.feature_tables["repurposing_meta"] = repurposing_meta

        self.continuous_features = {
            "GE": self.feature_tables.GE,
            "CNA": self.feature_tables.CNA,
            "MET": self.feature_tables.MET,
            "miRNA": self.feature_tables.miRNA,
            "PROT": self.feature_tables.PROT,
            "XPR": self.feature_tables.XPR,
            "shRNA": self.feature_tables.shRNA,
            "REP": self.feature_tables.repurposing,
        }
        self.discrete_features = {
            "MUT": self.feature_tables.MUT,
            "LIN": self.feature_tables.LIN,
        }

    def load_concatenated_feature_tables(self):
        X_all = pd.read_csv(f"{self.feature_dir}/x-all.csv")
        X_all = X_all.set_index(X_all.columns[0])
        X_all.index.name = "ccle_name"
        # remove duplicate rows and columns
        X_all = X_all.drop_duplicates()
        X_all = X_all.loc[:, list(set(X_all.columns))]
        self.feature_tables["all"] = X_all

        X_ccle = pd.read_csv(f"{self.feature_dir}/x-ccle.csv")
        X_ccle = X_ccle.set_index(X_ccle.columns[0])
        X_ccle.index.name = "ccle_name"
        # remove duplicate rows and columns
        X_ccle = X_ccle.drop_duplicates()
        X_ccle = X_ccle.loc[:, list(set(X_ccle.columns))]
        self.feature_tables["ccle"] = X_ccle

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
