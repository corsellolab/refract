# Unit test to verify that the feature subset training works
import os
import unittest

from refract.datasets import FeatureSet, ResponseSet
from refract.trainers import CVRFTrainer
from refract.utils import RandomForestCVConfig

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))


class TestFeatureSubset(unittest.TestCase):
    ROOT_OUTPUT = os.path.join(REPO_ROOT, "tests/data/test_subset_output")

    def setUp(self):
        # load the test data
        self.response_set = ResponseSet("tests/data/test_responses.csv")
        self.response_set.load_response_table()
        self.feature_set = FeatureSet("tests/data/test_features")
        self.feature_set.load_concatenated_feature_tables()
        self.feature_set.load_individual_feature_tables()

        # get unique runs
        self.LFC = self.response_set.LFC
        runs = self.LFC[["pert_name", "dose", "pert_mfc_id"]].drop_duplicates()
        assert len(runs) == 1
        self.pert_name = runs.iloc[0]["pert_name"]
        self.pert_mfc_id = runs.iloc[0]["pert_mfc_id"]
        self.dose = runs.iloc[0]["dose"]

    def _verify_output(self, output_dir, feature_set_name="all"):
        output_basename = f"zotarolimus_BRD-K46843573-001-01-9_2.5_{feature_set_name}_"
        assert os.path.exists(output_dir)
        assert os.path.exists(output_dir + "/config.json")
        assert os.path.exists(
            output_dir + "/" + output_basename + "feature_importances.csv"
        )
        assert os.path.exists(output_dir + "/" + output_basename + "model_stats.csv")
        assert os.path.exists(output_dir + "/" + output_basename + "predictions.csv")
        assert os.path.exists(output_dir + "/" + output_basename + "trainer_result.pkl")

    def test_feature_subset_train(self):
        output_dir = os.path.join(self.ROOT_OUTPUT, "cv")
        config = RandomForestCVConfig()
        trainer = CVRFTrainer(
            pert_name=self.pert_name,
            pert_mfc_id=self.pert_mfc_id,
            dose=self.dose,
            feature_name="all",
            output_dir=output_dir,
            response_set=self.response_set,
            feature_set=self.feature_set,
            config=config,
        )
        trainer.train()
        # test output exists
        self._verify_output(output_dir)

        # test training on feature subsets
        feature_sets = [
            "GE",
            "CNA",
            "MET",
            "miRNA",
            "PROT",
            "XPR",
            "shRNA",
            "REP",
            "MUT",
            "LIN",
        ]
        for feature_set_name in feature_sets:
            # train
            trainer = CVRFTrainer(
                pert_name=self.pert_name,
                pert_mfc_id=self.pert_mfc_id,
                dose=self.dose,
                feature_name=feature_set_name,
                output_dir=output_dir,
                response_set=self.response_set,
                feature_set=self.feature_set,
                config=config,
            )
            trainer.train()
            # test output exists
            self._verify_output(
                output_dir,
            )

        # test output exists
        self._verify_output(output_dir)
