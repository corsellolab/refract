# Unit test to check nested cross-validation training
import unittest
import os
import sys
from ..refract.datasets import ResponseSet, FeatureSet
from ..refract.trainers import NestedCVRFTrainerNoRetrain
from ..refract.utils import RandomForestCVConfig

REPO_ROOT = sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)


class TestNestedCVNoRetrain(unittest.TestCase):
    output_dir = os.path.join(REPO_ROOT, "tests/data/test_output")

    def setUp(self):
        # load the test data
        self.response_set = ResponseSet("tests/data/test_responses.csv")
        self.response_set.load_response_table()
        self.feature_set = FeatureSet("tests/data/test_features")
        self.feature_set.load_concatenated_feature_tables()

        # get unique runs
        self.LFC = self.response_set.LFC
        runs = self.LFC[["pert_name", "dose", "pert_mfc_id"]].drop_duplicates()
        assert len(runs) == 1
        self.pert_name = runs.iloc[0]["pert_name"]
        self.pert_mfc_id = runs.iloc[0]["pert_mfc_id"]
        self.dose = runs.iloc[0]["dose"]

    def test_run_training(self):
        # test running a quick training workflow
        config = RandomForestCVConfig()
        trainer = NestedCVRFTrainerNoRetrain(
            pert_name=self.pert_name,
            pert_mfc_id=self.pert_mfc_id,
            dose=self.dose,
            feature_name="all",
            output_dir=self.output_dir,
            response_set=self.response_set,
            feature_set=self.feature_set,
            config=config,
        )
        trainer.train()
