# Unit test to check nested cross-validation training
import os
import unittest

from refract.datasets import FeatureSet, ResponseSet
from refract.trainers import (
    CVRFTrainer,
    CVXGBoostTrainer,
    NestedCVRFTrainer,
    NestedCVRFTrainerNoRetrain,
    WeightedCVRFTrainer,
    WeightedNestedCVRFTrainer,
)
from refract.utils import (
    RandomForestCVConfig,
    RandomForestNestedCVConfig,
    WeightedRandomForestCVConfig,
    WeightedRandomForestNestedCVConfig,
    XGBoostCVConfig,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))


class TestNestedCV(unittest.TestCase):
    ROOT_OUTPUT = os.path.join(REPO_ROOT, "tests/data/test_output")

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

    def _verify_output(self, output_dir):
        output_basename = "zotarolimus_BRD-K46843573-001-01-9_2.5_all_"
        assert os.path.exists(output_dir)
        assert os.path.exists(output_dir + "/config.json")
        assert os.path.exists(
            output_dir + "/" + output_basename + "feature_importances.csv"
        )
        assert os.path.exists(output_dir + "/" + output_basename + "model_stats.csv")
        assert os.path.exists(output_dir + "/" + output_basename + "predictions.csv")
        assert os.path.exists(output_dir + "/" + output_basename + "trainer_result.pkl")

    def test_nested_cv_rf_train(self):
        # test running a quick training workflow
        output_dir = os.path.join(self.ROOT_OUTPUT, "nested_cv")
        config = RandomForestNestedCVConfig(
            param_grid={"n_estimators": [5, 10], "max_depth": [2, 3]},
            n_splits=4,
            n_jobs=4,
        )
        trainer = NestedCVRFTrainer(
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

    def test_nested_cv_weighted_rf_train(self):
        output_dir = os.path.join(self.ROOT_OUTPUT, "weighted_nested_cv")
        config = WeightedRandomForestNestedCVConfig()
        trainer = WeightedNestedCVRFTrainer(
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

    def test_nested_cv_rf_no_refit_train(self):
        # test running a quick training workflow
        output_dir = os.path.join(self.ROOT_OUTPUT, "nested_cv_no_refit")
        config = RandomForestNestedCVConfig(
            param_grid={"n_estimators": [5, 10], "max_depth": [2, 3]},
            n_splits=4,
            n_jobs=4,
        )
        trainer = NestedCVRFTrainerNoRetrain(
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

    def test_cv_rf_train(self):
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

    def test_cv_weighted_rf_train(self):
        output_dir = os.path.join(self.ROOT_OUTPUT, "weighted_cv")
        config = WeightedRandomForestCVConfig()
        trainer = WeightedCVRFTrainer(
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
        self._verify_output(output_dir)

    def test_cv_xgboost_train(self):
        output_dir = os.path.join(self.ROOT_OUTPUT, "xgboost_cv")
        config = XGBoostCVConfig()
        trainer = CVXGBoostTrainer(
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
        self._verify_output(output_dir)
