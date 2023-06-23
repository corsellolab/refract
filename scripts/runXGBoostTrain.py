"""Run nested CV RF model training
"""
# add parent dir to path
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import logging

from refract.datasets import FeatureSet, ResponseSet
from refract.rf_trainers import CVXGBoostTrainer
from refract.utils import XGBoostCVConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


def run(response_path, feature_dir, output_dir, config_path):
    # load data
    logger.info("Loading response data...")
    response_set = ResponseSet(response_path)
    response_set.load_response_table()
    logger.info("Loading feature data...")
    feature_set = FeatureSet(feature_dir)
    feature_set.load_concatenated_feature_tables()

    # if config, load config
    if config_path:
        with open(config_path, "r") as f:
            config = json.load(f)
        config = XGBoostCVConfig(**config)
    else:
        config = XGBoostCVConfig()

    # get the unique runs from the response set
    LFC = response_set.LFC
    runs = LFC[["pert_name", "dose", "pert_mfc_id"]].drop_duplicates()
    logger.info("Found {} unique runs".format(len(runs)))

    # train model for each unique run
    for _, run in runs.iterrows():
        for feature_name in args.feature_name_list:
            pert_name = run["pert_name"]
            pert_mfc_id = run["pert_mfc_id"]
            dose = run["dose"]
            if dose != 0.3704:
                continue
            logger.info(
                "    Training model for {} {} {}".format(pert_name, dose, pert_mfc_id)
            )

            trainer = CVXGBoostTrainer(
                pert_name=pert_name,
                pert_mfc_id=pert_mfc_id,
                dose=dose,
                feature_name=feature_name,
                output_dir=output_dir,
                feature_set=feature_set,
                response_set=response_set,
                config=config,
            )
            trainer.train()
    logger.info("done")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str)
    parser.add_argument("--response_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--config_path", type=str, required=False, default=None)
    parser.add_argument("--feature_name_list", type=str, nargs="+", default=["all"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.response_path, args.feature_dir, args.output_dir, args.config_path)
