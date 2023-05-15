"""Run model training
"""
import os
from datasets import ResponseSet, FeatureSet
from trainers import (
    RFBaseTrainer,
    NestedCVRFTrainer,
    NestedCVXGBoostTrainer,
    NestedCVLGBMTrainer,
)
from utils import RandomForestCVConfig, XGBoostCVConfig, LGBMCVConfig
import json
import argparse
import logging

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
        config = RandomForestCVConfig(**config)
    else:
        config = RandomForestCVConfig()

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
            logger.info(
                "    Training model for {} {} {}".format(pert_name, dose, pert_mfc_id)
            )

            trainer = NestedCVRFTrainer()
            trainer.train(
                response_set=response_set,
                feature_set=feature_set,
                config=config,
                output_dir=output_dir,
            )
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
