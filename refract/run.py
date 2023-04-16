"""Run model training
"""
import os
from datasets import ResponseSet, FeatureSet
from trainers import RFBaseTrainer, NestedCVRFTrainer, NestedCVXGBoostTrainer
from utils import RandomForestCVConfig, XGBoostCVConfig
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
        config = XGBoostCVConfig(**config)
    else:
        config = XGBoostCVConfig()

    # train model
    trainer = NestedCVXGBoostTrainer()
    logger.info("Training model...")
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.response_path, args.feature_dir, args.output_dir, args.config_path)
