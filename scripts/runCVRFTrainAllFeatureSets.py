"""Run CV RF Model Training
Train on all feature sets individually
"""
# add parent dir to path
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import logging

from refract.datasets import FeatureSet, ResponseSet
from refract.trainers import CVRFTrainer
from refract.utils import RandomForestCVConfig

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
    logger.info("Loading individual feature tables...")
    feature_set.load_individual_feature_tables()

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
        pert_name = run["pert_name"]
        pert_mfc_id = run["pert_mfc_id"]
        dose = run["dose"]
        logger.info(
            "    Training model for {} {} {}".format(pert_name, dose, pert_mfc_id)
        )
        all_output_dir = os.path.join(output_dir, "all")
        trainer = CVRFTrainer(
            pert_name=pert_name,
            pert_mfc_id=pert_mfc_id,
            dose=dose,
            feature_name="all",
            output_dir=all_output_dir,
            feature_set=feature_set,
            response_set=response_set,
            config=config,
        )
        trainer.train()

    # train on each of the individual feature sets
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
        # create output dir
        feature_set_output_dir = os.path.join(output_dir, feature_set_name)
        if not os.path.exists(feature_set_output_dir):
            os.makedirs(feature_set_output_dir)
        logger.info("Using feature set {}".format(feature_set_name))

        for _, run in runs.iterrows():
            pert_name = run["pert_name"]
            pert_mfc_id = run["pert_mfc_id"]
            dose = run["dose"]
            logger.info(
                "    Training model for {} {} {}".format(pert_name, dose, pert_mfc_id)
            )

            trainer = CVRFTrainer(
                pert_name=pert_name,
                pert_mfc_id=pert_mfc_id,
                dose=dose,
                feature_name=feature_set_name,
                output_dir=feature_set_output_dir,
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
