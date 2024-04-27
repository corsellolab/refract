import logging
import os
import torch
import sys
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import argparse

from refract.datasets import PrismDataset
from refract.models import get_model
from refract.trainers import train_model

logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

def run(
        response_path,
        feature_path,
        folds_path,
        output_dir,
        checkpoint_dir
):
    # create output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load data
    logger.info("Loading feature data...")
    with open(feature_path, "rb") as f:
        feature_df = pickle.load(f)

    logger.info("Loading response data...")
    response_df = pd.read_csv(response_path)

    logger.info("Loading ccle_name folds...")
    ccle_cv = pd.read_csv(folds_path)

    # split into train, val, test based on fold id
    train_folds = [0,1,2,3,4,5]
    val_folds = [6,7]
    test_folds = [8,9]

    train_ccle_names = ccle_cv[ccle_cv['fold'].isin(train_folds)]['ccle_name']
    val_ccle_names = ccle_cv[ccle_cv['fold'].isin(val_folds)]['ccle_name']
    test_ccle_names = ccle_cv[ccle_cv['fold'].isin(test_folds)]['ccle_name']

    # subset the response data
    logger.info("Subsetting response data...")
    response_df_train = response_df[response_df['ccle_name'].isin(train_ccle_names)]
    response_df_val = response_df[response_df['ccle_name'].isin(val_ccle_names)]
    response_df_test = response_df[response_df['ccle_name'].isin(test_ccle_names)]

    # subset the feature data
    logger.info("Subsetting feature data...")
    feature_df_train = feature_df.loc[train_ccle_names]
    feature_df_val = feature_df.loc[val_ccle_names]
    feature_df_test = feature_df.loc[test_ccle_names]

    # standard scale the input
    logger.info("Scaling features...")
    scaler = StandardScaler()
    feature_df_train = pd.DataFrame(scaler.fit_transform(feature_df_train), index=feature_df_train.index, columns=feature_df_train.columns)
    feature_df_val = pd.DataFrame(scaler.transform(feature_df_val), index=feature_df_val.index, columns=feature_df_val.columns)
    feature_df_test = pd.DataFrame(scaler.transform(feature_df_test), index=feature_df_test.index, columns=feature_df_test.columns)

    # get the drug encoder
    logger.info("Fitting drug encoder...")
    unique_drugs = response_df['pert_name'].unique()
    drug_encoder = LabelEncoder()
    drug_encoder.fit(unique_drugs)
    # save drug encoder to output path
    with open(os.path.join(output_dir, 'drug_encoder.pkl'), 'wb') as f:
        pickle.dump(drug_encoder, f)

    # create datasets
    logger.info("Creating datasets...")
    train_dataset = PrismDataset(feature_df_train, response_df_train, drug_encoder)
    val_dataset = PrismDataset(feature_df_val, response_df_val, drug_encoder)
    test_dataset = PrismDataset(feature_df_test, response_df_test, drug_encoder)

    # create dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=8)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=8)

    # get the model 
    logger.info("Getting Model...")
    model = get_model(input_dim=train_dataset.num_features, num_embeddings=train_dataset.num_embeddings, embedding_dim=10)

    # train
    logger.info("Training...")
    train_model(model, train_dataloader, val_dataloader, num_epochs=200, patience=10, chkpt_dir=checkpoint_dir)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--response_path", type=str, required=True)
    argparser.add_argument("--feature_path", type=str, required=True)
    argparser.add_argument("--folds_path", type=str, required=True)
    argparser.add_argument("--output_dir", type=str, required=True)
    argparser.add_argument("--checkpoint_dir", type=str, required=False, default="checkpoints")
    args = argparser.parse_args()
    run(args.response_path, args.feature_path, args.folds_path, args.output_dir, args.checkpoint_dir)