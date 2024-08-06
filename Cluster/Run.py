import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
from sklearn.model_selection import KFold
import os
import glob
import copy
import shap
import gc
from Data import *
from Train import *
import argparse
import scipy


def main(out_dir, cluster_number, storage_dir):
    print("Starting Data Processing")
    process_data(cluster_number,storage_dir + '/clusters200_with_viability.csv', storage_dir + '/processed_data/x-all.csv', storage_dir + '/responses', out_dir + '/Data')
    data_dir = out_dir + f"/Data/cluster_{str(cluster_number)}_responses"
    print("Starting Training")
    pearson_correlations, preds, shap_values = train(data_dir, num_epochs=10000)

    if not os.path.exists(out_dir + "/Analysis"):
        os.mkdir(out_dir + "/Analysis")
    analysis_dir = out_dir + f"/Analysis/cluster_{str(cluster_number)}_analysis/"
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)
    
    with open(analysis_dir + "/Pearson_Correlations.txt", 'w') as f:
        for key, value in pearson_correlations.items():
            f.write('%s:%s\n' % (str(key),str(value)))
    
    preds.to_csv(analysis_dir + f'Cluster_{cluster_number}_Predictions.csv')
    shap_values.to_csv(analysis_dir + f'Cluster_{cluster_number}_Shap_Values.csv')
    return

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process directory name and cluster number.')
    parser.add_argument('output_dir', type=str, help='The name of the output directory')
    parser.add_argument('cluster_number', type=int, help='The cluster number')
    parser.add_argument('data_dir', type=str, help='The name of the data directory')

    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args.output_dir, args.cluster_number, args.data_dir)

