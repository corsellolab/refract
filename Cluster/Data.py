import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
import random as random
import glob
import os
import sys
import gc  # Import the garbage collector

def load_cluster_data(cluster_df, feature_df, cluster_id, responses_dir):
    cluster_df = cluster_df[cluster_df['Less than 30% Viable'] >= 2]
    cluster_df = cluster_df[cluster_df['Cluster'] == cluster_id]
    cluster_df = cluster_df.drop_duplicates(subset=['pert_name', 'MOA'])

    print(f"Processing cluster ID: {cluster_id}")
    print(f"Number of drugs in cluster: {len(cluster_df)}")

    data_frames = []
    drugs = []
    for drug in list(cluster_df['pert_name']):
        drugs.append(drug)
        for file in glob.glob(os.path.join(responses_dir, f'{drug}*.csv')):
            df = pd.read_csv(file)
            df = df[['LFC.cb', 'ccle_name']]
            df = df.merge(feature_df, on="ccle_name")
            df = df.drop(columns=['ccle_name'])
            df['ID'] = drug
            data_frames.append(df)
    
   
    

    pretrain_df = pd.concat(data_frames)
    one_hot = pd.get_dummies(pretrain_df['ID'])
    

    print("Getting correlations")
    corr = dict(pretrain_df.drop(columns=['ID']).corrwith(pretrain_df['LFC.cb']))
    top_features = sorted(corr, key=lambda k: abs(corr[k]), reverse=True)[:502]
    if 'ID' not in top_features:
        top_features = top_features[:501]
        top_features.append('ID')

    pretrain_df =pretrain_df.sample(frac=0.05)
    pretrain_df = pretrain_df[top_features]
    pretrain_df = pretrain_df.join(one_hot)
    pretrain_df = pretrain_df.dropna()

    grouped = pretrain_df.groupby('ID')
    grouped_dfs = [group for _, group in grouped]

    

    if len(grouped_dfs) > 10:
        sampled_dfs = random.sample(grouped_dfs, 10)
    else:
        sampled_dfs = grouped_dfs
    pretrain_df = pretrain_df.drop(columns=['ID'])
    filtered_data_frames = [df.drop(columns=['ID']) for df in sampled_dfs]
    del data_frames
    del sampled_dfs
    gc.collect()
    

    print(f"Number of data frames: {len(filtered_data_frames)}")
    return pretrain_df, filtered_data_frames, drugs

def process_data(cluster_number,cluster_data_path, feature_data_path, response_dir, output_dir):
    cluster_df = pd.read_csv(cluster_data_path)
    features = pd.read_csv(feature_data_path)

    for cluster in cluster_df.Cluster.unique():
        if int(cluster) == cluster_number:
            print(f"Starting processing for cluster: {cluster}")
            pretrain_df, data_frames, drugs = load_cluster_data(cluster_df, features, cluster, response_dir)
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            new_dir = os.path.join(output_dir, f'cluster_{cluster}_individual_responses')
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            print(f"Creating directory: {new_dir}")
            
            pretrain_df.to_csv(os.path.join(new_dir, f'Cluster_{cluster}_Pretrain.csv'))
            print(f"Saved pretrain data for cluster: {cluster}")
            
            i = 0
            while 0 < len(data_frames):
                data_frames[0].to_csv(os.path.join(new_dir, f'Cluster_{cluster}_{str(drugs[i])}.csv'))
                del data_frames[0]
                i+=1
                gc.collect()
            
            print(f"Processed cluster {cluster}")
            return
def main():
    return

if __name__ == "__main__":
    main()

