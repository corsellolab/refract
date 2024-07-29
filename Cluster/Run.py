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
from Analysis import *
import argparse
import scipy


def main(out_dir, cluster_number, storage_dir):
    print("Starting Data Processing")
    process_data(cluster_number,storage_dir + '/clusters200_with_viability.csv', storage_dir + '/processed_data/x-all.csv', storage_dir + '/responses', out_dir + '/Data')
    data_dir = out_dir + f"/Data/cluster_{str(cluster_number)}_individual_responses"
    print("Finished Data Processing")

    print("Starting Training")
    cluster_model, cluster_df, stats, dataframes = train(data_dir, num_epochs=50)

    if not os.path.exists(out_dir + "/Models"):
        os.mkdir(out_dir + "/Models")
    model_dir = out_dir + f"/Models/cluster_{str(cluster_number)}_models"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    torch.save(cluster_model, model_dir + f'/Cluster_{str(cluster_number)}_Pretrain.pt')
    

    if not os.path.exists(out_dir + "/Analysis"):
        os.mkdir(out_dir + "/Analysis")
    analysis_dir = out_dir + f"/Analysis/cluster_{str(cluster_number)}_analysis/"
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)
    
    with open(analysis_dir + "/Pearson_Correlations.txt", 'w') as f:
        for key, value in stats.items():
            f.write('%s:%s\n' % (str(key),str(value)))
    
    cluster_df.to_csv(analysis_dir + 'Cluster_Shap_Values.csv')
    for drug,pair in dataframes.items():
        pair[0].to_csv(analysis_dir + f'{drug}_Finetuned_Shap_Values.csv')
        pair[1].to_csv(analysis_dir + f'{drug}_Raw_Shap_Values.csv')



    # for key,value in models.items():
    #     torch.save(value[0],model_dir + f'/Drug_{str(key)}_Finetuned_Model.pt')
    #     torch.save(value[1], model_dir + f'/Drug_{str(key)}_Raw_Model.pt')
    # print("Finished Training")

    # print("Starting Analysis")
    # df = shap_analysis(load_analysis_data(num, data_dir, model_dir))
    # cluster_df = pd.read_csv(storage_dir + '/clusters200_with_viability.csv')
    # cluster_df = cluster_df[cluster_df['Cluster'] == cluster_number]
    # drug_names = list(cluster_df['pert_name'])
    # drug_names = [f'Cluster_{cluster_number}'] + drug_names
    # names = sorted(list(df.columns))
    # mapping = dict(zip(names,drug_names))
    # df=df.rename(columns=mapping)
    # df.to_csv(out_dir + f'/cluster_{str(cluster_number)}_top_features.csv')
    # print(df.head())
    # print("Finished Analysis")
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

