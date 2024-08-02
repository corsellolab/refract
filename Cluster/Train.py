import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import shap
import copy

def add_noise(features, noise_level=0.01):
    noise = torch.randn_like(features) * noise_level
    return features + noise

def scale_features(features, scale_factor=0.1):
    scale = 1 + (torch.randn_like(features) * scale_factor)
    return features * scale

class ImprovedNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        
    def forward(self, x):
        out = self.relu1(self.fc1(x))
        out = self.relu2(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.features = torch.tensor(X.values, dtype=torch.float32)
        self.targets = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience=5):
    best_model = None
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        if len(val_loader) != 0:
            val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Check for early stopping
        if epochs_without_improvement >= patience:
            break

    return best_model

def train(cluster_data_folder, num_epochs=100):
    pretrain_csv = next((file for file in os.listdir(cluster_data_folder) if file.endswith('Pretrain.csv')), None)
    if pretrain_csv is None:
        raise FileNotFoundError("No file ending with 'Pretrain.csv' found in the specified folder.")
    
    cluster_data_path = os.path.join(cluster_data_folder, pretrain_csv)
    drug_files = [os.path.join(cluster_data_folder, file) for file in os.listdir(cluster_data_folder) if file != pretrain_csv]
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(cluster_data_path)
    df = df.drop_duplicates()
    cell_line_names = df.pop('ccle_name')
    print("Pretrained Dataframe Loaded")
    drugs = df.select_dtypes(exclude=['number']).columns
    ids = pd.concat([df.pop(x) for x in drugs], axis=1)
    ids = ids.astype(int)
    X = (df.iloc[:, 1:] - df.iloc[:, 1:].mean()) / df.iloc[:, 2:].std()
    X = pd.concat([X, ids], axis=1)
    features = list(X.columns)
    y = df.iloc[:, [0]]
    X.fillna(0, inplace=True)
    y.fillna(0, inplace=True)
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)
    input_size = X.shape[1]
    output_size = 1
    batch_size = 32
    learning_rate = 0.001
    n_splits = 5

    cluster_preds = np.zeros(len(y), dtype=float)
    cluster_shap = []
    finetune_preds = np.zeros(len(y), dtype=float)
    finetune_shap = {drug: [] for drug in drugs}
    single_drug_preds = np.zeros(len(y), dtype=float)
    single_drug_shap = {drug: [] for drug in drugs}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        train_dataset = CustomDataset(X_train, y_train)
        test_dataset = CustomDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        pretrained_model = ImprovedNN(input_size, 512, 256, output_size)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(pretrained_model.parameters(), lr=learning_rate)



        # Train the model
        pretrained_model = train_model(pretrained_model, train_loader, test_loader, criterion, optimizer, num_epochs)
        pretrained_state = pretrained_model.state_dict()

        # Compute SHAP values for the cluster model
        cluster_explainer = shap.DeepExplainer(pretrained_model, torch.tensor(X_train.values, dtype=torch.float32))
        cluster_shap_values = cluster_explainer.shap_values(torch.tensor(X_test.values, dtype=torch.float32))
        cluster_shap.append(np.abs(cluster_shap_values).mean(axis=0))

        # Predictions with the pretrained model
        cluster_preds[test_idx] = pretrained_model(torch.tensor(X_test.values, dtype=torch.float32)).detach().numpy().flatten()

        for drug in drugs:
            drug_test_X = X_test[X_test[drug] == 1]
    
            finetune_model = ImprovedNN(input_size, 512, 256, output_size)
            raw_model = ImprovedNN(input_size, 512, 256, output_size)
            finetune_model.load_state_dict(pretrained_state)
            for name, param in finetune_model.named_parameters():
                if name not in ['fc3.weight', 'fc3.bias']:
                    param.requires_grad = False
            finetune_optimizer = optim.AdamW(finetune_model.parameters(), lr=learning_rate)
            raw_optimizer = optim.AdamW(raw_model.parameters(), lr=learning_rate)

            # Filter data for the current drug
            X_finetune = X_train[X_train[drug] == 1]
            y_finetune = y_train.loc[X_finetune.index, :]

            train_dataset = CustomDataset(X_finetune, y_finetune)
            test_dataset = CustomDataset(drug_test_X, y_test.loc[drug_test_X.index, :])
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Fine-tune the model
            finetune_model = train_model(finetune_model, train_loader, test_loader, criterion, finetune_optimizer, num_epochs)
            raw_model = train_model(raw_model, train_loader, test_loader, criterion, raw_optimizer, num_epochs)

            # Compute SHAP values for fine-tuned and raw models
            finetune_explainer = shap.DeepExplainer(finetune_model, torch.tensor(X_finetune.values, dtype=torch.float32))
            raw_explainer = shap.DeepExplainer(raw_model, torch.tensor(X_finetune.values, dtype=torch.float32))

            finetune_shap_values = finetune_explainer.shap_values(torch.tensor(X_finetune.values, dtype=torch.float32))
            raw_shap_values = raw_explainer.shap_values(torch.tensor(X_finetune.values, dtype=torch.float32))

            finetune_shap[drug].append(np.abs(finetune_shap_values).mean(axis=0))
            single_drug_shap[drug].append(np.abs(raw_shap_values).mean(axis=0))

            # Predictions with fine-tuned and raw models
            finetune_preds[drug_test_X.index] = finetune_model(torch.tensor(drug_test_X.values, dtype=torch.float32)).detach().numpy().flatten()
            single_drug_preds[drug_test_X.index] = raw_model(torch.tensor(drug_test_X.values, dtype=torch.float32)).detach().numpy().flatten()

    # Compute average SHAP values for each drug
    cluster_shap = np.mean(cluster_shap, axis=0)
    average_finetune_shap = {drug: np.mean(finetune_shap[drug], axis=0) for drug in drugs}
    average_single_drug_shap = {drug: np.mean(single_drug_shap[drug], axis=0) for drug in drugs}

   
    cluster_shap = dict(zip(features, cluster_shap))
   
    dfs = {}
    preds = pd.DataFrame(zip(cluster_preds, finetune_preds, single_drug_preds, y['LFC.cb']), columns=['Cluster Level Prediction', 'Finetune Prediction', 'Raw Prediction', 'LFC.cb'], index=cell_line_names)
    cluster_shap_values = pd.DataFrame([cluster_shap]).transpose()
    cluster_shap_values.columns = ['Cluster']

    for key in average_finetune_shap.keys():
        drug_df = pd.DataFrame.from_records([average_finetune_shap[key], average_single_drug_shap[key]])
        drug_df = drug_df.transpose().reset_index()
        drug_df['index'] = features
        drug_df = drug_df.set_index('index')
        drug_df.columns = [f"Finetune {key}", f"Single Drug {key}"]
        dfs[key] = drug_df


    shap_values = cluster_shap_values.join(list(dfs.values()),how='inner')
    pearson_correlations = (preds.corrwith(preds['LFC.cb']))
    return pearson_correlations, preds, shap_values
   

def main():
    return

if __name__ == "__main__":
    main()
